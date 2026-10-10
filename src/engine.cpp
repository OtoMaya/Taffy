// engine.cpp - phase-vocoder time-stretch / pitch-shift engine.
//
// Path: input FIFO -> analysis window -> FFT -> phase vocoder (instantaneous
// frequency + identity phase locking) -> IFFT -> synthesis window -> WOLA
// overlap-add normalized by the window-square sum -> resample when pitch != 1.
//
// Time/pitch model: internalStretch = stretch * pitch; Hs = Ha / internalStretch
// (synthesis hop); then linear-interpolation resampling by pitch.
// Net effect: duration x stretch, pitch x pitch.
// No latency compensation is applied; callers must align the output themselves.

#include "taffy.h"
#include "fft.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <new>
#include <cstdio>
#include <cstdlib>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace {

inline float princarg(float p) {
    p = std::fmod(p + (float)M_PI, 2.0f * (float)M_PI);
    if (p < 0.0f) p += 2.0f * (float)M_PI;
    return p - (float)M_PI;
}

struct Chan {
    std::vector<float> inBuf;      // input FIFO
    std::vector<float> ola;        // overlap-add accumulator, length N
    std::vector<float> wsum;       // window-square sum accumulator, length N
    std::vector<float> lastPhase;  // N/2+1
    std::vector<float> sumPhase;   // N/2+1
    std::vector<float> mag;        // N/2+1
    std::vector<float> origPhase;  // N/2+1
    std::vector<float> re, im;     // N
    std::vector<float> hist;       // input history (transient source snippets)
    long long histBase = 0;        // global input index of hist[0]

    // transient detection state
    std::vector<float> prevBand;   // previous frame's sub-band magnitude sum
    float fluxMean = 0.0f, fluxVar = 0.0f;
    long long frames = 0;
    int sinceTransient = 1000;
    bool isTransient = false;
    float auxG = 1.0f;             // smoothed aux gate value (scales Pro's complementary swap)
    long long auxSinceCopy = 1000000;   // hops since the last main-path transient copy (gate source 1)
    /* High-band-only transient measure for the Pro aux gate: broadband flux is dominated
       by low-frequency drums and would keep aux closed, so only spectral flux above the
       crossover bin is summed (the criterion is per bin: "is this bin changing"). */
    std::vector<float> hiPrev;     // previous frame's per-bin high-band magnitude
    float hiFluxMean = 0.0f, hiFluxVar = 0.0f;
    float hiEnvMean = 0.0f;        // slow mean of high-band energy (gate source 2)
    float hiEnvPrevRatio = 0.0f;   // previous hop's ratio (rising-edge detection)
    /* Parallel OLA aux path: one frame of the input buffer is FFT'd, masked, IFFT'd and
       overlap-added with the same synthesis window as the main path, so it has no block
       boundary jumps and stays aligned with the main path. The main path already computes
       the input-frame spectrum, so no extra forward transform is needed. */
    std::vector<float> srcRe, srcIm;         // input-frame spectrum (before the phase advance)
    std::vector<float> magIn;                // input-frame magnitude (pre-envelope, for identity frames)
    /* Cumulative normalized spectral flux:
       cum[b] = sum_{j<=b}(mag[j]-prev[j])^2 / sum_{j<=b} prev[j]^2;
       cum[bin] < 0.3 decides whether aux takes over that bin. */
    std::vector<float> prevMag;              // previous frame's per-bin magnitude
    std::vector<float> fluxGate;             // per-bin gate (1 = aux may take over)
    float fluxScale = 1.0f;                  // this frame's flux gate, folded into blk.sc
    int transHold = 0;                       // remaining hops of the transient phase reset
    float magEnvMean = 0.0f;                 // slow envelope for the spectral-peak magnitude threshold
    float peakCov = 1.0f;                    // coverage of the peak-band mask
    std::vector<float> statB;                // per-bin signed phase deviation (stationarity test + segmentation)

    std::vector<float> auxHpOla, auxHpWsum;  // OLA accumulator of HP(input)
    std::vector<float> mainHpOla, mainHpWsum;// OLA accumulator of HP(main path)
    std::vector<float> re2, im2, re3, im3;   // scratch spectra for the HP(main) and HP(aux) frames
    int hpDbg = 0;                 // diagnostic flag (TAFFY_HPDBG)
    float hiEnvRatio = 0.0f;       // current ratio (hysteresis)
    unsigned char hiEnvArmed = 1;  // hysteresis armed flag
    /* Gate delay-alignment ring: a transient detected in the analysis frame appears in
       the output at `h*Hs + N/2`, so the flag is delayed by auxLag hops before use. */
    unsigned char gateRing[64] = {};
    int gateRingPos = 0;
    long long sinceGate = 1000000;   // hops since the last delayed transient
    long long gateTot = 0, gateClosed = 0;   // diagnostics: gate duty cycle
    int sinceHiTransient = 1000;
    bool hiTransient = false;

    /* Pure-SOLA state (Soloist tier): the SOLA and phase-vocoder paths are mutually
       exclusive, so Soloist output comes entirely from granular overlap-add. Indexing
       matches ola/wsum: the accumulator is W = 2*Hs long, each step emits the first Hs
       samples and shifts left by Hs, mirroring the WOLA emit semantics. */
    std::vector<float> solaAcc;     // overlap-add accumulator, length W
    std::vector<float> solaWsum;    // window-square sum, length W
    std::vector<float> solaTail;    // previous grain's [Hs,W) segment (unwindowed), for WSOLA alignment
    std::vector<float> solaWin;     // grain window, length W
    long long solaInPos = 0;        // nominal input start of the current grain, relative to inBuf[0]
    int solaSearch = 0;             // alignment search half-width from the pitch-mark spacing; 0 = default Hs/2
    int solaWinW = -1;              // grain length solaWin was built for (rebuilt when grains vary)
    float solaTonality = 0.0f;      // periodicity confidence (normalized cross-correlation peak)
};

// file scope, so pv_step needs no function-local static
static std::vector<Chan> g_unused;   // placeholder, not used

} // namespace

struct taffy_engine {
    int framesOut = 0;
    int channels = 0;
    int mode = 0;
    float sampleRate = 44100.0f;

    /* Structural differences between the three tiers:
         - Efficient: pure phase vocoder -- no pitch-mark analysis, no SOLA analysis or
           synthesis layer, so this tier has no time-domain SOLA overlay.
         - Pro: an embedded Efficient core plus pitch-mark analysis, a polyphonic SOLA
           analysis/synthesis overlay (mix coefficient 0.7) and a split filterbank at
           1500 Hz.
         - Soloist: pitch marks plus SOLA analysis/synthesis only -- no phase vocoder,
           no resampler, no split filterbank. Modes 3 and 4 differ only in the analysis
           window passed to the SOLA analysis enhancer (2048 vs 0). */
    enum SolaTier {
        SOLA_NONE = 0,   /* Efficient / Eff-mobile: pure phase vocoder, no time-domain overlay */
        SOLA_POLY = 1,   /* Pro: phase vocoder + polyphonic SOLA overlay */
        SOLA_MONO = 2    /* Soloist: pure SOLA, mono, pitch-mark driven */
    };
    int solaTier = SOLA_POLY;

    /* Whether time-domain transient copying is enabled, kept separate from transient
       detection/suppression: suppression exists in every tier, whereas the copy lives in
       the SOLA synthesizer and so only exists in Pro/Soloist. The engine exposes them as
       `ev.sup` (suppression, all tiers) and `ev.gain` (copy, per tier). */
    bool transientCopy = true;
    /* Pro's sample-by-sample PV/SOLA mix weight. Default 0 (off): measurements show
       mix=0 is optimal or tied on every test signal, since the two paths have different
       fine phase and summing them combs the spectrum. */
    float solaMix = 0.0f;
    bool  solaPrimed = false;   /* Soloist block pull: whether the first full block has been pulled */
    bool  grainVar = false;     /* grain length driven by pitch-mark spacing (TAFFY_GRAINVAR=1) */

    /* Pro's split spectral-envelope stage, inserted in pv_step between the FFT and the
       phase advance for bins >= splitBin. It receives already-transformed spectra and
       applies a cepstral envelope (lifter <= 16), multiplies the complex spectrum by the
       real envelope and can zero a band. Geometry: FFT 1024, hop 512,
       split bin = floor(1024*1500/sr) = 34. */
    /* Default off: the pure envelope-multiplication model was disproven -- it lifts the
       vowel band but does nothing for noise, so a frequency-dependent component is
       still missing. */
    bool  proEnv = false;       /* Pro only; TAFFY_PROENV=1 enables it for experiments */
    /* Pro's aux path: putting its crossover at Nyquist collapses the Pro/Efficient
       spectral distance, so the whole difference comes from this path. Crossover =
       (1500/sr)*N (139 @ N=4096 = 1496.6 Hz); Efficient never sets it, leaving it at
       Nyquist, i.e. disabled. Above the crossover there is both frequency-dependent gain
       and phase re-steering. Implemented here as: no identity phase locking above the
       crossover. */
    int   proAux = 0;           /* 0=off; 1=no locking above crossover; 2=analysis phase above crossover (TAFFY_PROAUX) */
    int   auxCrossBin = -1;
    int   auxFcBin = -1;                     /* crossover used only by the aux mask, decoupled from the transient detector */
    /* Pitch-mark re-estimation interval in steps; mode 4 (Soloist speech) uses 4x the
       mode-3 value. */
    int   markEvery = 16;
    /* How Pro combines the two aux paths: the per-bin aux decision
           if (bin >= auxCrossOverBin && state170[bin] != 3 && state190[bin] != 2
               && transient210[bin] < 0.3f)
               { mask1[bin] = 1; mask2[bin] = 0; }      // aux takes the bin
       falls back to switching the whole [139, 2048) range when the coverage
       (cutOff - aux - v86)/(cutOff - aux) drops below 0.25. So aux is a complementary
       switch between two independent synthesis paths, not a gain/EQ/phase tweak:
       mask=1 bins come from the aux (SOLA) synthesis and mask=0 bins from the main PV,
       and the two are summed. This engine realizes it as
       out = LP(PV) + HP(SOLA) = PV + HP(SOLA - PV); both paths share the same group
       delay and the LP/HP split is complementary, so it is a per-bin replacement. */
    int proAuxMode = -1;   /* -1 = per-tier default; 0=off; 1=mix; 2=full-band sum; 3=high-band sum; 4=complementary replace */
    /* Transient gate for aux: the aux decision includes `transient210[bin] < 0.3f`, so
       aux only acts where there is no transient. This shapes the per-signal aux
       difference, so the gate is a main component of the target, not decoration. */
    float auxGateTransient = 0.0f;   /* aux gain at transients (TAFFY_AUXGATET; default 0 = fully closed) */
    float auxGateSm = 0.5f;          /* first-order smoothing coefficient of the gate (TAFFY_AUXGATESM) */
    /* Calibrated jointly on two independent targets; these are fitted values, not
       constants read out of the decision logic. auxGain is the most suspect: the
       per-signal optimum is inconsistent (noise ~0.75, drums ~2), so it acts as a proxy
       for the aux synthesizer's own level in those bins. */
    int   auxHold = 4;               /* hops the gate stays closed after a transient (TAFFY_AUXHOLD, calibrated) */
    float auxGateK = 2.0f;           /* high-band energy ratio threshold (TAFFY_AUXGATEK) */
    float auxGain = 1.45f;           /* aux contribution gain (TAFFY_SOLAG), recalibrated after zero-phase + windowing */
    double auxEnvMean = 0.0;         /* slow envelope for the aux level gate (mode 8) */
    bool anaCopy = false;            /* analysis-domain transient copying (TAFFY_ANACOPY) */
    int transHoldHops = 0;           /* length in hops of the transient phase reset (TAFFY_THOLD; default 0) */
    /* HP(delta) must use the same frame-level OLA operator as HP(main), otherwise
       HP(main) + HP(delta) != HP(actual). A frame [base, base+N) needs
       N/Hs - 1 = 3 further hops of delta, so the whole delta OLA lags 4 hops and is
       written back through global indices (a stored base would go stale once pvTrimmed
       moves). */
    struct AuxPend {
        long long gBase = 0;          /* global output index */
        std::vector<float> d;         /* HP(aux) - HP(main) */
        int cov = 0;
        float g = 1.0f;               /* gate gain computed at enqueue time */
    };
    std::vector<std::vector<float>> auxDeltaHist;      /* [ch][4*Hs] delta of the last 4 hops */
    std::vector<std::vector<float>> auxDHpOla, auxDHpWsum;   /* [ch][N] frame-level OLA of delta */
    std::vector<std::vector<float>> auxDRe, auxDIm;    /* [ch][N] scratch spectra */
    std::vector<std::vector<AuxPend>> auxPend;         /* [ch] pending blocks */
    bool auxDrain = false;                             /* drain during flush */
    bool auxPipe = false;            /* frame-level HP(delta) pipeline (TAFFY_AUXPIPE, default off) */
    std::vector<std::vector<float>> auxDeltaPrev;  /* previous hop's copy correction (old block operator, kept for comparison) */
    bool auxActual = false;          /* use the actual output as the replacement reference (TAFFY_AUXACTUAL) */
    double auxLvlDb = -200.0;        /* input silence gate threshold in dB; Pro default -60 */
    int auxSilRun = 0;               /* consecutive silent hops */
    std::vector<float> kurtRing;     /* input ring buffer for the kurtosis gate */
    long long kurtPos = 0;
    std::vector<float> auxSilSaved;  /* backup taken before clearing the single-hop criterion */
    std::vector<float> auxDutyRing;  /* hop-level ring buffer for the transient-sparsity gate */
    long long auxDutyPos = 0;
    int   auxLag = getenv("TAFFY_AUXLAG") ? atoi(getenv("TAFFY_AUXLAG")) : 3;
                                     /* gate delay in hops (aligns the OLA's N/2 offset) */
    int   auxGateSrc = getenv("TAFFY_AUXGATESRC") ? atoi(getenv("TAFFY_AUXGATESRC")) : 1;
                                     /* 0 = high-band spectral flux; 1 = main-path transient copy span (default) */
    /* delay line of the linear-phase FIR high-pass, one per channel */
    /* This hop's aux correction (already high-passed) and the pv buffer position it
       belongs to; filled by pv_step_with_sola, added by apply_aux after
       apply_transients. One taffy_process call may run several hops while
       apply_transients runs once per call, so the corrections are queued -- keeping only
       the last block would drop all earlier hops' corrections. */
    struct AuxBlk { size_t base = 0; int cov = 0; std::vector<float> d, snap;             int skip = 0;                 /* skip this hop entirely (gate fully closed) */
            float sc = 1.0f;              /* gate scale applied to the whole contribution */
            float lvm = 1.0f;             /* level-neutral replacement ratio, applied to the whole contribution */
            float alpha = 1.0f;           /* partial mix coefficient */
        };
    std::vector<std::vector<AuxBlk>> auxQ;    /* [channel][block] */
    /* Previous hop's d: the zero-phase band split uses overlap-save, so two hops of d
       are needed for a wrap-around-free result in the middle of each hop. */
    std::vector<std::vector<float>> auxDPrev;
    bool auxPending = false;
    /* aux suppression windows per event type, in samples relative to outPos (aux_sup_covered) */
    int auxSupBurstPre  = getenv("TAFFY_AUXSUPBPRE")  ? atoi(getenv("TAFFY_AUXSUPBPRE"))  : 1024;
    int auxSupBurstPost = getenv("TAFFY_AUXSUPBPOST") ? atoi(getenv("TAFFY_AUXSUPBPOST")) : 0;
    int auxSupSpikePre  = getenv("TAFFY_AUXSUPSPRE")  ? atoi(getenv("TAFFY_AUXSUPSPRE"))  : 1024;
    int auxSupSpikePost = getenv("TAFFY_AUXSUPSPOST") ? atoi(getenv("TAFFY_AUXSUPSPOST")) : 0;
    /* 1 = cancel per sample from the samples apply_transients rewrote (off by default). */
    int auxExact = getenv("TAFFY_AUXEXACT") ? atoi(getenv("TAFFY_AUXEXACT")) : 0;
    /* 1 = recent-event ring: keeps spikes of already-removed events active (off by default). */
    int auxRingOn = getenv("TAFFY_AUXRING") ? atoi(getenv("TAFFY_AUXRING")) : 0;
    /* Events vanish from `pending` once apply_transients consumes them, but OLA energy
       lands at outPos and later: stashing outPos/type here lets coverage see both. */
    struct AuxEv { long long outPos = 0; int dmap = 0; };
    AuxEv auxRing[64];
    int   auxRingPos = 0;

    int   envOrder = 16;        /* cepstral lifter length in bins, max 16 */
    float envAlpha = -1.0f;     /* envelope power: -1 = divide (flatten), +1 = multiply (emphasize) */
    float envMix = 1.0f;        /* 0 = bypass this stage */

    float stretch = 1.0f;
    float pitch = 1.0f;
    float internalStretch = 1.0f;

    /* Formant correction stage (Soloist only). The output resampler moves the spectral
       envelope along with the pitch; this stage estimates the envelope of the resampled
       output, warps it back by 1/pitch, and re-applies the fine structure, so the pitch
       stays shifted while the formants stay where they were. */
    std::vector<std::vector<float>> fcIn;     /* [channel] pending input samples */
    std::vector<std::vector<float>> fcAcc;    /* [channel] overlap-add accumulator */
    std::vector<float> fcWsum;                /* window square sum for the accumulator */
    std::vector<float> fcWin;                 /* analysis/synthesis window */
    long long fcEmitted = 0;                  /* samples already emitted */
    int fcN = 0;                              /* FFT size, 0 = stage disabled */

    int N = 2048;
    int Ha = 512;
    int Hs = 512;

    std::vector<float> win;
    std::vector<float> winAux;      /* aux-only window (Hamming); main path unaffected */
    std::vector<float> omega;

    std::vector<Chan> ch;

    std::vector<std::vector<float>> pv;   // phase-vocoder output (planar)
    std::vector<size_t> pvRead;

    size_t inRead = 0;      // input read position, relative to inBuf[0]
    double rsPhase = 0.0;   // resample phase

    /* Kaiser-windowed sinc polyphase resampler: beta = 9.0, table density 256,
       taps = 2*a1+2, cutoff = 0.9+0.0125*a1. */
    static const int RS_SUB = 1024;   // fractional-position resolution
    std::vector<float> rsTable;       // [RS_SUB][taps]
    int rsTaps = 0;
    float rsCutoff = 0.95f;
    bool  rsBypass = false;
    /* stretch == 1 && pitch == 1: straight copy, no WOLA fade-in. */
    bool  pvBypass = false;
    void build_rs_table();

    // diagnostics
    long long stepsDone = 0;

    // global input/output accounting (locates transient time-domain copies)
    long long inFed = 0;        // total input samples fed
    long long pvEmitted = 0;    // total pv samples produced
    size_t pvTrimmed = 0;       // samples dropped from the front of pv[]

    // time-domain transient copy
    float transientGain = 1.35f;   // copy gain (calibrated on an impulse response)
    float transientSupGain = 0.10f;  // residual amplitude while suppressing (TAFFY_TSUPG)
    /* SOLA grain path: enabled by default, but only for isolated-spike transients (see the
       frac test in confirm_events). The 856-sample pre-cover is the smear span
       (N/Ha-1)(Hs-Ha) at stretch=1.5; burst transients are excluded (a net loss for drums).
       TAFFY_SOLA overrides, TAFFY_SOLA=0 disables. */
    int   transientSolaPre = 856;
    int   transientSolaPost = 0;   // post-cover length (default 0)
    /* Grain length 448 (was 1024): the optimum range for drums is 384..896, and this is
       also the fallback used before pitch marks are ready, so one value keeps early and
       later transients consistent. */
    int   transientSolaW = 448;   // grain length (TAFFY_SOLAW)
    /* Burst-transient cover region (TAFFY_SOLABPRE / TAFFY_SOLABPOST): bursts need both
       the pre-echo and the following decay covered; <= 0 disables it for bursts. */
    /* Burst direct map: read the input straight through a stretch mapping over
       [outPos-pre, outPos+post] with no grain overlap, which removes grain duplication
       outright. Enabled by TAFFY_BURSTMAP=1. */
    /* Burst "pre-side only" direct map, enabled by default: it cuts the pre-region
       residue by an order of magnitude and improves or ties the spectral distance at
       every stretch setting, leaving the transient body itself unchanged. */
    bool  burstDirectMap = getenv("TAFFY_BURSTMAP") ? (atoi(getenv("TAFFY_BURSTMAP")) != 0) : true;
    /* Spike direct map: covers only the region before the impulse, sized by the actual
       smear span. Improves every alignment-independent criterion (residue, RMS,
       sharpness). */
    bool  spikeDirectMap = getenv("TAFFY_SPIKEMAP") ? (atoi(getenv("TAFFY_SPIKEMAP")) != 0) : true;
    int   burstMapPre  = getenv("TAFFY_BMAPPRE")  ? atoi(getenv("TAFFY_BMAPPRE"))  : 1024;
    int   burstMapPost = getenv("TAFFY_BMAPPOST") ? atoi(getenv("TAFFY_BMAPPOST")) : 0;
    /* The spike map covers only before the impulse (default 0): the residue ends up far
       away, and the near field is better left to the PV. */
    int   spikeMapPost = getenv("TAFFY_SPIKEPOST") ? atoi(getenv("TAFFY_SPIKEPOST")) : 0;
    int   transientSolaBurstPre = 0;
    int   transientSolaBurstPost = 0;
    /* At compression settings (internalStretch < 1, i.e. Ha > Hs) whole-frame OLA
       necessarily misplaces transients, so the WSOLA grain path is the structural fix. */
    bool  burstGrainOnCompress = getenv("TAFFY_BURSTCOMP") ? (atoi(getenv("TAFFY_BURSTCOMP")) != 0) : true;
    /* Also use the WSOLA grain path when stretching (TAFFY_BURSTALL=1), but only for
       pitch == 1: with pitch != 1 the resampler is active and grain placement in the pv
       domain needs separate handling. */
    bool  burstGrainOnStretch = getenv("TAFFY_BURSTALL") ? (atoi(getenv("TAFFY_BURSTALL")) != 0) : true;
    int   transientSuppress = 0;  // PV suppression half-width before a transient, to remove
                                 // pre-echo (TAFFY_TSUP). Off by default: it helps impulses but
                                 // damages drums, and the two are not distinguishable here.
    int   transientWin = 160;      // transient replacement half-width in samples
    bool  transientResample = false; // resample the copy onto the output time axis

    float envPrev = 0.0f;
    long long lastTransientIn = -1000000;
    long long scanPos = 0;      // transient scan cursor, independent of inFed
    /* Transient segment copied directly after resampling onto the output time axis:
     * pv[op+j] = w(j) * gain * in[tp + j/stretch] + (1-w(j)) * pv[op+j]
     * For an impulse this replaces the vocoder smear with silence; for a drum hit it
     * stretches the input decay envelope onto the output time axis. */
    struct TEvent {
        long long outPos = 0;
        int win = 0;
        long long tp = 0;   // transient position on the global input axis
        int sup = 0;   // PV suppression half-width (0 = off)
        long long supLo = 0;   // left suppression cursor (advanced incrementally so it is
                               // not consumed first by resample_out)
        long long supHi = 0;   // right suppression cursor
        long long grainQ = 0;   // SOLA grain cursor (output start of the next grain)
        long long grainIn = 0;   // matching input start (nominal, before WSOLA)
        long long grainSearch = 0;   // actual input start after WSOLA alignment
        int grainVerbatim = 0;   // 1 = verbatim sample copy (true SOLA, keeps pitch)
        int directMap = 0;   // 1 = burst direct map (plain stretch mapping, no grain overlap)
        long long mapQ = 0;   // direct-map cursor
        int mapPre = 0, mapPost = 0;   // map region relative to outPos (must be initialized)
        int mapNoRightTaper = 0;   // 1 = no fade at the right edge (bursts need it, spikes must not)
        double grainInAcc = 0;   // accumulated input advance in equal-hop mode
        int absGrid = 0;   // 1 = grain grid anchored at input 0 (phase-locked to the signal)
        int markPeriod = 0;   // pitch-mark spacing in the input domain; 0 = not extracted
        long long confirmAt = 0;   // confirmation time (global input index); 0 = confirmed
        int confirmed = 0;
        int grainW = 0;   // grain length
        int solaPre, solaPost;     // SOLA cover region relative to outPos
        long long inBase = 0;   // global input index of src[0]
        float gain = 0;
        float stretch = 0;
        bool  resample = false;
        std::vector<std::vector<float>> src;
    };
    std::vector<TEvent> pending;

    // drain state
    bool flushing = false;
    long flushRemaining = 0;   // output frames still to produce
};

namespace {

inline double bessel_i0(double z) {
    double sum = 1.0, term = 1.0, z2 = z * z * 0.25;
    for (int k = 1; k < 64; k++) {
        term *= z2 / ((double)k * (double)k);
        sum += term;
        if (term < 1e-16 * sum) break;
    }
    return sum;
}

} // namespace

/* Kaiser-windowed sinc polyphase resampling kernel:
 *   sinc(x)   = sin(pi*x*cutoff)/(pi*x)
 *   window(x) = I0(beta*sqrt(1-(x/(n-1))^2)) / I0(beta)
 *   taps = 2*a1+2, cutoff = 0.9+0.0125*a1, beta = 9.0, table density 256.
 * When downsampling, the kernel is widened proportionally and the cutoff lowered to
 * avoid aliasing. */
void taffy_engine::build_rs_table() {
    static const int A1 = getenv("TAFFY_RSA1") ? atoi(getenv("TAFFY_RSA1")) : 2;
    double scale = pitch > 1.0f ? (double)pitch : 1.0;
    int taps = (int)std::lround((2.0 * A1 + 2.0) * scale);
    if (taps < 4) taps = 4;
    if (taps > 64) taps = 64;
    if (taps % 2) taps++;
    rsTaps = taps;
    rsBypass = std::fabs(pitch - 1.0f) < 1e-6f;
    pvBypass = rsBypass && std::fabs(internalStretch - 1.0f) < 1e-6f;
    float cutoff = getenv("TAFFY_RSCUT") ? (float)atof(getenv("TAFFY_RSCUT"))
                                      : (0.9f + 0.0125f * (float)A1);
    cutoff /= (float)scale;
    if (cutoff > 1.0f) cutoff = 1.0f;
    rsCutoff = cutoff;
    const double beta = 9.0;
    const double i0b = bessel_i0(beta);
    double half = taps / 2.0;
    rsTable.assign((size_t)RS_SUB * (size_t)taps, 0.0f);
    for (int s = 0; s < RS_SUB; s++) {
        double frac = (double)s / RS_SUB;
        double sum = 0;
        for (int t = 0; t < taps; t++) {
            double x = frac - ((double)t - half + 1.0);
            double h;
            if (std::fabs(x) < 1e-12) h = cutoff;
            else h = std::sin(M_PI * x * cutoff) / (M_PI * x);
            double u = x / half;
            if (std::fabs(u) >= 1.0) h = 0.0;
            else h *= bessel_i0(beta * std::sqrt(1.0 - u * u)) / i0b;
            rsTable[(size_t)s * taps + t] = (float)h;
            sum += h;
        }
        if (sum != 0.0 && !getenv("TAFFY_RSNONORM"))
            for (int t = 0; t < taps; t++) rsTable[(size_t)s * taps + t] /= (float)sum;
    }
}

namespace {

// Phase vocoder: stretch = Hs / Ha (Hs > Ha means longer). Standard approach: fixed
// synthesis hop, varying analysis hop. Hs = N/4 gives 4x overlap for a smooth OLA;
// Ha = round(Hs / internalStretch), clamped to [1, N/2].
void rebuild_hopped(taffy_engine *e) {
    float is = e->internalStretch;
    if (is <= 0.0f) is = 1.0f;
    int  Hs = std::max(1, e->N / 4);
    long ha = std::lround((double)Hs / (double)is);
    if (ha < 1) ha = 1;
    /* Ha must not exceed N/2 or analysis frames stop overlapping, but it cannot simply be
       clamped: below internalStretch 0.5 that would silently turn every request into 0.5.
       Instead Hs is scaled down by the same ratio so Hs/Ha == internalStretch still holds;
       the resulting larger N/Hs overlap is smoother and the window sum accumulates
       accordingly. */
    const long haMax = e->N / 2;
    if (ha > haMax) {
        long hs2 = std::lround((double)haMax * (double)is);
        if (hs2 < 1) hs2 = 1;
        Hs = (int)hs2;
        ha = haMax;
    }
    e->Hs = Hs;
    e->Ha = (int)ha;
}

// Pre-seed the steady-state window-square sum used by WOLA normalization; it satisfies
// the fixed point w[i] = sum over k with i+k*Hs < N of win[i+k*Hs]^2.
void prime_wsum(taffy_engine *e) {
    for (int c = 0; c < e->channels; c++) {
        Chan &k = e->ch[(size_t)c];
        std::fill(k.ola.begin(), k.ola.end(), 0.0f);
        /* By default the steady-state sum is NOT pre-seeded: the numerator and the divisor
           accumulate together, so with a divisor floor the startup segment does not fade
           in. Pre-seeding makes the divisor full while the numerator is still growing,
           which is exactly what causes that fade. */
        if (!(getenv("TAFFY_PRIME") && atoi(getenv("TAFFY_PRIME")) != 0)) {
            std::fill(k.wsum.begin(), k.wsum.end(), 0.0f);
            continue;
        }
        for (int i = 0; i < e->N; i++) {
            float s = 0.0f;
            for (int j = i; j < e->N; j += e->Hs) s += e->win[(size_t)j] * e->win[(size_t)j];
            k.wsum[(size_t)i] = s;
        }
    }
}

void build_window(taffy_engine *e) {
    e->win.resize((size_t)e->N);
    for (int i = 0; i < e->N; i++)
        e->win[(size_t)i] = 0.5f * (1.0f - std::cos(2.0f * (float)M_PI * i / e->N));
    /* Aux-only window: the aux path uses Hamming coefficients 0.538 / 0.462 while e->win
       is Hann 0.5/0.5. The window is applied twice (before the FFT and after the IFFT), so
       its shape matters. A separate copy is required: changing e->win would change the
       main path too and therefore the Efficient tier. */
    e->winAux.resize((size_t)e->N);
    {
        static const int WH = getenv("TAFFY_WINH") ? atoi(getenv("TAFFY_WINH")) : 0;   /* default Hann; Hamming was no net gain */
        for (int i = 0; i < e->N; i++) {
            const double c = std::cos(2.0 * M_PI * (double)i / (double)e->N);
            e->winAux[(size_t)i] = WH ? (float)(0.538 - 0.462 * c)
                                      : e->win[(size_t)i];
        }
    }
    e->omega.resize((size_t)e->N / 2 + 1);
    for (int k = 0; k <= e->N / 2; k++)
        e->omega[(size_t)k] = 2.0f * (float)M_PI * k / e->N;
}

/* One phase-vocoder step: consumes Ha input samples, appends Hs samples to pv[]. */
/* Aux band mask H(f): 0 below the crossover, 1 above, with a smooth transition. It
   multiplies the spectrum by a per-bin real mask applied to the complete analysis frame,
   so frame edges are handled by the OLA window and no hard-truncation high-band artifacts
   appear. */
static const std::vector<float> *aux_mask(int N, double sr, double fc) {
    static std::vector<float> m;
    static int cN = -1; static double csr = -1.0, cfc = -1.0;
    static const int LENV0 = getenv("TAFFY_MASKL") ? atoi(getenv("TAFFY_MASKL")) : 513;
    static int cL = -1;
    if (cN == N && csr == sr && cfc == fc && cL == LENV0) return &m;
    cL = LENV0;
    /* mask = 1 - |H_LP(f)|, where H_LP is a windowed-sinc FIR low-pass. A frequency-domain
       step would have a long sinc tail whose circular convolution over an N-point frame
       leaks the low band back in (only -12 dB stopband), and the differing residual phases
       of the two HP terms would then boost it. Using the FIR response instead gets the
       Hamming window's inherent -53 dB stopband. */
    static const int LENV = getenv("TAFFY_MASKL") ? atoi(getenv("TAFFY_MASKL")) : 513;
    const int L = LENV;
    std::vector<float> h((size_t)L, 0.0f);
    {
        const double fcn = fc / sr;                     /* normalized cutoff, cycles/sample */
        double sum = 0.0;
        for (int i = 0; i < L; i++) {
            const double t = (double)i - 0.5 * (double)(L - 1);
            const double sv = (std::fabs(t) < 1e-12) ? 2.0 * fcn
                                                     : std::sin(2.0 * M_PI * fcn * t) / (M_PI * t);
            const double w = 0.54 - 0.46 * std::cos(2.0 * M_PI * (double)i / (double)(L - 1));
            h[(size_t)i] = (float)(sv * w);
            sum += sv * w;
        }
        for (int i = 0; i < L; i++) h[(size_t)i] /= (float)sum;   /* normalize DC gain to 1 */
    }
    /* |H_LP(f_b)| on the N-point grid: zero-pad to N, FFT, take the magnitude */
    {
        std::vector<float> hr((size_t)N, 0.0f), hi((size_t)N, 0.0f);
        for (int i = 0; i < L && i < N; i++) hr[(size_t)i] = h[(size_t)i];
        taffy::fft(hr, hi, false);
        m.assign((size_t)N, 0.0f);
        for (int b = 0; b < N; b++) {
            const double mag = std::sqrt((double)hr[(size_t)b] * hr[(size_t)b] +
                                         (double)hi[(size_t)b] * hi[(size_t)b]);
            float v = (float)(1.0 - mag);
            if (v < 0.0f) v = 0.0f;
            if (v > 1.0f) v = 1.0f;
            m[(size_t)b] = v;
        }
        if (getenv("TAFFY_AUXMASK0")) for (int b = 0; b < N; b++) m[(size_t)b] = 0.0f;
        /* Binary-mask variant: hard edge at the crossover bin (mask = 1 iff
           bin in {0, 1} u [xb, N-xb), with no filter). */
        if (getenv("TAFFY_MASKBIN")) {
            const int xb2 = (int)(fc * (double)N / sr);
            for (int b = 0; b < N; b++) {
                const int bb = (b <= N / 2) ? b : (N - b);
                m[(size_t)b] = (bb >= xb2) ? 1.0f : 0.0f;
            }
            m[0] = 1.0f; m[1] = 1.0f;          /* DC and the first bin stay on */
        }
        if (getenv("TAFFY_MASKDBG"))
            fprintf(stderr, "[mask2] fc=%.0f m[1]=%.6f m[5]=%.6f m[20]=%.6f m[50]=%.6f m[69]=%.6f m[139]=%.6f\n",
                    fc, (double)m[1], (double)m[5], (double)m[20], (double)m[50], (double)m[69], (double)m[139]);
    }
    cN = N; csr = sr; cfc = fc;
    return &m;
}

/* Forward declaration: per-event-type aux suppression test (defined before apply_aux) */
static bool aux_sup_covered(const taffy_engine *e, long long g0, long long g1);

/* Level-neutral replacement: scale the aux term by this hop's main/aux high-band energy
   ratio, so the replaced band keeps the main path's level. Kept in its own function --
   pv_step has very many locals and inlining this overflows its stack frame. */
static void aux_level_match(const Chan &k, taffy_engine::AuxBlk &blk, int Hs, float floorW) {
    const size_t need = (size_t)Hs;
    if (k.auxHpOla.size() < need || k.mainHpOla.size() < need ||
        k.auxHpWsum.size() < need || k.mainHpWsum.size() < need || blk.d.size() < need) return;
    /* Must use the normalized signals (divided by their wsum with the same floorW),
       matching the fill loop `d = auxHpOla/wa - mainHpOla/wm`. */
    double ea = 0.0, em = 0.0;
    for (size_t i = 0; i < need; i++) {
        float wa = k.auxHpWsum[i], wm = k.mainHpWsum[i];
        if (wa < floorW) wa = floorW;
        if (wm < floorW) wm = floorW;
        const double av = (double)k.auxHpOla[i] / (double)wa;
        const double mv = (double)k.mainHpOla[i] / (double)wm;
        ea += av * av; em += mv * mv;
    }
    if (ea <= 1e-30) return;
    const double r = std::sqrt(em / ea);
    for (size_t i = 0; i < need; i++) {
        float wa = k.auxHpWsum[i], wm = k.mainHpWsum[i];
        if (wa < floorW) wa = floorW;
        if (wm < floorW) wm = floorW;
        blk.d[i] = (float)(((double)k.auxHpOla[i] / (double)wa) * r
                           - (double)k.mainHpOla[i] / (double)wm);
    }
}


/* Aux-only high-frequency shelf: the per-band residual of real speech is tilted, so a
   single scalar gain cannot null it. Applied to the aux HP path only -- putting it in the
   mask would affect the main path as well and cancel out.
   Form: dB(f) = A + B*log10(f/2000), e.g. A=+0.3, B=-0.538. */
static float aux_shelf_db(double f) {
    static const double A = getenv("TAFFY_SHELFA") ? atof(getenv("TAFFY_SHELFA")) : 1.4;
    static const double B = getenv("TAFFY_SHELFB") ? atof(getenv("TAFFY_SHELFB")) : -0.75;
    if (f < 200.0) return 1.0f;
    const double db = A + B * std::log10(f / 2000.0);
    return (float)std::pow(10.0, db / 20.0);
}

/* True envelope: iterative maximum envelope in the log domain.
     1) log_spec = ln(|X| + tiny)
     2) cepstral smoothing (keep the first `order` cepstral coefficients)
     3) iterate env = max(env, log_spec), env = smooth(env) until convergence
   Returns a linear-domain per-bin gain env[] over N points (conjugate symmetric). */
static void true_envelope(const std::vector<float> &mag, int N, int order,
                          std::vector<float> &env) {
    const int H = N / 2;
    if ((int)env.size() != N) env.assign((size_t)N, 1.0f);
    if (order < 2) order = 2;
    if (order > H) order = H;
    std::vector<float> ls((size_t)N, 0.0f), tmp((size_t)N, 0.0f);
    std::vector<float> re((size_t)N, 0.0f), im((size_t)N, 0.0f);
    for (int b = 0; b <= H; b++) {
        const float m = (b < (int)mag.size()) ? mag[(size_t)b] : 0.0f;
        ls[(size_t)b] = std::log(m + 1e-20f);           /* log domain */
        if (b > 0 && b < H) ls[(size_t)(N - b)] = ls[(size_t)b];   /* mirror for symmetry */
    }
    /* Lifter: w[2i] = cos^2(i*pi/(2*order - 1)) for i < order, odd indices are 0, and
       w[k >= 2*order] = 0. */
    std::vector<float> win((size_t)N, 0.0f);
    {
        const double den = 2.0 * (double)order - 1.0;
        for (int i = 0; i < order; i++) {
            const double c = std::cos((double)i * M_PI / den);
            win[(size_t)(2 * i)] = (float)(c * c);
        }
    }
    /* cepstral smoothing: FFT -> multiply by the lifter -> IFFT */
    auto smooth = [&](const std::vector<float> &x, std::vector<float> &y) {
        re = x; std::fill(im.begin(), im.end(), 0.0f);
        taffy::fft(re, im, false);
        for (int k = 0; k < N; k++) { re[(size_t)k] *= win[(size_t)k];
                                      im[(size_t)k] *= win[(size_t)k]; }
        taffy::fft(re, im, true);
        y = re;
    };
    smooth(ls, env);                                     /* initial envelope */
    if (getenv("TAFFY_ENVDBG2")) {
        static long n2 = 0;
        if (n2++ % 40 == 0)
            fprintf(stderr, "[env2] ls[0]=%.3f ls[100]=%.3f ls[1000]=%.3f | env[0]=%.3f env[100]=%.3f env[1000]=%.3f\n",
                    (double)ls[0], (double)ls[100], (double)ls[1000],
                    (double)env[0], (double)env[100], (double)env[1000]);
    }
    static const int NIT = getenv("TAFFY_ENVITER") ? atoi(getenv("TAFFY_ENVITER")) : 10;
    for (int it = 0; it < NIT; it++) {
        float mx = 0.0f;
        for (int b = 0; b <= H; b++) {                   /* element-wise max */
            const float v = std::max(env[(size_t)b], ls[(size_t)b]);
            tmp[(size_t)b] = v;
            mx = std::max(mx, v - env[(size_t)b]);
            if (b > 0 && b < H) tmp[(size_t)(N - b)] = v;
        }
        smooth(tmp, env);                                /* smooth */
        if (mx <= 1.0f) break;                           /* converged: absolute threshold 1.0 */
    }
    /* Back to the linear domain: the log envelope's absolute scale is arbitrary, so the
       mean is subtracted before exp to obtain a unit-mean spectral shape. */
    double mean = 0.0;
    for (int b = 0; b <= H; b++) mean += (double)env[(size_t)b];
    mean /= (double)(H + 1);
    static const double RNG = getenv("TAFFY_ENVRNG") ? atof(getenv("TAFFY_ENVRNG")) : 12.0;  /* +/- dB clamp */
    const double glo = std::pow(10.0, -RNG / 20.0), ghi = std::pow(10.0, RNG / 20.0);
    /* Upsample with out-of-range zeroing:
         spec[i] *= env[2*floor((i/2)*shift + 0.5)], out-of-range index -> 0;
         shift = 2*order. */
    static const double SHIFT = getenv("TAFFY_ENVSHIFT") ? atof(getenv("TAFFY_ENVSHIFT")) : 256.0;
    std::vector<float> up((size_t)N, 0.0f);
    for (int i = 0; i <= H; i++) {
        const double idx = 2.0 * std::floor(((double)i / 2.0) * SHIFT + 0.5);
        if (idx < (double)N) {
            int k = (int)idx;
            double g = std::exp((double)env[(size_t)k] - mean);
            if (!(g > 1e-12) || g > 1e12) g = 1.0;
            if (g < glo) g = glo;
            if (g > ghi) g = ghi;
            up[(size_t)i] = (float)g;
        } else {
            up[(size_t)i] = 0.0f;              /* out of range -> 0 */
        }
    }
    for (int b = 0; b <= H; b++) {
        env[(size_t)b] = up[(size_t)b];
        if (b > 0 && b < H) env[(size_t)(N - b)] = up[(size_t)b];
    }
}

bool pv_step(taffy_engine *e) {
    const int N = e->N, Ha = e->Ha, Hs = e->Hs;
    for (int c = 0; c < e->channels; c++) {
        Chan &k = e->ch[(size_t)c];
        if (k.inBuf.size() < e->inRead + (size_t)N) return false;
    }
    for (int c = 0; c < e->channels; c++) {
        Chan &k = e->ch[(size_t)c];
        for (int i = 0; i < N; i++) {
            k.re[(size_t)i] = k.inBuf[e->inRead + (size_t)i] * e->win[(size_t)i];
            k.im[(size_t)i] = 0.0f;
        }
        taffy::fft(k.re, k.im, false);

        if (e->proAuxMode == 8) {                 /* keep the input-frame spectrum (pre phase advance) */
            if ((int)k.srcRe.size() != N) {
                k.srcRe.assign((size_t)N, 0.0f); k.srcIm.assign((size_t)N, 0.0f);
                k.auxHpOla.assign((size_t)N, 0.0f); k.auxHpWsum.assign((size_t)N, 0.0f);
                k.mainHpOla.assign((size_t)N, 0.0f); k.mainHpWsum.assign((size_t)N, 0.0f);
                k.re2.assign((size_t)N, 0.0f); k.im2.assign((size_t)N, 0.0f);
                k.re3.assign((size_t)N, 0.0f); k.im3.assign((size_t)N, 0.0f);
            }
            k.srcRe = k.re; k.srcIm = k.im;
            /* N/2-centered aux frame layout: the current input sits in the middle of the
               frame instead of at its start, so the phase reference differs by N/2 and the
               masked IFFT+OLA frame interference changes accordingly.
               TAFFY_AUXSHIFT=1 enables it. */
            if (getenv("TAFFY_AUXSHIFT") && atoi(getenv("TAFFY_AUXSHIFT"))) {
                const long long base = (long long)e->inRead - (long long)(N / 2);
                if (base >= 0 && base + N <= (long long)k.inBuf.size()) {
                    for (int i = 0; i < N; i++) {
                        k.srcRe[(size_t)i] = k.inBuf[(size_t)(base + i)] *
                            (((int)e->winAux.size() == N) ? e->winAux[(size_t)i] : e->win[(size_t)i]);
                        k.srcIm[(size_t)i] = 0.0f;
                    }
                    taffy::fft(k.srcRe, k.srcIm, false);       /* FFT of the centered frame */
                }
            }
        }
        if (e->anaCopy || e->proAuxMode == 8) {
            if ((int)k.magIn.size() != N) k.magIn.assign((size_t)N, 0.0f);
            for (int b = 0; b <= N / 2; b++)
                k.magIn[(size_t)b] = std::sqrt(k.re[(size_t)b] * k.re[(size_t)b] +
                                               k.im[(size_t)b] * k.im[(size_t)b]);
        }
        /* Cumulative normalized spectral flux gate:
             cum[b] = sum_{j<=b}(cur[j]-prev[j])^2 / sum_{j<=b} prev[j]^2
           cum[b] < 0.3 lets aux take over that bin: bins whose spectrum changes strongly
           (impulses, onsets) keep the main path, stationary content may go to aux. */
        if (e->proAuxMode == 8) {
            if ((int)k.prevMag.size() != N / 2 + 1) {
                k.prevMag.assign((size_t)(N / 2 + 1), 0.0f);
                k.fluxGate.assign((size_t)N, 0.0f);
            }
            /* fe2 starts at 1e-24 with no guard: a silent previous frame is exactly the
               largest-flux case, so guarding it would leave the gate permanently open. */
            static const double FT = getenv("TAFFY_FLUXT") ? atof(getenv("TAFFY_FLUXT")) : 999.0;   /* default off (999) */
            /* Per-bin "sticky" cumulative flux: the array persists across frames and only
               bins with (delta)^2 > 1e-14 are updated, so bins stay high long after an
               impulse and keep the gate closed there. */
            double fc2 = 0.0, fe2 = 1e-24;
            {
                if ((int)k.fluxGate.size() != N) k.fluxGate.assign((size_t)N, 1.0f);
                for (int b = 0; b <= N / 2; b++) {
                    const double cu = (double)k.magIn[(size_t)b];
                    const double pv = (double)k.prevMag[(size_t)b];
                    const double d2 = (cu - pv) * (cu - pv);
                    fc2 += d2; fe2 += pv * pv;
                    if (d2 > 1e-14)                       /* update only on significant change */
                        k.fluxGate[(size_t)b] = ((fc2 / fe2) < FT) ? 1.0f : 0.0f;
                    /* otherwise keep the previous value (sticky) */
                }
            }
            for (int b = N / 2 + 1; b < N; b++) k.fluxGate[(size_t)b] = k.fluxGate[(size_t)(N - b)];
            /* State-3 mechanism: when the frame is not a transient and the normalized flux
               exceeds 0.4, the peak state is set to 3, which blocks aux. The threshold is
               0.4 (not 0.3) and is only set in the non-transient branch. */
            const double fluxRatio = fc2 / fe2;
            static const double F3 = getenv("TAFFY_FLUX3") ? atof(getenv("TAFFY_FLUX3")) : 999.0;   /* default off (999) */
            const bool st3 = (!k.isTransient) && (fluxRatio > F3);
            /* Per-bin cumulative normalized flux: aux may not take over a bin whose value
               exceeds 0.3, which blocks both the impulse frame and the silence after it
               (large delta against the impulse frame's energy). Note that it is cumulative
               up to that bin, not a per-bin instantaneous ratio. */
            float cov = 0.0f;
            {
                int nb = 0, npass = 0;
                for (int b = 139; b <= N / 2; b++) { nb++; if (k.fluxGate[(size_t)b] > 0.5f) npass++; }
                cov = nb ? (float)((double)npass / (double)nb) : 1.0f;
            }
            const float fg = st3 ? 0.0f : cov;
            /* Spectral-peak band mask: aux takes over only at spectral peaks, and each peak
               owns the band delimited by the midpoints to its neighbours. Silent regions
               have no peaks, so aux leaves them untouched. */
            static const int PEAK = getenv("TAFFY_PEAK") ? atoi(getenv("TAFFY_PEAK")) : 0;
            /* Stationary-bin mask (TAFFY_STAT, default off): aux may only take over bins
               with a small |phase deviation|. */
            static const double STAT = getenv("TAFFY_STAT") ? atof(getenv("TAFFY_STAT")) : 0.0;
            if ((int)k.statB.size() != N) k.statB.assign((size_t)N, 1e9f);
            for (int b = 0; b < N; b++) k.fluxGate[(size_t)b] = 1.0f;
            if (STAT > 0.0) {
                for (int b = 0; b <= N / 2; b++)
                    k.fluxGate[(size_t)b] = (k.statB[(size_t)b] < STAT) ? 1.0f : 0.0f;
                for (int b = N / 2 + 1; b < N; b++) k.fluxGate[(size_t)b] = k.fluxGate[(size_t)(N - b)];
            }
            if (PEAK) {
                const int M2 = N / 2;
                std::vector<int> pk;
                pk.reserve(256);
                /* Amplitude threshold: only significant peaks count, otherwise numerical
                   noise in silent regions would put peaks everywhere. The floor is relative
                   to a slow envelope (TAFFY_PEAKDB, default -40 dB), so silent frames have
                   no peaks and aux does not act there. */
                static const double PDB = getenv("TAFFY_PEAKDB") ? atof(getenv("TAFFY_PEAKDB")) : -40.0;
                float mx = 0.0f;
                for (int b = 1; b < M2 - 1; b++) if (k.magIn[(size_t)b] > mx) mx = k.magIn[(size_t)b];
                if (k.magEnvMean <= 1e-20f) k.magEnvMean = mx;
                else k.magEnvMean += 0.05f * (mx - k.magEnvMean);
                const float floorMag = k.magEnvMean * (float)std::pow(10.0, PDB / 20.0);
                for (int b = 1; b < M2 - 1; b++) {
                    const float c0 = k.magIn[(size_t)b];
                    if (c0 < floorMag) continue;
                    if (c0 > k.magIn[(size_t)(b - 1)] && c0 >= k.magIn[(size_t)(b + 1)]) pk.push_back(b);
                }
                for (int b = 0; b < N; b++) k.fluxGate[(size_t)b] = 0.0f;
                for (size_t q = 0; q < pk.size(); q++) {
                    const int lo = (q == 0) ? 0 : (pk[q - 1] + pk[q]) / 2;
                    const int hi = (q + 1 == pk.size()) ? M2 : (pk[q] + pk[q + 1]) / 2;
                    for (int b = lo; b < hi; b++) k.fluxGate[(size_t)b] = 1.0f;
                }
                for (int b = M2 + 1; b < N; b++) k.fluxGate[(size_t)b] = k.fluxGate[(size_t)(N - b)];
                /* Coverage must be folded into blk8.sc and force skip when zero, otherwise
                   the -HP(delta) of auxActual ignores the mask. */
                {
                    int cov2 = 0, tot2 = 0;
                    for (int b = 139; b < M2; b++) { tot2++; if (k.fluxGate[(size_t)b] > 0.5f) cov2++; }
                    k.peakCov = tot2 ? (float)((double)cov2 / (double)tot2) : 1.0f;
                }
                if (getenv("TAFFY_PKDBG")) {
                    static long np = 0;
                    if (np++ % 20 == 0) {
                        int cov = 0, tot = 0;
                        for (int b = 139; b < M2; b++) { tot++; if (k.fluxGate[(size_t)b] > 0.5f) cov++; }
                        fprintf(stderr, "[pk] peaks=%zu coverage=%d/%d (%.1f%%) floor=%.3g\n",
                                pk.size(), cov, tot, 100.0*(double)cov/(double)(tot?tot:1), (double)floorMag);
                    }
                }
            }
            k.fluxScale = fg;
            k.prevMag.assign(k.magIn.begin(), k.magIn.begin() + (N / 2 + 1));
            if (getenv("TAFFY_FLUXDBG")) {
                fprintf(stderr, "[fluxT] whole-frame ratio=%.4f\n", fc2 / fe2);
                static long nf = 0;
                if (nf++ % 40 == 0) {
                    int z = 0, tot = 0;
                    for (int b = 139; b < N / 2; b++) { tot++; if (k.fluxGate[(size_t)b] < 0.5f) z++; }
                    fprintf(stderr, "[flux] closed above bin 139: %d/%d (%.1f%%)\n", z, tot,
                            100.0 * (double)z / (double)(tot ? tot : 1));
                }
            }
        }

        for (int b = 0; b <= N / 2; b++) {
            float rr = k.re[(size_t)b], ii = k.im[(size_t)b];
            k.mag[(size_t)b] = std::sqrt(rr * rr + ii * ii);
            k.origPhase[(size_t)b] = std::atan2(ii, rr);
        }

        // ---- transient detection: sub-band spectral flux + adaptive threshold ----
        // Sub-band count = N>>8; the threshold is mean + 2 SD with a 1.1x mean floor.
        {
            const int NB = std::max(1, N >> 8);
            if ((int)k.prevBand.size() != NB) k.prevBand.assign((size_t)NB, 0.0f);
            const int span = N / 2 + 1;
            const int bandSize = (span + NB - 1) / NB;
            float flux = 0.0f;
            for (int b = 0; b < NB; b++) {
                int lo = b * bandSize, hi = std::min(lo + bandSize, span);
                float sum = 0.0f;
                for (int j = lo; j < hi; j++) sum += k.mag[(size_t)j];
                if (k.frames > 0) {
                    float d = sum - k.prevBand[(size_t)b];
                    if (d > 0.0f) flux += d;
                }
                k.prevBand[(size_t)b] = sum;
            }
            const float a = 0.02f;
            if (k.frames > 0) {
                k.fluxMean += a * (flux - k.fluxMean);
                float dv = flux - k.fluxMean;
                k.fluxVar += a * (dv * dv - k.fluxVar);
            }
            float sd = std::sqrt(std::max(0.0f, k.fluxVar));
            float thr = k.fluxMean + 2.0f * sd;
            if (getenv("TAFFY_ISTDBG") && k.frames > 8 && flux > thr) fprintf(stderr, "[ist] f=%lld flux=%.2f thr=%.2f\n", (long long)k.frames, flux, thr);
            k.isTransient = (k.frames > 8) && (flux > thr) &&
                            (flux > k.fluxMean * 1.1f) && (k.sinceTransient > 4);
            k.sinceTransient = k.isTransient ? 0 : k.sinceTransient + 1;
            k.frames++;
        }

        /* High-band-only spectral flux for the Pro aux gate: summing positive flux over
           total magnitude above the crossover keeps low-frequency drums out of the test. */
        {
            const int span = N / 2 + 1;
            const int hiBin = (e->auxCrossBin >= 0)
                                  ? e->auxCrossBin
                                  : (int)(1500.0 / (double)e->sampleRate * (double)N);
            const int lo = std::max(2, std::min(hiBin, span - 2));
            if ((int)k.hiPrev.size() != span) k.hiPrev.assign((size_t)span, 0.0f);
            float flux = 0.0f, energy = 0.0f;
            for (int b = lo; b < span; b++) {
                const float m = k.mag[(size_t)b];
                energy += m;
                const float d = m - k.hiPrev[(size_t)b];
                if (d > 0.0f) flux += d;
                k.hiPrev[(size_t)b] = m;
            }
            const float rel = flux / (energy + 1e-12f);
            const float a = 0.02f;
            if (k.frames > 1) {
                k.hiFluxMean += a * (rel - k.hiFluxMean);
                const float dv = rel - k.hiFluxMean;
                k.hiFluxVar += a * (dv * dv - k.hiFluxVar);
            }
            const float sd = std::sqrt(std::max(0.0f, k.hiFluxVar));
            /* Gate source 2: isolated peaks of the high-band energy envelope. Spectral flux
               stays high for steady noise, so energy / slow mean is used instead: ~1 for
               steady material, >> 1 at drum onsets and at every impulse of a spike train. */
            k.hiEnvMean += 0.02f * (energy - k.hiEnvMean);
            const float ratio = energy / (k.hiEnvMean + 1e-12f);
            /* Rising edge plus hysteresis: the ratio stays above K across a whole burst,
               so a plain threshold would retrigger every hop; one burst must fire once. */
            if (k.hiEnvRatio <= e->auxGateK * 0.5f) k.hiEnvArmed = 1;
            k.hiTransient = (k.frames > 20) && (ratio > e->auxGateK) && k.hiEnvArmed;
            if (k.hiTransient) k.hiEnvArmed = 0;
            k.hiEnvRatio = ratio;
            k.hiEnvPrevRatio = ratio;
            k.sinceHiTransient = k.hiTransient ? 0 : k.sinceHiTransient + 1;
            if (const char *gd = getenv("TAFFY_GATEDUMP")) {
                static FILE *gf = NULL;
                if (!gf) gf = fopen(gd, "w");
                if (gf) fprintf(gf, "%lld %.8g %.6f %d %d\n",
                                (long long)k.frames, energy, ratio,
                                (int)k.hiTransient, k.sinceHiTransient);
            }
            /* Dumps the gate inputs for threshold calibration (TAFFY_GATEDBG=1). */
            if (getenv("TAFFY_GATEDBG"))
                fprintf(stderr, "[gate] f=%lld rel=%.4f mean=%.4f sd=%.4f trans=%d since=%d\n",
                        (long long)k.frames, rel, k.hiFluxMean, sd,
                        (int)k.hiTransient, k.sinceHiTransient);
        }

        /* Pro spectral-envelope stage: only bins >= splitBin are touched, everything below
           stays unchanged. The envelope is a real cepstrum plus lifter, applied per envAlpha:
           -1 = divide by envelope (flatten), +1 = multiply (emphasize formants). */
        if (e->proEnv && e->envMix > 0.0f) {
            const int span = N / 2 + 1;
            int splitBin = (int)std::floor((double)N * 1500.0 / (double)e->sampleRate);
            if (splitBin < 1) splitBin = 1;
            if (splitBin >= span) splitBin = span - 1;
            int order = e->envOrder;
            if (order < 1) order = 1;
            if (order > span / 2) order = span / 2;

            static std::vector<float> lr, li, lg;
            lr.assign((size_t)N, 0.0f); li.assign((size_t)N, 0.0f);
            lg.resize((size_t)span);
            for (int b = 0; b < span; b++)
                lg[(size_t)b] = std::log(std::max(k.mag[(size_t)b], 1e-12f));
            /* Mirror to length N so the transform gives a real cepstrum. */
            for (int b = 0; b < span; b++) lr[(size_t)b] = lg[(size_t)b];
            for (int b = 1; b < span - 1; b++) lr[(size_t)(N - b)] = lg[(size_t)b];
            taffy::fft(lr, li, false);
            /* lifter: keep only the first `order` coefficients plus their conjugates */
            for (int b = order; b <= N - order; b++) { lr[(size_t)b] = 0.0f; li[(size_t)b] = 0.0f; }
            taffy::fft(lr, li, true);
            /* Normalize against the mean over b >= splitBin so in-band gain is unchanged. */
            double sum = 0; int cnt = 0;
            for (int b = splitBin; b < span; b++) { sum += (double)lr[(size_t)b]; cnt++; }
            double mean = cnt > 0 ? sum / cnt : 0.0;
            for (int b = splitBin; b < span; b++) {
                double rel = (double)lr[(size_t)b] - mean;          /* relative value in log domain */
                double g = std::exp((double)e->envAlpha * rel);
                float gg = (float)((1.0 - (double)e->envMix) + (double)e->envMix * g);
                k.mag[(size_t)b] *= gg;
            }
        }

        /* Analysis-domain transient copy: frames inside a copy span [outPos +/- win] become
           identity frames (analysis magnitude and phase, not the pv state). Hann^2 with
           hop = N/4 reconstructs the input exactly, so the frame spectrum equals the input
           spectrum as well, and a complementary aux mask cancels it to zero. A time-domain
           overwrite changes the output but not the spectrum. */
        bool identFrame = false;
        if (e->anaCopy) {
            const long long g0 = (long long)e->pvTrimmed + (long long)e->pv[(size_t)c].size();
            for (size_t q = 0; q < e->pending.size(); q++) {
                const taffy_engine::TEvent &E2 = e->pending[q];
                if (E2.win <= 0) continue;
                const long long lo = E2.outPos - (long long)E2.win;
                const long long hi = E2.outPos + (long long)E2.win;
                if (hi < g0 || lo > g0 + (long long)N) continue;   /* disjoint from the frame's OLA support */
                identFrame = true;
                break;
            }
            if (identFrame)
                for (int b = 0; b <= N / 2; b++) k.mag[(size_t)b] = k.magIn[(size_t)b];
        }

        // ---- Phase advance; transient frames reset straight to the analysis phase. ----
        /* The first frame must adopt the analysis phase directly. With sumPhase/lastPhase
         * starting at 0, delta and the first synthesis phase are both meaningless, which
         * spikes the first hop's output.
         * k.frames was already incremented by the transient detection above. */
        /* The transient phase reset has to persist for several hops, not one frame:
           neighbouring frames folded in by the OLA would otherwise break the identity
           inside a copy span. Held for TAFFY_THOLD hops (default 4). */
        if (k.isTransient) k.transHold = e->transHoldHops;
        const bool holdFrame = (k.transHold > 0);
        if (k.transHold > 0) k.transHold--;
        if (k.frames <= 1 || k.isTransient || holdFrame || identFrame) {
            for (int b = 0; b <= N / 2; b++) {
                k.sumPhase[(size_t)b] = k.origPhase[(size_t)b];
                k.lastPhase[(size_t)b] = k.origPhase[(size_t)b];
            }
        } else {
            for (int b = 0; b <= N / 2; b++) {
                float delta = princarg(k.origPhase[(size_t)b] - k.lastPhase[(size_t)b]
                                       - e->omega[(size_t)b] * (float)Ha);
                /* Stationary-bin test: a "peak" bin is one whose instantaneous frequency is
                   close to its nominal frequency, i.e. |delta| is small. Silent regions have
                   random phase, so they produce no peaks and aux leaves them alone. */
                if (e->proAuxMode == 8) {
                    if ((int)k.statB.size() != N) k.statB.assign((size_t)N, 0.0f);
                    k.statB[(size_t)b] = delta;                        /* signed, used for segmentation */
                }
                k.lastPhase[(size_t)b] = k.origPhase[(size_t)b];
                float trueFreq = e->omega[(size_t)b] + delta / (float)Ha;
                k.sumPhase[(size_t)b] = princarg(k.sumPhase[(size_t)b] + trueFreq * (float)Hs);
            }
        }

        // Identity phase locking (Laroche & Dolson): regions around a spectral peak share one phase shift.
        /* aux crossover bin (/64). proAux modes: 1 = no identity phase locking above the
           crossover; 2 = use the analysis phase above it directly, without accumulation. */
        int xb = N / 2;
        if (e->proAux) {
            const int xsrc = (e->auxFcBin >= 0) ? e->auxFcBin : e->auxCrossBin;
            xb = (xsrc >= 0) ? xsrc
                             : (int)(1500.0 * (double)N / (double)e->sampleRate);
            if (xb < 1) xb = 1;
            if (xb > N / 2) xb = N / 2;
        }
        if (e->proAux == 2) {
            for (int b = xb; b <= N / 2; b++) { k.sumPhase[(size_t)b] = k.origPhase[(size_t)b]; }
        }
        int lockHi = (e->proAux == 1) ? xb : (N / 2);
        for (int b = 1; !k.isTransient && b < lockHi; b++) {
            if (k.mag[(size_t)b] > k.mag[(size_t)b - 1] &&
                k.mag[(size_t)b] >= k.mag[(size_t)b + 1] &&
                k.mag[(size_t)b] > 1e-9f) {
                int lo = b, hi = b;
                while (lo > 1 && k.mag[(size_t)lo - 1] <= k.mag[(size_t)lo]) lo--;
                while (hi < N / 2 - 1 && k.mag[(size_t)hi + 1] <= k.mag[(size_t)hi]) hi++;
                float shift = k.sumPhase[(size_t)b] - k.origPhase[(size_t)b];
                for (int j = lo; j <= hi; j++)
                    k.sumPhase[(size_t)j] = princarg(k.origPhase[(size_t)j] + shift);
            }
        }

        for (int b = 0; b <= N / 2; b++) {
            float m = k.mag[(size_t)b], p = k.sumPhase[(size_t)b];
            float rr = m * std::cos(p), ii = m * std::sin(p);
            k.re[(size_t)b] = rr;
            k.im[(size_t)b] = ii;
            if (b > 0 && b < N / 2) {
                k.re[(size_t)(N - b)] = rr;
                k.im[(size_t)(N - b)] = -ii;
            }
        }
        if (e->proAuxMode == 8 && !getenv("TAFFY_M8SKIPHP")) {
            const double fcx = (e->auxFcBin >= 0 ? e->auxFcBin : e->auxCrossBin) >= 0
                                   ? (double)(e->auxFcBin >= 0 ? e->auxFcBin : e->auxCrossBin)
                                         * (double)e->sampleRate / (double)e->N
                                   : 1500.0;
            const std::vector<float> *M2 = aux_mask(N, (double)e->sampleRate, fcx);
            if (getenv("TAFFY_MASKDBG")) {
                double sm = 0; for (int b = 0; b < N; b++) sm += (*M2)[(size_t)b];
                static int once = 0;
                if (once++ < 3)
                    fprintf(stderr, "[mask] N=%d sr=%.0f fcx=%.1f auxCrossBin=%d maskSum=%.1f "
                                    "m[0]=%.3f m[69]=%.3f m[139]=%.3f m[1000]=%.3f\n",
                            N, (double)e->sampleRate, fcx, e->auxCrossBin, sm,
                            (double)(*M2)[0], (double)(*M2)[69], (double)(*M2)[139], (double)(*M2)[1000]);
            }
            k.re2 = k.re; k.im2 = k.im;
            for (int b = 0; b < N; b++) { const float mg = (*M2)[(size_t)b] * k.fluxGate[(size_t)b];
                                          k.re2[(size_t)b] *= mg; k.im2[(size_t)b] *= mg; }
            taffy::fft(k.re2, k.im2, true);          /* time-domain frame of HP(main) */
            if (getenv("TAFFY_LVLDBG")) {            /* per-hop level comparison */
                static long nl = 0;
                if (nl++ % 20 == 0) {
                    double s_re=0,s_src=0,s_re2=0,s_re3=0;
                    for (int b = 0; b <= N/2; b++) {
                        s_re  += (double)k.re[(size_t)b]*k.re[(size_t)b];
                        s_src += (double)k.srcRe[(size_t)b]*k.srcRe[(size_t)b];
                    }
                    for (int i = 0; i < N; i++) {
                        s_re2 += (double)k.re2[(size_t)i]*k.re2[(size_t)i];
                        s_re3 += (double)k.re3[(size_t)i]*k.re3[(size_t)i];
                    }
                    fprintf(stderr, "[lvl] spec: re=%.5g srcRe=%.5g (diff %+.2f dB) | time: HP(main)=%.5g HP(in)=%.5g (diff %+.2f dB)\n",
                            std::sqrt(s_re), std::sqrt(s_src),
                            20.0*std::log10((std::sqrt(s_re)+1e-30)/(std::sqrt(s_src)+1e-30)),
                            std::sqrt(s_re2), std::sqrt(s_re3),
                            20.0*std::log10((std::sqrt(s_re3)+1e-30)/(std::sqrt(s_re2)+1e-30)));
                }
            }
            k.re3 = k.srcRe; k.im3 = k.srcIm;
            static const int SHELF = getenv("TAFFY_SHELF") ? atoi(getenv("TAFFY_SHELF")) : 1;
            for (int b = 0; b < N; b++) {
                float mg = (*M2)[(size_t)b] * k.fluxGate[(size_t)b];
                if (SHELF) {                       /* aux path only */
                    const double fb = (b <= N / 2) ? (double)b : (double)(N - b);
                    mg *= aux_shelf_db(fb * (double)e->sampleRate / (double)N);
                }
                static const int ENVON = getenv("TAFFY_ENV") ? atoi(getenv("TAFFY_ENV")) : 0;
                if (ENVON) {                       /* true-envelope weighting */
                    static const int EORD = getenv("TAFFY_ENVORDER") ? atoi(getenv("TAFFY_ENVORDER")) : 128;
                    static std::vector<float> envg;
                    if (b == 0) {
                        true_envelope(k.magIn, N, EORD, envg);
                        if (getenv("TAFFY_ENVDBG")) {
                            static long ne = 0;
                            if (ne++ % 40 == 0)
                                fprintf(stderr, "[env] mag[0]=%.4g mag[100]=%.4g mag[1000]=%.4g "
                                                "env[0]=%.4g env[100]=%.4g env[1000]=%.4g\n",
                                        (double)k.magIn[0], (double)k.magIn[100], (double)k.magIn[1000],
                                        (double)envg[0], (double)envg[100], (double)envg[1000]);
                        }
                    }
                    if ((int)envg.size() == N) mg *= envg[(size_t)b];
                }
                k.re3[(size_t)b] *= mg; k.im3[(size_t)b] *= mg;
            }
            taffy::fft(k.re3, k.im3, true);          /* time-domain frame of HP(input) */
            if (getenv("TAFFY_HPDBG")) {
                k.hpDbg = 1;      /* defer the print until after the main IFFT, when k.re holds the main frame */
            }
            /* The accumulation must reuse the main path's window w: on transient frames that
               window is zeroed before transD, and an unmodified window here would make
               LP(main) + HP(main) != main and leak the difference into the low band. */
        }
        taffy::fft(k.re, k.im, true);

        if (k.hpDbg && getenv("TAFFY_HPDBG")) {
            static int once2 = 0;
            if (once2++ < 4) {
                auto rms = [&](const std::vector<float> &v) {
                    double s = 0; for (int i = 0; i < N; i++) s += (double)v[(size_t)i]*v[(size_t)i];
                    return std::sqrt(s / N);
                };
                /* crude low-pass: RMS of a 32-sample moving average; a high-band-only signal should be far below the full-band RMS */
                auto lprms = [&](const std::vector<float> &v) {
                    double acc = 0; int cnt = 0;
                    for (int i = 0; i + 32 <= N; i += 8) {
                        double m = 0; for (int j = 0; j < 32; j++) m += v[(size_t)(i + j)];
                        m /= 32.0; acc += m * m; cnt++;
                    }
                    return cnt ? std::sqrt(acc / cnt) : 0.0;
                };
                fprintf(stderr, "[hp] main RMS=%.4g LP=%.4g | HP(main) RMS=%.4g LP=%.4g | "
                                "HP(aux) RMS=%.4g LP=%.4g\n",
                        rms(k.re), lprms(k.re), rms(k.re2), lprms(k.re2), rms(k.re3), lprms(k.re3));
            }
        }

        /* Short synthesis window on transient frames to suppress pre-echo (TAFFY_TSHORT=1,
           off by default): the full window smears N samples ahead of an impulse. Length
           2*Hs, accumulated into wsum with the same window so normalization stays valid. */
        static const bool TSHORT = getenv("TAFFY_TSHORT") && atoi(getenv("TAFFY_TSHORT")) != 0;
        const bool shortWin = TSHORT && k.isTransient;
        const int SW = 2 * Hs;
        const int SW0 = (N - SW) / 2;
        /* Forward window on transient frames, cuts pre-echo (TAFFY_PRESUP=1, off by default):
           an impulse at intra-frame offset d spreads +/-N/2 in the OLA buffer and the part
           with i < d lands at earlier output positions. Zeroing w[i] for i < d truncates
           that half. d is the intra-frame time-domain energy peak, computed per frame. */
        static const bool PRESUP = getenv("TAFFY_PRESUP") && atoi(getenv("TAFFY_PRESUP")) != 0;
        int transD = -1;
        if (PRESUP && k.isTransient) {
            double best = -1.0; int bd = 0;
            for (int i = 0; i < N; i++) {
                double v = (double)k.re[(size_t)i] * (double)k.re[(size_t)i];
                if (v > best) { best = v; bd = i; }
            }
            transD = bd;
        }
        for (int i = 0; i < N; i++) {
            float w = e->win[(size_t)i];
            if (transD > 0 && i < transD) w = 0.0f;
            if (shortWin) {
                w = (i >= SW0 && i < SW0 + SW)
                        ? 0.5f * (1.0f - std::cos(2.0f * (float)M_PI * (float)(i - SW0) / (float)SW))
                        : 0.0f;
            }
            k.ola[(size_t)i]  += k.re[(size_t)i] * w;
            k.wsum[(size_t)i] += w * w;
            if (e->proAuxMode == 8) {
                k.mainHpOla[(size_t)i]  += k.re2[(size_t)i] * w;
                k.mainHpWsum[(size_t)i] += w * w;
                /* aux uses its own window (Hamming), so the main path keeps w. */
                float wa2 = ((int)e->winAux.size() == N) ? e->winAux[(size_t)i] : w;
                if (transD > 0 && i < transD) wa2 = 0.0f;
                if (shortWin) wa2 = w;
                k.auxHpOla[(size_t)i]   += k.re3[(size_t)i] * wa2;
                k.auxHpWsum[(size_t)i]  += wa2 * wa2;
            }
        }
        std::vector<float> &dst = e->pv[(size_t)c];
        /* mode 8 aux gate, evaluated once per hop (windows chosen by event type, same
           criteria as mode 4) and applied to every output sample of this hop. aux is added
           before apply_transients, so a transient copy overwrites it again. */
        float floorW8 = 1e-8f;
        {
            static const float WF8 = getenv("TAFFY_WFLOOR") ? (float)atof(getenv("TAFFY_WFLOOR")) : 0.01f;
            float steady8 = 0.0f;
            if (WF8 > 0.0f) for (int j = 0; j < N; j += Hs) steady8 += e->win[(size_t)j] * e->win[(size_t)j];
            floorW8 = WF8 > 0.0f ? steady8 * WF8 : 1e-8f;
        }
        /* Dump the aux OLA output, interleaved (aux, main), Hs pairs per hop. */
        if (e->proAuxMode == 8 && getenv("TAFFY_DUMPAUXO") && c == 0) {
            static FILE *auxo_f = NULL;
            if (!auxo_f) auxo_f = fopen(getenv("TAFFY_DUMPAUXO"), "wb");
            if (auxo_f) {
                for (int i = 0; i < Hs; i++) {
                    float wa = k.auxHpWsum[(size_t)i];  if (wa < floorW8) wa = floorW8;
                    float wm = k.mainHpWsum[(size_t)i]; if (wm < floorW8) wm = floorW8;
                    float va = k.auxHpOla[(size_t)i] / wa;
                    float vm = k.mainHpOla[(size_t)i] / wm;
                    fwrite(&va, 4, 1, auxo_f);
                    fwrite(&vm, 4, 1, auxo_f);
                }
            }
        }
        if (e->proAuxMode == 8 && !getenv("TAFFY_M8NOQ")) {
            /* Queue this hop's HP(aux) - HP(main) for apply_aux to add after
               apply_transients: adding it here would be overwritten by the per-sample
               transient copy. The gate is evaluated now, before events are consumed. */
            if ((int)e->auxQ.size() < e->channels) e->auxQ.resize((size_t)e->channels);
            e->auxQ[(size_t)c].emplace_back();
            taffy_engine::AuxBlk &blk8 = e->auxQ[(size_t)c].back();
            blk8.base = dst.size();
            if (getenv("TAFFY_SZDBG")) {
                static long ns = 0;
                if (ns++ < 6)
                    fprintf(stderr, "[sz] Hs=%d auxHpOla=%zu mainHpOla=%zu auxHpWsum=%zu "
                                    "magIn=%zu prevMag=%zu fluxGate=%zu statB=%zu\n",
                            Hs, k.auxHpOla.size(), k.mainHpOla.size(), k.auxHpWsum.size(),
                            k.magIn.size(), k.prevMag.size(), k.fluxGate.size(), k.statB.size());
            }
            /* Level-neutral replacement: store only the ratio and let apply_aux scale the
               whole (d - hpDelta) contribution; scaling d alone leaves -hpDelta untouched. */
            /* Partial blend coefficient: out = main + alpha*(aux - main), TAFFY_AUXA. */
            {
                static const double ALPHA = getenv("TAFFY_AUXA") ? atof(getenv("TAFFY_AUXA")) : 0.92;
                blk8.alpha = (float)ALPHA;
            }
            blk8.sc = k.fluxScale * k.peakCov;   /* every gate must fold into sc, otherwise the
                                            -HP(delta) term of auxActual ignores it. The
                                            kurtosis gate below overwrites sc, so this only
                                            matters while the peak mask is active. */
            blk8.d.assign((size_t)Hs, 0.0f);
            const long long gg = (long long)e->pvTrimmed + (long long)dst.size();
            if (e->auxGateSrc == 1)
                blk8.cov = aux_sup_covered(e, gg, gg + (long long)Hs) ? 1 : 0;
            /* Level gate: aux must close at silence or very low level, otherwise
               HP(aux) - HP(main) = 0 - HP(main) erases the phase vocoder's decay tail.
               Tests the frame RMS against its running mean; TAFFY_AUXLVL sets the
               threshold in dB (default -40). */
            /* Fill unconditionally, then gate: gates that must restore wrongly zeroed
               content (e.g. sustained silence) cannot work the other way round. */
            for (int i = 0; i < Hs; i++) {
                float wa = k.auxHpWsum[(size_t)i], wm = k.mainHpWsum[(size_t)i];
                if (wa < floorW8) wa = floorW8;
                if (wm < floorW8) wm = floorW8;
                /* The output stage is asymmetric: the main path divides by its window sum,
                   aux does not. TAFFY_AUXNORM = exponent on the aux wsum (1 = divide, 0 = raw). */
                static const double AN = getenv("TAFFY_AUXNORM") ? atof(getenv("TAFFY_AUXNORM")) : 1.0;
                const float waN = (AN == 0.0) ? 1.0f
                                : ((AN == 2.0) ? (wa * wa) : wa);
                blk8.d[(size_t)i] = k.auxHpOla[(size_t)i] / waN - k.mainHpOla[(size_t)i] / wm;
            }
            /* Level matching must run after the fill, otherwise it is overwritten at once. */
            {
                static const int LVM = getenv("TAFFY_AUXLVM") ? atoi(getenv("TAFFY_AUXLVM")) : 0;   /* off by default */
                if (LVM) aux_level_match(k, blk8, Hs, floorW8);
            }
            bool lvlOk = true;
            {
                /* See the comment below; default -30 dB (aux off while the input is silent). */
                const double LV = getenv("TAFFY_AUXLVL") ? atof(getenv("TAFFY_AUXLVL")) : e->auxLvlDb;
                if (LV > -200.0) {
                    /* The test must ask whether the input segment matching this output hop
                       is silent, so it looks at the last Ha input samples: between impulses
                       that segment is exactly zero, while drums, noise and vowels always
                       carry energy. The full N-point analysis frame straddles the impulses
                       and separates them far less cleanly. */
                    double sq = 0.0;
                    const int segLen = Ha > 0 ? Ha : Hs;
                    for (int i = 0; i < segLen; i++) {
                        const double v = (double)k.inBuf[e->inRead + (size_t)i];
                        sq += v * v;
                    }
                    const double rms = std::sqrt(sq / (double)segLen);
                    if (e->auxEnvMean <= 1e-12) e->auxEnvMean = rms;
                    else e->auxEnvMean += 0.02 * (rms - e->auxEnvMean);
                    const double thr = e->auxEnvMean * std::pow(10.0, LV / 20.0);
                    lvlOk = rms >= thr;
                }
            }
            /* Sustained-silence gate: a drum burst decays exponentially so its tail is also
               exactly silent, but only impulses stay silent for many hops. aux closes after
               K consecutive silent hops (TAFFY_AUXSILK). Update the run length first. */
            {
                static const int K = getenv("TAFFY_AUXSILK") ? atoi(getenv("TAFFY_AUXSILK")) : 0;
                e->auxSilRun = lvlOk ? 0 : (e->auxSilRun + 1);
                if (!lvlOk && e->auxSilRun >= K) blk8.d.assign((size_t)Hs, 0.0f);
            }
            /* Silent-hop duty gate: the fraction of the last 128 hops whose matching input
               segment was silent. Roughly 0.8 for sparse impulses, 0.2 for drums, ~0 for
               noise or vowels. Above DHI (default 0.8) aux fades out completely; between
               DLO (0.4) and DHI it fades linearly; below DLO it is left alone. */
            {
                static const double DLO = getenv("TAFFY_AUXDLO") ? atof(getenv("TAFFY_AUXDLO")) : 0.4;
                static const double DHI = getenv("TAFFY_AUXDHI") ? atof(getenv("TAFFY_AUXDHI")) : 0.8;
                const bool silent = !lvlOk;
                if ((int)e->auxDutyRing.size() < 128) e->auxDutyRing.assign(128, 0.0f);
                e->auxDutyRing[(size_t)(e->auxDutyPos & 127)] = silent ? 1.0f : 0.0f;
                e->auxDutyPos++;
                if (e->auxDutyPos >= 128 && DHI > DLO) {
                    int ns = 0;
                    for (int i = 0; i < 128; i++) if (e->auxDutyRing[(size_t)i] > 0.5f) ns++;
                    const double frac = (double)ns / 128.0;
                    double sc = 1.0;
                    if (frac >= DHI) sc = 0.0;
                    else if (frac > DLO) sc = (DHI - frac) / (DHI - DLO);
                    if (getenv("TAFFY_CRDBG")) {
                        static long nc = 0;
                        if (nc++ % 40 == 0)
                            fprintf(stderr, "[sil] silent fraction=%.3f scale=%.3f\n", frac, sc);
                    }
                    if (sc < 1.0)
                        for (int i = 0; i < Hs; i++) blk8.d[(size_t)i] *= (float)sc;
                }
            }
            /* Transient-sparsity gate: the fraction of the last 128 hops whose RMS is within
               20 dB of the window maximum. About 20% for sparse impulses and ~100% for
               drums, noise and vowels, which carry energy continuously. Below TAFFY_AUXDUTY
               the aux contribution is faded out proportionally. */
            {
                static const double DUTY = getenv("TAFFY_AUXDUTY") ? atof(getenv("TAFFY_AUXDUTY")) : 0.0;
                if (DUTY > 0.0 && DUTY <= 1.0) {
                    double sq2 = 0.0;
                    for (int i = 0; i < Hs; i++) {
                        float wv = k.wsum[(size_t)i];
                        const float fw = wv > 1e-8f ? wv : 1e-8f;
                        const double v = (double)k.ola[(size_t)i] / (double)fw;
                        sq2 += v * v;
                    }
                    const double r2 = std::sqrt(sq2 / (double)Hs);
                    if ((int)e->auxDutyRing.size() < 128) e->auxDutyRing.assign(128, 0.0f);
                    e->auxDutyRing[(size_t)(e->auxDutyPos & 127)] = (float)r2;
                    e->auxDutyPos++;
                    if (e->auxDutyPos >= 128) {
                        double mx = 0.0;
                        for (int i = 0; i < 128; i++) if (e->auxDutyRing[(size_t)i] > mx) mx = e->auxDutyRing[(size_t)i];
                        if (mx > 1e-9) {
                            const double thr2 = mx * 0.1;      /* -20 dB */
                            int hi = 0;
                            for (int i = 0; i < 128; i++) if (e->auxDutyRing[(size_t)i] >= thr2) hi++;
                            const double duty = (double)hi / 128.0;
                            if (getenv("TAFFY_CRDBG")) {
                                static long nc = 0;
                                if (nc++ % 40 == 0)
                                    fprintf(stderr, "[duty] rms=%.4g max=%.4g duty=%.3f\n", r2, mx, duty);
                            }
                            if (duty < DUTY) {
                                const double sc = duty / DUTY;
                                for (int i = 0; i < Hs; i++) blk8.d[(size_t)i] *= (float)sc;
                            }
                        }
                    }
                }
            }
            /* Kurtosis gate (fourth moment over the last W input samples): sparse impulses
               score in the thousands, drums around 20, vowels and noise below 10. The
               smoother the signal, the more aux may take over: sc = 1/(1+(kurt/K0)^p). */
            {
                static const int W = getenv("TAFFY_KURTW") ? atoi(getenv("TAFFY_KURTW")) : 16384;
                static const double KLO = getenv("TAFFY_KURTLO") ? atof(getenv("TAFFY_KURTLO")) : 20.0;
                static const double KHI = getenv("TAFFY_KURTHI") ? atof(getenv("TAFFY_KURTHI")) : 500.0;
                if (W > 0 && KHI > KLO) {
                    if ((int)e->kurtRing.size() != W) { e->kurtRing.assign((size_t)W, 0.0f); e->kurtPos = 0; }
                    const int step = Ha > 0 ? Ha : Hs;
                    for (int i = 0; i < step; i++) {
                        e->kurtRing[(size_t)(e->kurtPos & (W - 1))] =
                            k.inBuf[e->inRead + (size_t)i];
                        e->kurtPos++;
                    }
                    /* Use however many samples are available rather than waiting for the ring
                       to fill, or the first W/step hops would run ungated. */
                    const int nav = (int)std::min<long long>(e->kurtPos, (long long)W);
                    if (nav >= 1024) {
                        double m1 = 0, m2 = 0, m4 = 0;
                        for (int i = 0; i < nav; i++) {
                            const double v = (double)e->kurtRing[(size_t)i];
                            m1 += v; m2 += v * v;
                        }
                        m1 /= nav; m2 /= nav;
                        const double var = m2 - m1 * m1;
                        if (var > 1e-20) {
                            for (int i = 0; i < nav; i++) {
                                const double dv = (double)e->kurtRing[(size_t)i] - m1;
                                m4 += dv * dv * dv * dv;
                            }
                            m4 /= nav;
                            const double kurt = m4 / (var * var);
                            /* Smooth form, robust across orders of magnitude. With K0 = 225 and
                               p = 2: noise 1.8 -> 0.9999, drums 26.6 -> 0.986,
                               impulses 4094 -> 0.003. */
                            static const double K0 = getenv("TAFFY_KURTK0") ? atof(getenv("TAFFY_KURTK0")) : 225.0;
                            static const double KP = getenv("TAFFY_KURTP") ? atof(getenv("TAFFY_KURTP")) : 2.0;
                            double sc = 1.0;
                            if (K0 > 1e-9) sc = 1.0 / (1.0 + std::pow(kurt / K0, KP));
                            if (getenv("TAFFY_KURTDBG") && sc < 0.5) {
                                static long nk2 = 0;
                                if (nk2++ % 20 == 0)
                                    fprintf(stderr, "[kurt] kurtosis=%.1f scale=%.5f\n", kurt, sc);
                            }
                            if (getenv("TAFFY_KURTDBG")) {
                                static long nk = 0;
                                if (nk++ % 40 == 0)
                                    fprintf(stderr, "[kurt] kurtosis=%.1f scale=%.3f\n", kurt, sc);
                            }
                            /* Store the scale on the block so apply_aux scales the whole
                               (d - HP(delta)) contribution; scaling d alone does nothing. */
                            blk8.sc *= (float)sc;
                            if (sc <= 0.0) blk8.skip = 1;
                        }
                    }
                }
            }
            e->auxPending = true;
        }
        {
            static const float WF = getenv("TAFFY_WFLOOR") ? (float)atof(getenv("TAFFY_WFLOOR")) : 0.01f;
            float steady = 0.0f;
            if (WF > 0.0f) {
                for (int j = 0; j < N; j += Hs) steady += e->win[(size_t)j] * e->win[(size_t)j];
            }
            float floorW = WF > 0.0f ? steady * WF : 1e-8f;
            for (int i = 0; i < Hs; i++) {
                float w = k.wsum[(size_t)i];
                if (w < floorW) w = floorW;
                float v = w > 1e-8f ? k.ola[(size_t)i] / w : 0.0f;
                dst.push_back(v);
            }
        }
        std::memmove(k.ola.data(), k.ola.data() + Hs, sizeof(float) * (size_t)(N - Hs));
        std::fill(k.ola.begin() + (N - Hs), k.ola.end(), 0.0f);
        std::memmove(k.wsum.data(), k.wsum.data() + Hs, sizeof(float) * (size_t)(N - Hs));
        std::fill(k.wsum.begin() + (N - Hs), k.wsum.end(), 0.0f);
        if (e->proAuxMode == 8) {
            std::vector<float> *vv[4] = {&k.auxHpOla, &k.auxHpWsum, &k.mainHpOla, &k.mainHpWsum};
            for (int q = 0; q < 4; q++) {
                std::memmove(vv[q]->data(), vv[q]->data() + Hs, sizeof(float) * (size_t)(N - Hs));
                std::fill(vv[q]->begin() + (N - Hs), vv[q]->end(), 0.0f);
            }
        }
    }
    /* Dump the pv samples appended by this hop (any tier); TAFFY_DUMPPV=<prefix>. */
    if (const char *dp = getenv("TAFFY_DUMPPV")) {
        static FILE *fp = NULL;
        if (!fp) { char b[512]; snprintf(b, sizeof b, "%s.pv.raw", dp); fp = fopen(b, "wb"); }
        for (int c = 0; c < e->channels && fp; c++) {
            std::vector<float> &pp = e->pv[(size_t)c];
            size_t st = pp.size() >= (size_t)Hs ? pp.size() - (size_t)Hs : 0;
            fwrite(pp.data() + st, sizeof(float), pp.size() - st, fp);
        }
    }
    e->inRead += (size_t)Ha;
    e->stepsDone++;
    return true;
}

/* ===================== Pure SOLA granular path (Soloist tier) =====================
 *
 * Soloist output comes entirely from pitch-mark-driven granular overlap-add; no phase
 * vocoder is involved. Contract identical to pv_step: consumes Ha samples from
 * inBuf[inRead], appends Hs samples to pv[], so the downstream resample/flush path is
 * unchanged and pitch is still applied by the output resampler.
 *
 * Algorithm = WSOLA, grain length W = 2*Hs, overlap OV = W - Hs = Hs (Hann, 50% overlap,
 * window-square sum constant at 1):
 *   1) nominal input start = k*Ha;
 *   2) pick the offset d in [nominal-S, nominal+S] maximizing the normalized
 *      cross-correlation between the candidate grain's first OV samples and the previous
 *      grain's [Hs,W) segment (unwindowed solaTail); grains are copied sample for sample,
 *      so pitch is unchanged;
 *   3) Hann-window the grain, accumulate into solaAcc, emit its first Hs samples;
 *   4) solaTail <- this grain's [Hs,W) segment, step++, inRead += Ha.
 *
 * S comes from the pitch-mark spacing: clamp(markPeriod/2, 32, Hs), else Hs/2. Marks bound
 * the alignment search only; they do not change grain length. */
/* Forward declaration; defined later in this file. */
static double g_lastMarkC = 0.0;   /* last normalized cross-correlation peak from estimate_mark_period */
static int estimate_mark_period(const std::vector<float>& h, long long base, long long pos,
                                int minLag, int maxLag);

static std::vector<std::vector<float>> g_solaTmp;   /* per-step SOLA scratch; single-threaded, reusable */

static bool sola_step_core(taffy_engine *e) {
    const int Hs = e->Hs;
    const int Ha = e->Ha;
    /* accumulator capacity: grains vary in length, 8*Hs reaches down to ~21 Hz at 44.1 kHz */
    const int WMAX = 8 * Hs;

    static const int  SR_OVERRIDE = getenv("TAFFY_SOLASR")     ? atoi(getenv("TAFFY_SOLASR")) : -1;
    static const bool MARK_SR     = !getenv("TAFFY_SOLAMARK") || atoi(getenv("TAFFY_SOLAMARK")) != 0;
    static const int  SR_DEF      = getenv("TAFFY_SOLASRDEF")  ? atoi(getenv("TAFFY_SOLASRDEF")) : 128;
    int markPeriod = 0;
    /* Periodicity confidence gate, default 0.5. Aperiodic content (drums, noise) reaches a
       normalized cross-correlation peak of only 0.17-0.19 while periodic content reaches
       1.00, so the threshold is insensitive across 0.2-1.0. Below it the search falls back
       to the default half-width SR_DEF instead of trusting a meaningless period. */
    static const double TONGATE = getenv("TAFFY_TONGATE") ? atof(getenv("TAFFY_TONGATE")) : 0.5;

    /* ---- Pitch-mark-driven search half-width ----
       Half-width = mark spacing / 2, clamped to [32, 4096]. A fixed small radius cannot
       reach the neighbouring impulse of a pulse train, so every lag correlates near zero
       and the alignment degrades to an arbitrary pick, stacking impulses out of phase. */
    if (MARK_SR && SR_OVERRIDE < 0) {
        Chan &k0 = e->ch[0];
        if (!k0.hist.empty() && ((e->stepsDone % (e->markEvery > 0 ? e->markEvery : 16)) == 0)) {
            long long gpos = (long long)e->inFed - (long long)k0.inBuf.size() + k0.solaInPos;
            int per = estimate_mark_period(k0.hist, k0.histBase, gpos, 32, 8192);
            if (getenv("TAFFY_SOLADBG"))
                fprintf(stderr, "[sola] step=%lld gpos=%lld histBase=%lld histSize=%zu -> period=%d\n",
                        (long long)e->stepsDone, gpos, k0.histBase, k0.hist.size(), per);
            int s = per > 0 ? per / 2 : SR_DEF;
            if (s < 32) s = 32;
            if (s > 4096) s = 4096;
            for (int c = 0; c < e->channels; c++) e->ch[(size_t)c].solaSearch = s;
            for (int c = 0; c < e->channels; c++) e->ch[(size_t)c].solaTonality = (float)g_lastMarkC;
            markPeriod = per;
        }
    }

    /* ---- Grain length can also follow the marks (TAFFY_GRAINVAR=1, off by default) ----
       W = round(markPeriod * GA), clamped to [2*Hs, WMAX - Hs]. Using a fixed W = 2*Hs
       lets the marks influence only the search radius, i.e. half of their purpose. */
    int W = 2 * Hs;
    if (e->grainVar) {
        static const double GA = getenv("TAFFY_GRAINA") ? atof(getenv("TAFFY_GRAINA")) : 1.0;
        int WLO = getenv("TAFFY_GRAINLO") ? atoi(getenv("TAFFY_GRAINLO")) : 2 * Hs;
        /* Hard floor 2*Hs: the overlap OV = W - Hs must stay positive, otherwise the
           windowing and COLA normalization break down and the output diverges. */
        if (WLO < 2 * Hs) WLO = 2 * Hs;
        int wp = (markPeriod > 0) ? (int)((double)markPeriod * GA + 0.5) : 0;
        if (wp < WLO) wp = WLO;
        if (wp > WMAX - Hs) wp = WMAX - Hs;
        W = wp;
    }
    const int OV = W - Hs;                       /* overlap region */
    const int SMAX = 4096;
    if ((int)g_solaTmp.size() != e->channels) g_solaTmp.assign((size_t)e->channels, {});

    for (int c = 0; c < e->channels; c++) {
        Chan &k = e->ch[(size_t)c];
        if ((int)k.solaAcc.size() != WMAX) continue;       /* not initialized */
        /* Everything in [pos + lo, pos + hi + W) must lie inside inBuf. lo is clamped once
           more here: at step 0 with pos == 0, pos - S is negative, which would always fail. */
        int S = SR_OVERRIDE >= 0 ? SR_OVERRIDE : std::max(32, std::min(SMAX, k.solaSearch));
        if (SR_OVERRIDE < 0 && TONGATE > 0.0 && k.solaTonality < TONGATE)
            S = std::max(32, std::min(SMAX, SR_DEF));
        long long lo = std::max<long long>(-(long long)S, -k.solaInPos);
        long long hi = S;
        if (k.solaInPos + hi + W > (long long)k.inBuf.size()) return false;
        (void)lo;
    }

    for (int c = 0; c < e->channels; c++) {
        Chan &k = e->ch[(size_t)c];
        if ((int)k.solaAcc.size() != WMAX) continue;
        int S = SR_OVERRIDE >= 0 ? SR_OVERRIDE : std::max(32, std::min(SMAX, k.solaSearch));
        /* Tonality gate (TAFFY_TONGATE, 0 = off): fall back to the default half-width when
           the periodicity confidence is low. */
        if (SR_OVERRIDE < 0 && TONGATE > 0.0 && k.solaTonality < TONGATE)
            S = std::max(32, std::min(SMAX, SR_DEF));
        long long pos = k.solaInPos;
        const float *buf = k.inBuf.data();
        long long lo = std::max<long long>(-(long long)S, -pos);

        /* ---- 1) WSOLA alignment search ---- */
        int bestD = 0;
        double bestScore = -1e30;
        bool haveTail = false;
        {
            double te = 0;
            for (int i = 0; i < OV; i++) te += (double)k.solaTail[(size_t)i] * k.solaTail[(size_t)i];
            if (te > 1e-9) haveTail = true;
            if (haveTail) {
                for (long long d = lo; d <= (long long)S; d++) {
                    const float *x = buf + (size_t)(pos + d);
                    double dot = 0, xe = 0;
                    for (int i = 0; i < OV; i++) { double v = x[i]; dot += v * k.solaTail[(size_t)i]; xe += v * v; }
                    if (xe < 1e-12) continue;
                    double score = dot / std::sqrt(xe * te);   /* normalized cross-correlation */
                    if (score > bestScore) { bestScore = score; bestD = (int)d; }
                }
            }
        }
        const float *g = buf + (size_t)(pos + bestD);

        /* ---- 2) window and accumulate ---- */
        if (k.solaWinW != W) {                     /* rebuild the window when the grain length changed */
            k.solaWin.resize((size_t)W);
            for (int i = 0; i < W; i++)
                k.solaWin[(size_t)i] = 0.5f * (1.0f - std::cos(2.0f * (float)M_PI * (float)i / (float)W));
            k.solaWinW = W;
        }
        for (int i = 0; i < W; i++)
            k.solaAcc[(size_t)i] += g[i] * k.solaWin[(size_t)i];

        /* ---- 3) emit the first Hs samples ----
           No normalization divide is needed. With Hann and 50% overlap the window sums to
           1, and for a variable grain length the steady-state window sum of Hann at hop Hs
           is analytically W/(2*Hs), so gain = 2*Hs/W. Dividing by the accumulated sum of
           win^2 instead would blow up at the window edges where win -> 0. */
        const float gain = (float)(2.0 * (double)Hs / (double)W);
        std::vector<float> &dst = g_solaTmp[(size_t)c];
        dst.assign((size_t)Hs, 0.0f);
        for (int i = 0; i < Hs; i++) dst[(size_t)i] = k.solaAcc[(size_t)i] * gain;

        /* ---- 4) advance state ---- */
        for (int i = 0; i < OV; i++) k.solaTail[(size_t)i] = g[Hs + i];
        std::memmove(k.solaAcc.data(), k.solaAcc.data() + Hs, sizeof(float) * (size_t)(WMAX - Hs));
        std::fill(k.solaAcc.begin() + (WMAX - Hs), k.solaAcc.end(), 0.0f);
        k.solaInPos += Ha;
    }
    return true;      /* inRead / stepsDone are advanced by the caller */
}

/* Soloist tier: SOLA-only path. */
static bool sola_step(taffy_engine *e) {
    if (!sola_step_core(e)) return false;
    for (int c = 0; c < e->channels; c++)
        e->pv[(size_t)c].insert(e->pv[(size_t)c].end(),
                                g_solaTmp[(size_t)c].begin(), g_solaTmp[(size_t)c].end());
    e->inRead += (size_t)e->Ha;
    e->stepsDone++;
    return true;
}

/* The aux band split must not use a causal FIR: a linear-phase FIR has m/2 of group delay,
   so the correction adds out of step with the main path and interferes constructively,
   inflating the high band. A zero-phase frequency-domain split is used instead (apply_aux). */

/* aux suppression windows for Pro. The two transient kinds are controlled separately:
   burst events (directMap, i.e. attack + decay such as a drum hit) are copied over a wide
   TAFFY_AUXSUPBPRE window covering most of their high-band energy, while isolated spikes
   use a much narrower one. A single shared length cannot satisfy both. */
static bool aux_sup_covered(const taffy_engine *e, long long g0, long long g1) {
    /* check the recent-event ring first; those events may already be deleted */
    for (int k = 0; e->auxRingOn && k < 64; k++) {
        const taffy_engine::AuxEv &r = e->auxRing[k];
        if (r.outPos == 0) continue;
        long long pre, post;
        if (r.dmap) { pre = e->auxSupBurstPre;  post = e->auxSupBurstPost; }
        else        { pre = e->auxSupSpikePre;  post = e->auxSupSpikePost; }
        if (g0 < r.outPos + post && g1 > r.outPos - pre) return true;
    }
    for (size_t n = 0; n < e->pending.size(); n++) {
        const taffy_engine::TEvent &ev = e->pending[n];
        /* No confirmed check: this runs while queueing in pv_step_with_sola, before
           confirm_events, but the event will still be handled by apply_transients. */
        long long pre, post;
        if (ev.directMap) { pre = e->auxSupBurstPre;  post = e->auxSupBurstPost; }
        else              { pre = e->auxSupSpikePre;  post = e->auxSupSpikePost; }
        if (g0 < ev.outPos + post && g1 > ev.outPos - pre) return true;
    }
    return false;
}

/* Add this hop's aux correction to pv. Must run after apply_transients: aux is layered on
   top of the transient copy rather than being overwritten by it. */
/* Snapshot the queued blocks' pv content before apply_transients, so apply_aux can tell
   sample by sample whether the main path's transient copy rewrote a stretch. Where it did,
   both paths produce the same input and aux must stay out exactly, not approximately. */
void snapshot_aux(taffy_engine *e) {
    if (!e->auxPending) return;
    for (int c = 0; c < e->channels && c < (int)e->auxQ.size(); c++) {
        std::vector<float> &p = e->pv[(size_t)c];
        for (size_t k = 0; k < e->auxQ[(size_t)c].size(); k++) {
            taffy_engine::AuxBlk &blk = e->auxQ[(size_t)c][k];
            const size_t b = blk.base;
            const int n = (int)std::min((size_t)e->Hs, p.size() > b ? p.size() - b : 0);
            blk.snap.assign((size_t)(n > 0 ? n : 0), 0.0f);
            for (int i = 0; i < n; i++) blk.snap[(size_t)i] = p[b + (size_t)i];
        }
    }
}

void apply_aux(taffy_engine *e) {
    if (getenv("TAFFY_AUXDBG")) {
        static int n = 0;
        if (n++ < 5) {
            size_t tot = 0;
            for (size_t q = 0; q < e->auxQ.size(); q++) tot += e->auxQ[q].size();
            double sq = 0; size_t nn = 0;
            for (size_t q = 0; q < e->auxQ.size(); q++)
                for (size_t r = 0; r < e->auxQ[q].size(); r++)
                    for (size_t z = 0; z < e->auxQ[q][r].d.size(); z++) { sq += (double)e->auxQ[q][r].d[z]*e->auxQ[q][r].d[z]; nn++; }
            fprintf(stderr, "[aux] pending=%d mode=%d blocks=%zu RMS(d)=%.6g gain=%.3f\n",
                    (int)e->auxPending, e->proAuxMode, tot,
                    nn ? std::sqrt(sq/nn) : 0.0, (double)e->auxGain);
        }
    }
    if (!e->auxPending) return;
    e->auxPending = false;
    if (e->proAuxMode <= 0) { for (size_t c = 0; c < e->auxQ.size(); c++) e->auxQ[c].clear(); return; }
    if (e->proAuxMode == 8 && e->auxPipe) {
        /* Parallel-OLA aux plus frame-level HP(delta) (experimental, off by default).
           The queue holds HP(aux) - HP(main). HP(delta) must use the same frame-level
           operator: a frame [base, base+N) needs the three future hops of delta, so the
           whole delta OLA lags by 4 hops. Results are written back through a global index,
           because pvTrimmed moves and a queued base would go stale. */
        const int Hs = e->Hs, N = e->N;
        if ((int)e->auxPend.size() < e->channels) {
            e->auxPend.assign((size_t)e->channels, std::vector<taffy_engine::AuxPend>());
            e->auxDeltaHist.assign((size_t)e->channels, std::vector<float>());
            e->auxDHpOla.assign((size_t)e->channels, std::vector<float>());
            e->auxDHpWsum.assign((size_t)e->channels, std::vector<float>());
            e->auxDRe.assign((size_t)e->channels, std::vector<float>());
            e->auxDIm.assign((size_t)e->channels, std::vector<float>());
        }
        const double fcx8 = (e->auxFcBin >= 0 ? e->auxFcBin : e->auxCrossBin) >= 0
                                ? (double)(e->auxFcBin >= 0 ? e->auxFcBin : e->auxCrossBin)
                                      * (double)e->sampleRate / (double)e->N
                                : 1500.0;
        float steady8 = 0.0f;
        for (int j = 0; j < N; j += Hs) steady8 += e->win[(size_t)j] * e->win[(size_t)j];
        const float floorW8b = steady8 * 0.01f;
        for (int c = 0; c < e->channels && c < (int)e->auxQ.size(); c++) {
            std::vector<float> &p = e->pv[(size_t)c];
            Chan &kc = e->ch[(size_t)c];
            std::vector<float> &hist = e->auxDeltaHist[(size_t)c];
            if ((int)hist.size() != 4 * Hs) hist.assign((size_t)(4 * Hs), 0.0f);
            std::vector<float> &re2 = e->auxDRe[(size_t)c];
            std::vector<float> &im2 = e->auxDIm[(size_t)c];
            std::vector<float> &dO  = e->auxDHpOla[(size_t)c];
            std::vector<float> &dW  = e->auxDHpWsum[(size_t)c];
            if ((int)re2.size() != N) { re2.assign((size_t)N, 0.0f); im2.assign((size_t)N, 0.0f); }
            if ((int)dO.size()  != N) { dO.assign((size_t)N, 0.0f);  dW.assign((size_t)N, 0.0f); }
            for (size_t q = 0; q < e->auxQ[(size_t)c].size(); q++) {
                taffy_engine::AuxBlk &b8 = e->auxQ[(size_t)c][q];
                if (b8.skip) continue;            /* gate fully closed: skip this hop */
                float tgt = 1.0f;
                if (e->auxGateSrc == 1) {
                    if (b8.cov) kc.sinceGate = 0;
                    else if (kc.sinceGate < 1000000) kc.sinceGate++;
                    tgt = (kc.sinceGate < e->auxHold) ? e->auxGateTransient : 1.0f;
                }
                kc.auxG += e->auxGateSm * (tgt - kc.auxG);
                /* delta = the per-sample correction apply_transients applied to this hop */
                std::vector<float> dcur((size_t)Hs, 0.0f);
                for (int i = 0; i < Hs && b8.base + (size_t)i < p.size(); i++) {
                    const size_t idx = b8.base + (size_t)i;
                    if ((size_t)i < b8.snap.size()) dcur[(size_t)i] = p[idx] - b8.snap[(size_t)i];
                }
                std::memmove(hist.data(), hist.data() + Hs, sizeof(float) * (size_t)(3 * Hs));
                for (int i = 0; i < Hs; i++) hist[(size_t)(3 * Hs + i)] = dcur[(size_t)i];
                taffy_engine::AuxPend pe;
                pe.gBase = (long long)e->pvTrimmed + (long long)b8.base;
                pe.d = b8.d;
                pe.cov = b8.cov;
                pe.g = kc.auxG * e->auxGain;
                e->auxPend[(size_t)c].push_back(pe);
            }
            /* Drain while fewer than 4 hops are pending: a frame needs the last 4 hops of delta. */
            while ((int)e->auxPend[(size_t)c].size() >= 4 ||
                   (e->auxDrain && !e->auxPend[(size_t)c].empty())) {
                taffy_engine::AuxPend pj = e->auxPend[(size_t)c].front();
                e->auxPend[(size_t)c].erase(e->auxPend[(size_t)c].begin());
                for (int i = 0; i < N; i++) {
                    re2[(size_t)i] = (i < 4 * Hs ? hist[(size_t)i] : 0.0f) * e->win[(size_t)i];
                    im2[(size_t)i] = 0.0f;
                }
                taffy::fft(re2, im2, false);
                const std::vector<float> *MK = aux_mask(N, (double)e->sampleRate, fcx8);
                for (int b = 0; b < N; b++) { re2[(size_t)b] *= (*MK)[(size_t)b];
                                              im2[(size_t)b] *= (*MK)[(size_t)b]; }
                taffy::fft(re2, im2, true);
                for (int i = 0; i < N; i++) {
                    const float w = e->win[(size_t)i];
                    dO[(size_t)i] += re2[(size_t)i] * w;
                    dW[(size_t)i] += w * w;
                }
                long long gb = pj.gBase;
                long long rel = gb - (long long)e->pvTrimmed;
                if (getenv("TAFFY_PDBG")) {
                    static long na = 0, nskip = 0, nneg = 0;
                    na++;
                    if (rel < 0) nneg++;
                    if (rel + Hs > (long long)p.size()) nskip++;
                    if (na % 40 == 0)
                        fprintf(stderr, "[pipe] applied=%ld rel<0=%ld oob=%ld pending=%zu pvTrimmed=%zu psize=%zu g=%.3f\n",
                                na, nneg, nskip, e->auxPend[(size_t)c].size(),
                                (size_t)e->pvTrimmed, p.size(), (double)pj.g);
                }
                for (int i = 0; i < Hs; i++) {
                    if (rel + i < 0) continue;
                    const size_t idx = (size_t)(rel + (long long)i);
                    if (idx >= p.size()) break;
                    float wv = dW[(size_t)i];
                    if (wv < floorW8b) wv = floorW8b;
                    const float hpD = dO[(size_t)i] / wv;
                    const float dv = ((size_t)i < pj.d.size()) ? pj.d[(size_t)i] : 0.0f;
                    p[idx] += (dv - hpD) * pj.g;
                }
                std::memmove(dO.data(), dO.data() + Hs, sizeof(float) * (size_t)(N - Hs));
                std::fill(dO.begin() + (N - Hs), dO.end(), 0.0f);
                std::memmove(dW.data(), dW.data() + Hs, sizeof(float) * (size_t)(N - Hs));
                std::fill(dW.begin() + (N - Hs), dW.end(), 0.0f);
            }
            e->auxQ[(size_t)c].clear();
        }
        return;
    }
    if (e->proAuxMode == 8) {
        /* Block-level HP(delta) (auxActual) plus gating. */
        for (int c = 0; c < e->channels && c < (int)e->auxQ.size(); c++) {
            std::vector<float> &p = e->pv[(size_t)c];
            Chan &kc = e->ch[(size_t)c];
            for (size_t q = 0; q < e->auxQ[(size_t)c].size(); q++) {
                taffy_engine::AuxBlk &b8 = e->auxQ[(size_t)c][q];
                if (b8.skip) continue;            /* gate fully closed: skip this hop */
                float tgt = 1.0f;
                if (e->auxGateSrc == 1) {
                    if (b8.cov) kc.sinceGate = 0;
                    else if (kc.sinceGate < 1000000) kc.sinceGate++;
                    tgt = (kc.sinceGate < e->auxHold) ? e->auxGateTransient : 1.0f;
                }
                kc.auxG += e->auxGateSm * (tgt - kc.auxG);
                const float g = kc.auxG * e->auxGain;
                if ((int)e->auxDeltaPrev.size() < e->channels)
                    e->auxDeltaPrev.assign((size_t)e->channels, std::vector<float>());
                std::vector<float> &dhist = e->auxDeltaPrev[(size_t)c];
                if ((int)dhist.size() != 3 * e->Hs) dhist.assign((size_t)(3 * e->Hs), 0.0f);
                std::vector<float> dcur((size_t)e->Hs, 0.0f);
                bool anyDelta = false;
                for (int i = 0; i < e->Hs && b8.base + (size_t)i < p.size(); i++) {
                    const size_t idx = b8.base + (size_t)i;
                    if ((size_t)i < b8.snap.size()) {
                        dcur[(size_t)i] = p[idx] - b8.snap[(size_t)i];
                        if (dcur[(size_t)i] != 0.0f) anyDelta = true;
                    }
                }
                std::vector<float> hpDelta((size_t)e->Hs, 0.0f);
                if (e->auxActual && anyDelta) {
                    const int Hs2 = e->Hs;
                    int M2 = 8 * Hs2;
                    { int pw = 1; while (pw < M2) pw <<= 1; M2 = pw; }
                    const int off2 = M2 / 2 - Hs2;
                    std::vector<float> re2((size_t)M2, 0.0f), im2((size_t)M2, 0.0f);
                    for (int i = 0; i < 3 * Hs2; i++) re2[(size_t)(off2 + i)] = dhist[(size_t)i];
                    for (int i = 0; i < Hs2; i++) re2[(size_t)(off2 + 3 * Hs2 + i)] = dcur[(size_t)i];
                    taffy::fft(re2, im2, false);
                    const double fcx2 = e->auxCrossBin >= 0
                                            ? (double)e->auxCrossBin * (double)e->sampleRate / (double)e->N
                                            : 1500.0;
                    const std::vector<float> *MK = aux_mask(M2, (double)e->sampleRate, fcx2);
                    for (int b = 0; b < M2; b++) { re2[(size_t)b] *= (*MK)[(size_t)b];
                                                   im2[(size_t)b] *= (*MK)[(size_t)b]; }
                    taffy::fft(re2, im2, true);
                    for (int i = 0; i < Hs2; i++) hpDelta[(size_t)i] = re2[(size_t)(M2 / 2 + 2 * Hs2 + i)];
                }
                {
                    const int Hh = e->Hs;
                    std::memmove(dhist.data(), dhist.data() + Hh, sizeof(float) * (size_t)(2 * Hh));
                    for (int i = 0; i < Hh; i++) dhist[(size_t)(2 * Hh + i)] = dcur[(size_t)i];
                }
                for (int i = 0; i < e->Hs && b8.base + (size_t)i < p.size(); i++) {
                    const size_t idx = b8.base + (size_t)i;
                    if (e->auxExact && (size_t)i < b8.snap.size() &&
                        p[idx] != b8.snap[(size_t)i]) continue;
                    p[idx] += (b8.d[(size_t)i] - (e->auxActual ? hpDelta[(size_t)i] : 0.0f))
                              * g * b8.sc * b8.alpha;
                }
            }
            e->auxQ[(size_t)c].clear();
        }
        return;
    }
    const float fc = e->auxCrossBin >= 0
                         ? (float)((double)e->auxCrossBin * (double)e->sampleRate / (double)e->N)
                         : 1500.0f;
    for (int c = 0; c < e->channels && c < (int)e->auxQ.size(); c++) {
        std::vector<float> &p = e->pv[(size_t)c];
        Chan &kc = e->ch[(size_t)c];
        if ((int)e->auxDPrev.size() < e->channels)
            e->auxDPrev.assign((size_t)e->channels, std::vector<float>());
        std::vector<float> &prev = e->auxDPrev[(size_t)c];
        if ((int)prev.size() != e->Hs) prev.assign((size_t)e->Hs, 0.0f);
        const int Hs = e->Hs;
        /* M must be a power of two: taffy::fft requires it. Heavy compression makes
           rebuild_hopped pick a non-power-of-two Hs (e.g. 384), so 4*Hs would produce
           garbage that feeds back through d = sv - p and diverges. Round up. */
        int M = 4 * Hs;
        { int pw = 1; while (pw < M) pw <<= 1; M = pw; }
        std::vector<float> re((size_t)M), im((size_t)M);
        for (size_t k = 0; k < e->auxQ[(size_t)c].size(); k++) {
            taffy_engine::AuxBlk &blk = e->auxQ[(size_t)c][k];
            if ((int)blk.d.size() < Hs) continue;
            float tgt = 1.0f;
            if (e->auxGateSrc == 1) {
                if (blk.cov) kc.sinceGate = 0;
                else if (kc.sinceGate < 1000000) kc.sinceGate++;
                tgt = (kc.sinceGate < e->auxHold) ? e->auxGateTransient : 1.0f;
            }
            kc.auxG += e->auxGateSm * (tgt - kc.auxG);
            const float g = kc.auxG * e->auxGain;
            /* Zero-phase band split. A causal linear-phase FIR would add m/2 of group delay,
               so the correction would land out of step with the main path and interfere
               constructively, inflating the high band. overlap-save: the block holds the
               previous and the current hop of d, zero-padded to 4*Hs and multiplied by a
               real even zero-phase mask; the second half [Hs,2Hs) is the wrap-free result. */
            /* The data must be centred: the mask's impulse response is symmetric about 0,
               so its negative-delay taps wrap to the end of the buffer. With the data at
               [0,2Hs) those taps would read padding instead of the previous hop. Centred,
               they read [M/2-Hs, ...) and the clean output is [M/2, M/2+Hs). */
            const int off = M / 2 - Hs;
            for (int i = 0; i < M; i++) { re[(size_t)i] = 0.0f; im[(size_t)i] = 0.0f; }
            /* Windowing is required: with a rectangular block the sinc leakage of a pure
               tone spreads over the whole spectrum and the mask cannot remove the part
               that already reached the high band. Tukey(alpha = 0.5) tapers only the outer
               Hs/2 on each side and leaves the middle (the current hop) at 1, so no
               hop-rate amplitude modulation is introduced. */
            const int tap = Hs / 2;
            for (int i = 0; i < Hs; i++) {
                float wv = 1.0f;
                if (i < tap) wv = 0.5f * (1.0f - std::cos((float)M_PI * (float)i / (float)tap));
                re[(size_t)(off + i)] = prev[(size_t)i] * wv;
            }
            for (int i = 0; i < Hs; i++) {
                float wv = 1.0f;
                if (i >= Hs - tap)
                    wv = 0.5f * (1.0f - std::cos((float)M_PI * (float)(Hs - 1 - i) / (float)tap));
                re[(size_t)(off + Hs + i)] = blk.d[(size_t)i] * wv;
            }
            taffy::fft(re, im, false);
            const double df = (double)e->sampleRate / (double)M;
            for (int b = 0; b < M / 2 + 1; b++) {
                const double f = (double)b * df;
                float m = 1.0f;
                if (f < (double)fc - 100.0) m = 0.0f;
                else if (f < (double)fc + 100.0) {
                    const double t = (f - ((double)fc - 100.0)) / 200.0;
                    m = (float)(0.5 - 0.5 * std::cos(M_PI * t));
                }
                re[(size_t)b] *= m; im[(size_t)b] *= m;
                if (b > 0 && b < M / 2) {
                    re[(size_t)(M - b)] *= m; im[(size_t)(M - b)] *= m;
                }
            }
            taffy::fft(re, im, true);
            for (int i = 0; i < Hs && blk.base + (size_t)i < p.size(); i++) {
                const size_t idx = blk.base + (size_t)i;
                if (e->auxExact && (size_t)i < blk.snap.size() &&
                    p[idx] != blk.snap[(size_t)i]) continue;
                p[idx] += re[(size_t)(M / 2 + i)] * g;
            }
            prev.assign(blk.d.begin(), blk.d.begin() + Hs);
        }
        e->auxQ[(size_t)c].clear();
    }
}

/* Pro tier: phase vocoder with a SOLA layer on top. SOLA and the phase vocoder emit the
   same Hs samples on the same hop grid; e->proAuxMode selects how they combine (see that
   field's comment in taffy_engine). */
static bool pv_step_with_sola(taffy_engine *e) {
    std::vector<size_t> base((size_t)e->channels);
    for (int c = 0; c < e->channels; c++) base[(size_t)c] = e->pv[(size_t)c].size();
    if (!pv_step(e)) return false;
    /* mode 8 must not run SOLA. Its aux source is the input frame, and sola_step_core
       disturbs state shared with apply_transients/drop_consumed, so mode 8 output would
       differ from mode 1 even at zero aux gain. */
    if (e->proAuxMode == 8) return true;
    if (getenv("TAFFY_NOSOLA")) return true;            /* diagnostic: skip SOLA entirely */
    if (!sola_step_core(e)) return true;              /* fall back to pure PV if SOLA is unavailable */
    /* Combination modes, all selected by proAuxMode (TAFFY_PROAUXMODE overrides):
        0 = off (pure PV)
        1 = mix by solaMix
        2 = full-band addition
        3 = add only the high band of SOLA
        4 = complementary replace, PV + HP(SOLA - PV)
       Modes 3 and 4 differ only by the -HP(PV) term, which is the same thing as zeroing
       the main path on the aux bins. */
    const int am = e->proAuxMode;
    if (am <= 0) return true;
    const float SOLAG = e->auxGain;
    const float w = e->solaMix, iw = 1.0f - w;
    for (int c = 0; c < e->channels; c++) {
        std::vector<float> &p = e->pv[(size_t)c];
        const std::vector<float> &sv = g_solaTmp[(size_t)c];
        if (sv.size() < (size_t)e->Hs) continue;
        if (p.size() < base[(size_t)c] + (size_t)e->Hs) continue;
        if (am == 3 || am == 4 || am == 6) {
            if ((int)e->auxQ.size() < e->channels) e->auxQ.resize((size_t)e->channels);
            e->auxQ[(size_t)c].emplace_back();
            taffy_engine::AuxBlk &blk = e->auxQ[(size_t)c].back();
            blk.base = base[(size_t)c];
            if (e->auxGateSrc == 1) {
                const long long gg = (long long)e->pvTrimmed + (long long)blk.base;
                for (size_t q = 0; q < e->pending.size(); q++) {
                    const taffy_engine::TEvent &E2 = e->pending[q];
                    bool dup = false;
                    for (int k = 0; e->auxRingOn && k < 64; k++)
                        if (e->auxRing[k].outPos == E2.outPos &&
                            e->auxRing[k].dmap == (E2.directMap ? 1 : 0)) { dup = true; break; }
                    if (!dup && e->auxRingOn) {
                        e->auxRing[e->auxRingPos & 63].outPos = E2.outPos;
                        e->auxRing[e->auxRingPos & 63].dmap = E2.directMap ? 1 : 0;
                        e->auxRingPos++;
                    }
                }
                blk.cov = aux_sup_covered(e, gg, gg + (long long)e->Hs) ? 1 : 0;
            }
            std::vector<float> &dd = blk.d;
            dd.assign((size_t)e->Hs, 0.0f);
            Chan &kc = e->ch[(size_t)c];
            for (int i = 0; i < e->Hs; i++) {
                float src_i;
                if (am == 6) {
                    const double is = e->internalStretch > 1e-6 ? (double)e->internalStretch : 1.0;
                    const double sp = ((double)base[(size_t)c] + (double)i) / is;
                    const long long si = (long long)std::floor(sp);
                    const double fr = sp - (double)si;
                    float a0 = 0.0f, a1 = 0.0f;
                    const long long hb = kc.histBase;
                    const long long he = hb + (long long)kc.hist.size();
                    if (si >= hb && si < he) a0 = kc.hist[(size_t)(si - hb)];
                    if (si + 1 >= hb && si + 1 < he) a1 = kc.hist[(size_t)(si + 1 - hb)];
                    src_i = (float)(a0 + fr * (a1 - a0));
                } else {
                    src_i = (am == 4) ? p[(size_t)(base[(size_t)c] + (size_t)i)] : 0.0f;
                }
                dd[(size_t)i] = sv[(size_t)i] - src_i;
            }
            /* Dump the aux source (SOLA) and the main PV segment; TAFFY_DUMPAUX=<prefix>. */
            if (const char *dp = getenv("TAFFY_DUMPAUX")) {
                static FILE *fs = NULL, *fp = NULL;
                if (!fs) {
                    char b1[512], b2[512];
                    snprintf(b1, sizeof b1, "%s.sv.raw", dp);
                    snprintf(b2, sizeof b2, "%s.pv.raw", dp);
                    fs = fopen(b1, "wb"); fp = fopen(b2, "wb");
                }
                if (fs) fwrite(sv.data(), sizeof(float), (size_t)e->Hs, fs);
                if (fp) fwrite(p.data() + base[(size_t)c], sizeof(float), (size_t)e->Hs, fp);
            }
            e->auxPending = true;
        } else {
            for (int i = 0; i < e->Hs; i++) {
                size_t idx = base[(size_t)c] + (size_t)i;
                if (am >= 2) p[idx] = p[idx] + sv[(size_t)i] * SOLAG;
                else         p[idx] = p[idx] * iw + sv[(size_t)i] * w;
            }
        }
    }
    return true;
}

/* Set up / reset the SOLA state. Capacity uses the maximum grain length 8*Hs because grains
   vary; the window actually used is decided and cached per step by sola_step_core. */
static void sola_init(taffy_engine *e) {
    const int Hs = e->Hs;
    for (int c = 0; c < e->channels; c++) {
        Chan &k = e->ch[(size_t)c];
        const int WMAX = 8 * Hs;                  /* grains vary; size the buffers for the maximum */
        k.solaAcc.assign((size_t)WMAX, 0.0f);
        k.solaWsum.assign((size_t)WMAX, 0.0f);
        k.solaTail.assign((size_t)WMAX, 0.0f);
        k.solaWin.resize((size_t)(2 * Hs));
        for (int i = 0; i < 2 * Hs; i++)
            k.solaWin[(size_t)i] = 0.5f * (1.0f - std::cos(2.0f * (float)M_PI * (float)i / (float)(2 * Hs)));
        k.solaWinW = 2 * Hs;
        k.solaInPos = 0;
        k.solaSearch = 0;
    }
}

/* Single "run one step" entry point, choosing the signal path by tier:
   Efficient / Pro -> phase vocoder (Pro adds the SOLA layer on top);
   Soloist         -> pure SOLA, no phase vocoder. */
static inline bool step_once(taffy_engine *e) {
    if (e->solaTier == taffy_engine::SOLA_MONO) return sola_step(e);
    if (e->solaTier == taffy_engine::SOLA_POLY && e->proAuxMode > 0) return pv_step_with_sola(e);
    return pv_step(e);
}

/* Time-domain transient detection: block-level envelope jumps plus intra-block |x| peaks. */
/* Pitch-mark position extraction, as opposed to estimate_mark_period, which returns a single
   period scalar. Each mark is placed at the |x| maximum within +/-P/2 of its prediction, so
   the spacing between neighbouring marks follows the waveform instead of staying exactly P.
   Returns the number of marks written to out[]. */
static int extract_marks(const std::vector<float>& h, long long base,
                         long long from, long long to, int period,
                         long long *out, int maxOut) {
    if (period < 8 || maxOut <= 0) return 0;
    if (from < base) from = base;
    if (to > base + (long long)h.size()) to = base + (long long)h.size();
    int n = 0;
    long long pred = from;
    while (n < maxOut && pred < to) {
        long long lo = pred - period / 2, hi = pred + period / 2;
        if (lo < base) lo = base;
        if (hi > to) hi = to;
        if (hi - lo < 4) break;
        long long best = lo; double bestv = -1.0;
        for (long long i = lo; i < hi; i++) {
            double v = std::fabs((double)h[(size_t)(i - base)]);
            if (v > bestv) { bestv = v; best = i; }
        }
        if (bestv <= 0.0) break;
        out[n++] = best;
        pred = best + period;               /* predict the next mark */
    }
    return n;
}

/* Half the distance between the mark straddling pos and the previous one, i.e.
   (current mark - previous mark) >> 1. Returns 0 when too few marks are found. */
static int mark_spacing_half(const std::vector<float>& h, long long base, long long pos,
                             int period, int maxLag) {
    if (period < 8) return 0;
    const int MAXM = 160;
    long long marks[MAXM];
    long long from = pos - (long long)period * 40;
    long long to   = pos + (long long)period * 4;
    if (from < base) from = base;
    int n = extract_marks(h, base, from, to, period, marks, MAXM);
    if (n < 2) { (void)maxLag; return 0; }
    /* the last mark <= pos is the current one, the one before it is the previous one */
    int k = -1;
    for (int i = 0; i < n; i++) if (marks[i] <= pos) k = i;
    if (k < 1) return 0;
    long long d = marks[k] - marks[k - 1];
    if (d <= 0) return 0;
    (void)maxLag;
    return (int)(d / 2);
}

/* Mark-period estimator: autocorrelates a W-sample window of the input history and picks
   the strongest peak for lag in [minLag, maxLag]. Causal, using only samples before pos.
   Returns the period in samples, or 0 when no pitch is found. */
/* Global mark-extraction switch, on by default; TAFFY_MARKW=0 disables it entirely. */
static bool mark_enabled() {
    static const bool on = !getenv("TAFFY_MARKW") || atoi(getenv("TAFFY_MARKW")) != 0;
    return on;
}

/* Causal mark-period estimator with a self-chosen window: a fixed window over sparse
   content such as a pulse train usually lands in silence, so the highest-energy W-sample
   window in the available past is located first and used as the autocorrelation base. */
static int estimate_mark_period(const std::vector<float>& h, long long base, long long pos,
                                int minLag, int maxLag) {
    /* Mark analysis window length in samples (TAFFY_MARKWIN, default 512). */
    static const int W = getenv("TAFFY_MARKWIN") ? atoi(getenv("TAFFY_MARKWIN")) : 512;
    /* The available region must exceed maxLag + W, otherwise the window has no room to
       slide: the selection loop would never run and a window landing in silence would
       return 0 for every input. */
    long long need = 2LL * maxLag + W;
    if (getenv("TAFFY_MARKDBG"))
        fprintf(stderr, "[p] pos=%lld base=%lld histBase=%lld histSize=%zu need=%lld %s\n",
                pos, base, base, h.size(), need, (pos - need < base) ? "range too short -> 0" : "ok");
    if (pos - need < base) return 0;
    long long lo = pos - need;
    const float* p0 = &h[(size_t)(lo - base)];
    /* pick the highest-energy window */
    double e = 0; for (int i = 0; i < W; i++) e += (double)p0[i] * p0[i];
    double bestE = e; int bestOff = 0;
    long long iters = 0;
    for (long long off = 1; off + W + maxLag <= need; off += 64) {
        iters++;
        /* Recompute the whole window: with a stride of 64, a sliding update over the two
           end samples would treat a 64-sample change as a 2-sample one. */
        double ee = 0;
        for (int i = 0; i < W; i++) ee += (double)p0[off + i] * p0[off + i];
        e = ee;
        if (e > bestE) { bestE = e; bestOff = (int)off; }
    }
    if (getenv("TAFFY_MARKDBG")) {
        double mx = 0; int mxi = -1;
        for (long long i = 0; i < need; i++) { double a = std::fabs(p0[i]); if (a > mx) { mx = a; mxi = (int)i; } }
        fprintf(stderr, "[p]   bestE=%.6g bestOff=%d iters=%lld lastE=%.6g regionMax|x|=%.4g @%d\n",
                bestE, bestOff, iters, e, mx, mxi);
    }
    if (bestE < 1e-12) return 0;
    const float* x = p0 + bestOff;      /* window [lo+bestOff, +W) */
    double e0 = 0; for (int i = 0; i < W; i++) e0 += (double)x[i] * x[i];
    if (e0 < 1e-12) return 0;
    /* Correlation threshold (TAFFY_MARKTHR), default 0, i.e. "always return the strongest
       peak whatever the correlation". A positive threshold makes unpitched content return 0,
       which collapses the period and degenerates the mark grid. */
    static const double MTHR = getenv("TAFFY_MARKTHR") ? atof(getenv("TAFFY_MARKTHR")) : 0.0;
    /* Take the first strong peak, not the global maximum: a periodic signal correlates near
       1 at every multiple of its period, so a plain argmax picks the largest multiple (a
       440 Hz sine with period 100.2 is reported as 2205). Harmless while the spacing only
       set the search radius, fatal once it drives the grain length. */
    static const double TOL = getenv("TAFFY_MARKTOL") ? atof(getenv("TAFFY_MARKTOL")) : 0.9;
    int nlag = maxLag - minLag + 1;
    if (nlag <= 0) return 0;
    static std::vector<float> cs;
    cs.assign((size_t)nlag, 0.0f);
    double bestC = MTHR; int bestLag = 0;
    for (int lag = minLag; lag <= maxLag; lag++) {
        if ((long long)bestOff + lag + W > need) break;
        double dot = 0, e1 = 0;
        for (int i = 0; i < W; i++) { double a = x[i], b = x[i + lag]; dot += a * b; e1 += b * b; }
        if (e1 < 1e-12) continue;
        double c = dot / std::sqrt(e0 * e1);
        cs[(size_t)(lag - minLag)] = (float)c;
        if (c > bestC) { bestC = c; bestLag = lag; }
    }
    int firstPeak = bestLag;
    if (bestLag > 0) {
        double thr = std::max(MTHR, TOL * bestC);
        for (int lag = minLag; lag <= maxLag; lag++) {
            double c = (double)cs[(size_t)(lag - minLag)];
            if (c < thr) continue;
            double cp = (lag > minLag) ? (double)cs[(size_t)(lag - 1 - minLag)] : -2.0;
            double cn = (lag < maxLag) ? (double)cs[(size_t)(lag + 1 - minLag)] : -2.0;
            if (c >= cp && c >= cn) { firstPeak = lag; break; }
        }
    }
    g_lastMarkC = bestC;
    if (getenv("TAFFY_MARKDBG"))
        fprintf(stderr, "[p]   -> bestC=%.3f bestLag=%d -> firstStrongPeak=%d (tol=%.2f)\n",
                bestC, bestLag, firstPeak, TOL);
    /* tonality: periodicity confidence, here the normalized cross-correlation peak in [0,1] */
    if (getenv("TAFFY_TONDBG"))
        fprintf(stderr, "[ton] bestLag=%d bestC=%.4f\n", bestLag, bestC);
    return firstPeak;
}

long long detect_transients(taffy_engine *e, long long from, long long to) {
    /* at internalStretch == 1 the engine is already transparent; a time-domain copy would only add duplication and gain error */
    if (std::fabs(e->internalStretch - 1.0f) < 1e-3f) return to;
    if (e->transientGain <= 0.0f) return to;   /* TAFFY_TGAIN=0 disables the copy path */
    /* The Soloist tier must skip this layer: it models the phase vocoder's own transient
       repair, and Soloist has no phase vocoder, its output being granular overlap-add only.
       Leaving it on would write every impulse twice, once from the SOLA grain copy and once
       at transientGain = 1.35, overshooting the peak by exactly that factor. */
    if (e->solaTier == taffy_engine::SOLA_MONO) return to;
    /* No per-tier gating is applied: this repair models the phase vocoder's built-in
       transient handling, which all three tiers share. */
    const int BLK = 64;
    Chan &c0 = e->ch[0];
    long long p = from;
    for (; p + BLK <= to; p += BLK) {
        long long idx = p - c0.histBase;
        if (idx < 0 || (size_t)(idx + BLK) > c0.hist.size()) break;
        double sum = 0;
        for (int j = 0; j < BLK; j++) {
            double v = c0.hist[(size_t)(idx + j)];
            sum += v * v;
        }
        float rms = (float)std::sqrt(sum / BLK);
        /* "Decaying local event" test: a real transient is well down 512 samples later while
           steady content such as a vowel is not, so a periodic signal is not misdetected
           once per period. */
        /* Look 1024 samples ahead, spanning several pitch periods, to avoid an accidental
           underestimate from beating between the block rate and the signal period. */
        double rmsLater = 0;
        {
            long long q = p + 128;
            const int LWIN = 1024;
            if (q + LWIN <= to + 256 + (getenv("TAFFY_CFWIN") ? atoi(getenv("TAFFY_CFWIN")) : 1024) && q >= c0.histBase
                && (long long)(q + LWIN) <= c0.histBase + (long long)c0.hist.size()) {
                double e2 = 0;
                for (int i = 0; i < LWIN; i++) { float v = c0.hist[(size_t)(q + i - c0.histBase)]; e2 += (double)v * v; }
                rmsLater = sqrt(e2 / LWIN);
            }
        }
        /* Two-stage detection: only the early "energy jump" test runs here, so the event is
           created without applying grains yet and confirm_events() verifies the decay 1280
           samples later. That way coverage starts before the smear region is consumed. */
        if (rms > 0.02f && rms > e->envPrev * 2.5f && (p - e->lastTransientIn) > 512) {
            long long tp = p;
            float best = 0.0f;
            for (long long q = p; q < p + BLK; q++) {
                float a = std::fabs(c0.hist[(size_t)(q - c0.histBase)]);
                if (a > best) { best = a; tp = q; }
            }
            int win0 = e->transientWin > 0 ? e->transientWin : (e->N / 2);
            if (win0 > 4096) win0 = 4096;
            /* SOLA reads +/-solaPre/stretch of input, so the source span must widen too, or the grain read would fall out of range */
            int pre = win0, post = win0;
            if (e->transientSolaPre > 0) {
                int a = (int)std::ceil((double)e->transientSolaPre / (double)e->internalStretch) + 8;
                int b = (int)std::ceil((double)e->transientSolaPost / (double)e->internalStretch) + 8;
                if (a > pre) pre = a;
                if (b > post) post = b;
            }
            long long lo = tp - pre, hi = tp + post;
            bool ok = true;
            /* Availability only has to satisfy the copy; SOLA grains read hist live and are
               not limited by the snapshot span. */
            for (int c = 0; c < e->channels; c++) {
                Chan &k = e->ch[(size_t)c];
                if (tp + win0 >= (long long)(k.histBase + (long long)k.hist.size())) ok = false;
            }
            if (getenv("TAFFY_TDBG"))
                fprintf(stderr, "[t] detect p=%lld tp=%lld rms=%.4f later=%.4f ratio=%.2f envPrev=%.4f ok=%d\n",
                        p, tp, rms, rmsLater, rmsLater > 0 ? rms / rmsLater : -1.0, e->envPrev, (int)ok);
            if (ok) {
                taffy_engine::TEvent ev;
                ev.win = win0;
                ev.sup = e->transientSuppress;
                ev.directMap = 0;
                ev.mapNoRightTaper = 0;
                ev.mapQ = 0;
                ev.grainInAcc = 0.0;
                ev.absGrid = 0;
                ev.markPeriod = 0;
                ev.gain = e->transientCopy ? e->transientGain : 0.0f;   /* copy layer only; suppression via ev.sup is unaffected */
                ev.stretch = e->internalStretch;
                ev.resample = e->transientResample;
                ev.tp = tp;
                ev.inBase = lo;                            /* segment base is always lo; samples below histBase are zero-padded */
                ev.outPos = (long long)std::llround((double)tp * (double)e->internalStretch);
                ev.supLo = ev.outPos - ev.sup;          /* must be computed after outPos */
                ev.supHi = ev.outPos + ev.win + 1;
                ev.confirmed = 0;
                ev.confirmAt = tp
                    + (getenv("TAFFY_CFSTART") ? atoi(getenv("TAFFY_CFSTART")) : 256)
                    + (getenv("TAFFY_CFWIN") ? atoi(getenv("TAFFY_CFWIN")) : 384);
                ev.grainW = e->transientSolaW;
                if (ev.grainW < 32) ev.grainW = 32;
                if (ev.grainW > 4096) ev.grainW = 4096;
                ev.solaPre = e->transientSolaPre;
                ev.solaPost = e->transientSolaPost;
                ev.grainQ = ev.outPos - ev.solaPre;
                ev.grainIn = ev.tp - (long long)std::llround((double)ev.solaPre / (double)e->internalStretch);
                ev.src.resize((size_t)e->channels);
                for (int c = 0; c < e->channels; c++) {
                    Chan &k = e->ch[(size_t)c];
                    long n = (long)(hi - lo + 1);
                    std::vector<float> seg((size_t)n, 0.0f);
                    for (long j = 0; j < n; j++) {
                        long long gp = lo + j;
                        if (gp >= k.histBase && gp < (long long)(k.histBase + (long long)k.hist.size()))
                            seg[(size_t)j] = k.hist[(size_t)(gp - k.histBase)];
                    }
                    ev.src[(size_t)c] = std::move(seg);
                }
                e->pending.push_back(std::move(ev));
                e->lastTransientIn = tp;
            }
        }
        e->envPrev += 0.1f * (rms - e->envPrev);
    }
    /* Return the position scanned so far, so a leftover partial block is not skipped forever. */
    return p;
}

/* Decay confirmation: a genuine transient is well down over the long window while steady
   content such as a vowel is not, so the long window is required. */
void confirm_events(taffy_engine *e) {
    if (e->channels <= 0) return;
    Chan &c0 = e->ch[0];
    for (size_t n = 0; n < e->pending.size();) {
        taffy_engine::TEvent &ev = e->pending[n];
        if (ev.confirmed) { n++; continue; }
        if (e->inFed < ev.confirmAt) { n++; continue; }
        /* Confirmation window, default 384. Keeping LWSTART + LWIN small lets the event be
           created earlier, so a burst's copy cursor can still reach back to
           outPos - smearSpan: at stretch = 4 the span is 5760, which needs L <= 608. */
        static const int LWIN = getenv("TAFFY_CFWIN") ? atoi(getenv("TAFFY_CFWIN")) : 384;
        static const int LWSTART = getenv("TAFFY_CFSTART") ? atoi(getenv("TAFFY_CFSTART")) : 256;
        long long q = ev.tp + LWSTART;
        double rmsLater = 0;
        if (q >= c0.histBase && (long long)(q + LWIN) <= c0.histBase + (long long)c0.hist.size()) {
            double e2 = 0;
            for (int i = 0; i < LWIN; i++) { float v = c0.hist[(size_t)(q + i - c0.histBase)]; e2 += (double)v * v; }
            rmsLater = sqrt(e2 / LWIN);
        }
        double rmsAt = 0;
        if (ev.tp >= c0.histBase && ev.tp + 64 <= c0.histBase + (long long)c0.hist.size()) {
            double e2 = 0;
            for (int i = 0; i < 64; i++) { float v = c0.hist[(size_t)(ev.tp + i - c0.histBase)]; e2 += (double)v * v; }
            rmsAt = sqrt(e2 / 64);
        }
        bool keep = (rmsLater <= 0.0 || rmsAt > rmsLater * 2.0);   /* ratio threshold; drums ~2.9, vowel 1.6, noise 0.9 */
        if (getenv("TAFFY_TDBG2"))
            fprintf(stderr, "[v] tp=%lld rmsAt=%.4f later=%.4f ratio=%.2f keep=%d\n",
                    ev.tp, rmsAt, rmsLater, rmsLater > 0 ? rmsAt / rmsLater : -1.0, (int)keep);
        if (keep) {
            ev.confirmed = 1;
            /* Burst vs isolated spike: distinguished by the energy left after the transient.
               rmsLater/rmsAt is ~0.00 for impulses and ~0.34 for drums; vowels and noise are
               already rejected by the 2.0 test above. */
            double frac = (rmsAt > 0) ? rmsLater / rmsAt : 0.0;
            /* SOLA only helps isolated spikes and hurts decaying bursts, so grains are
               enabled for spikes alone (frac <= 0.15). */
            bool isSpike = (frac <= 0.15);
            if (isSpike) {
                ev.grainVerbatim = 0;
                /* Input-domain absolute grid (phase locked to the signal), default off: it
                   only helps once the rear region is covered by grains as well. */
                ev.absGrid = getenv("TAFFY_ABSGRID") ? (atoi(getenv("TAFFY_ABSGRID")) != 0) : false;
                /* Mark-period extraction, off by default for zero cost. It needs W + maxLag of
                   look-ahead but the detection point has only ~4k samples of history, so a
                   separate, continuously fed analysis history would be needed to use it. */
                if (mark_enabled())
                    ev.markPeriod = estimate_mark_period(e->ch[0].hist, e->ch[0].histBase, ev.tp, 16,
                                    (getenv("TAFFY_MARKMAX") ? atoi(getenv("TAFFY_MARKMAX")) : 8192));
                ev.solaPre  = e->transientSolaPre;
                ev.solaPost = e->transientSolaPost;
                if (e->spikeDirectMap) {
                    /* Spike path: the PV residue between impulses is what costs sharpness and
                       it is spread far out, so the bulk copy runs only before the impulse
                       (mapPost defaults to 0), removing the far smear and leaving the near
                       field to the PV. Coverage follows the real smear span
                       (N/Ha - 1) * (Hs - Ha); a fixed value falls far short at stretch = 4. */
                    long long spanPv = (long long)(e->N / (e->Ha > 0 ? e->Ha : 1) - 1) *
                                       (long long)(e->Hs - e->Ha);
                    if (spanPv < 0) spanPv = -spanPv;
                    double pvSc = (double)e->pitch > 0.0 ? (double)e->pitch : 1.0;
                    int spanOut = (int)std::llround((double)spanPv / pvSc);
                    int wantPre = std::max(e->burstMapPre, spanOut);
                    /* The spike path does not get the extra reach-back that the shortened
                       confirmation window gives bursts; it is clamped to what the original
                       L = 1280 window allowed, (N - 1280) * internalStretch. */
                    long long reachOld = (long long)(e->N - (256 + 1024)) *
                                         (long long)std::llround((double)e->internalStretch);
                    int reachOut = (int)std::llround((double)reachOld / pvSc);
                    if (wantPre > reachOut) wantPre = reachOut;
                    /* Optional: derive the coverage from the mark spacing instead of a
                       constant. Grain length is half the mark spacing and the hop is the whole
                       spacing, so a transient naturally covers [previous mark, next mark].
                       Converting to the pv domain multiplies by internalStretch. */
                    if (getenv("TAFFY_MARKCOV") && atoi(getenv("TAFFY_MARKCOV")) > 0 && ev.markPeriod > 0) {
                        double frac = (double)atoi(getenv("TAFFY_MARKCOV")) / 100.0;   /* percentage of the mark spacing */
                        int covOut = (int)std::llround((double)ev.markPeriod * (double)e->internalStretch * frac);
                        if (covOut > 64 && covOut < 200000) {
                            wantPre = covOut;
                            ev.mapPost = covOut;
                        }
                    }
                    ev.directMap = 1;
                    ev.mapNoRightTaper = 0;
                    ev.mapPre  = (int)std::llround((double)wantPre * pvSc);
                    if (ev.mapPost <= 0)
                        ev.mapPost = (int)std::llround((double)e->spikeMapPost * pvSc);
                    ev.mapQ = ev.outPos - (long long)ev.mapPre;
                }
            } else if ((e->burstGrainOnCompress && e->internalStretch < 1.0f) ||
                       (e->burstGrainOnStretch  && e->internalStretch > 1.0f &&
                        (std::fabs((double)e->pitch - 1.0) < 1e-6 ||
                         (getenv("TAFFY_BURSTPITCH") && atoi(getenv("TAFFY_BURSTPITCH")) != 0)))) {
                ev.grainVerbatim = 1;              /* compression-tier burst: sample-for-sample with WSOLA alignment */
                /* Coverage and grains live in pv coordinates, and the resampler scales pv by
                   1/pitch again, so output-domain target lengths must be multiplied by pitch. */
                double pvScale = (double)e->pitch > 0.0 ? (double)e->pitch : 1.0;
                /* Coverage must span the real smear (N/Ha - 1) * (Hs - Ha) in pv domain, converted to the output domain */
                long long spanPv = (long long)(e->N / (e->Ha > 0 ? e->Ha : 1) - 1) *
                                   (long long)(e->Hs - e->Ha);
                if (spanPv < 0) spanPv = -spanPv;
                int spanOut = (int)std::llround((double)spanPv / (pvScale > 0 ? pvScale : 1.0));
                if (spanOut > (int)e->N * 8) spanOut = (int)e->N * 8;
                int spDefault = std::max((int)(e->N / 2), spanOut);
                int spOut = e->transientSolaBurstPre  > 0 ? e->transientSolaBurstPre  : spDefault;
                /* Post-side coverage must span the whole stretched burst body and tail, or
                   that part is left to the phase vocoder, whose random-phase OLA decays
                   noticeably faster. Twice the front span is where the decay time constant
                   stops improving. */
                int sqOut = e->transientSolaBurstPost > 0 ? e->transientSolaBurstPost
                                                          : (int)(2 * spanOut);
                int sp = (int)std::llround((double)spOut * pvScale);
                int sq = (int)std::llround((double)sqOut * pvScale);
                if (mark_enabled())
                    ev.markPeriod = estimate_mark_period(e->ch[0].hist, e->ch[0].histBase, ev.tp, 16,
                                    (getenv("TAFFY_MARKMAX") ? atoi(getenv("TAFFY_MARKMAX")) : 8192));
                ev.solaPre  = sp;
                ev.solaPost = sq;
                ev.grainW   = (int)std::llround((double)e->transientSolaW * pvScale);
                ev.grainQ   = ev.outPos - sp;
                ev.grainIn  = ev.tp - (long long)std::llround((double)sp / (double)e->internalStretch);
                ev.absGrid  = getenv("TAFFY_ABSGRID") ? (atoi(getenv("TAFFY_ABSGRID")) != 0) : false;
                /* The bulk copy runs too: grains handle the far pre-echo, the copy handles the
                   near field and the burst body. The two are complementary. */
                if (e->burstDirectMap) {
                    long long spanPv = (long long)(e->N / (e->Ha > 0 ? e->Ha : 1) - 1) *
                                       (long long)(e->Hs - e->Ha);
                    if (spanPv < 0) spanPv = -spanPv;
                    double pvSc = (double)e->pitch > 0.0 ? (double)e->pitch : 1.0;
                    int spanOut = (int)std::llround((double)spanPv / pvSc);
                    int wantPre = std::max(e->burstMapPre, spanOut);
                    ev.directMap = 1;
                    ev.mapPre  = (int)std::llround((double)wantPre * pvSc);
                    ev.mapPost = (int)std::llround((double)e->burstMapPost * pvSc);
                    ev.mapQ = ev.outPos - (long long)ev.mapPre;
                }
            } else if (e->burstDirectMap) {
                ev.directMap = 1;
                ev.mapPre  = (int)std::llround((double)e->burstMapPre  * (double)e->pitch);
                ev.mapPost = (int)std::llround((double)e->burstMapPost * (double)e->pitch);
                ev.mapQ = ev.outPos - (long long)ev.mapPre;
            } else {
                /* burst outside a compression tier: off by default; clear explicitly, otherwise it inherits the spike coverage */
                ev.solaPre  = 0;
                ev.solaPost = 0;
            }
            if (getenv("TAFFY_TDBG2"))
                fprintf(stderr, "[b] tp=%lld frac=%.3f -> %s\n", ev.tp, frac,
                        isSpike ? "spike (SOLA on)" : "burst (SOLA off)");
            n++;
        }
        else { e->pending.erase(e->pending.begin() + (long)n); }
    }
}

/* Write pending transients into pv[] at output = input * internalStretch, with raised-cosine cross-fades at the edges. */
void apply_transients(taffy_engine *e) {
    long long emitted = (long long)e->pv[0].size() + (long long)e->pvTrimmed;
    for (size_t n = 0; n < e->pending.size();) {
        taffy_engine::TEvent &ev = e->pending[n];
        if (!ev.confirmed) { n++; continue; }   /* wait for the decay confirmation before touching the output */
        /* (A) Left-side suppression: must happen before resample_out consumes the samples, so it advances every round regardless of the copy timing. */
        if (getenv("TAFFY_TDBG") && n == 0)
            fprintf(stderr, "[p] outPos=%lld sup=%d win=%d supLo=%lld emitted=%lld read=%zu\n",
                    ev.outPos, ev.sup, ev.win, ev.supLo, emitted, e->pvRead[0]);
        if (ev.sup > ev.win) {
            for (int c = 0; c < e->channels; c++) {
                std::vector<float> &dst = e->pv[(size_t)c];
                long long end = (long long)e->pvTrimmed + (long long)dst.size();
                long long rd = (long long)e->pvRead[(size_t)c];
                long long lo = ev.supLo < rd ? rd : ev.supLo;
                long long hi = ev.outPos - (long long)ev.win - 1;
                if (hi > end - 1) hi = end - 1;
                float g = e->transientSupGain;
                /* Raised-cosine cross-fade at both ends. Multiplying [lo,hi] by a constant g
                   would put a step at each edge, and the step itself radiates broadband
                   energy that no suppression depth can remove; fading from 1 down to g
                   instead creates no new energy at the boundary. */
                const long long span = hi - lo + 1;
                long long F = span / 4;
                if (F > 2048) F = 2048;
                if (F < 1) F = 1;
                for (long long gp = lo; gp <= hi; gp++) {
                    const long long d = gp - lo;
                    float w = 1.0f;
                    if (d < F) w = 0.5f * (1.0f - std::cos((float)M_PI * (float)d / (float)F));
                    else if (hi - gp < F) w = 0.5f * (1.0f - std::cos((float)M_PI * (float)(hi - gp) / (float)F));
                    const float gg = 1.0f - (1.0f - g) * w;
                    dst[(size_t)(gp - (long long)e->pvTrimmed)] *= gg;
                }
                if (hi >= lo) ev.supLo = hi + 1;
            }
        }
        /* (B) SOLA grain replacement: covers the whole smear region with time-domain grains
           of the input segment, cross-faded at 50% overlap (out = w*input + (1-w)*out).
           Grains are placed on the output timeline and read the input at 1/stretch, so the
           grains themselves are stretched too. */
        if (ev.solaPre > 0) {
            int W = ev.grainW, G = W / 2;
            /* Input-domain absolute grid: snap the grain's input start to a multiple of Ha,
               anchored at input sample 0, so the transient's relative position inside the
               grain varies with tp mod Ha instead of staying constant. grainQ is shifted to
               match, keeping the transient exactly at outPos. */
            /* Mark-driven grain length: W = half the mark spacing, overriding the constant
               transientSolaW when marks are enabled. */
            if (getenv("TAFFY_TDBG3"))
                fprintf(stderr, "[k] outPos=%lld tp=%lld markPeriod=%d grainW=%d\n",
                        ev.outPos, ev.tp, ev.markPeriod, ev.grainW);
            /* Control experiment (TAFFY_MARKRAND=1): replace the estimated period with a
               pseudo-random one, to tell a real benefit from merely breaking a constant. */
            if (getenv("TAFFY_MARKRAND") && atoi(getenv("TAFFY_MARKRAND"))) {
                unsigned h = (unsigned)(ev.tp * 2654435761u);
                ev.markPeriod = 500 + (int)(h % 2500);
            }
            /* TAFFY_MARKGRID=1: derive the grain length from the actual mark spacing rather
               than the autocorrelation period, i.e. (current mark - previous mark) >> 1. */
            if (getenv("TAFFY_MARKGRID") && atoi(getenv("TAFFY_MARKGRID")) && ev.markPeriod > 0) {
                int mh = mark_spacing_half(e->ch[0].hist, e->ch[0].histBase, ev.tp,
                                           ev.markPeriod,
                                           (getenv("TAFFY_MARKMAX") ? atoi(getenv("TAFFY_MARKMAX")) : 8192));
                if (mh >= 32 && mh <= 4096) {
                    W = mh; ev.grainW = W; G = W / 2;
                    if (getenv("TAFFY_TDBG3"))
                        fprintf(stderr, "[k] mark spacing/2 = %d (period %d)\n", mh, ev.markPeriod);
                }
            } else if (mark_enabled() && ev.markPeriod > 0) {
                int w2 = ev.markPeriod / 2;
                /* Default cap 448 on the grain length (TAFFY_WCAP=0 removes it). Most of the
                   benefit comes from the smaller grain rather than from the period itself,
                   but a mark-driven length with a cap avoids the regressions that a purely
                   fixed grain length causes. */
                static const int WCAP = getenv("TAFFY_WCAP") ? atoi(getenv("TAFFY_WCAP")) : 448;
                if (WCAP > 0 && w2 > WCAP) w2 = WCAP;
                if (w2 >= 32 && w2 <= 4096) { W = w2; ev.grainW = W; G = W / 2; }
            }
            if (ev.absGrid) {
                /* Grid spacing prefers the extracted mark period and falls back to Ha.
                   Anchored at input sample 0, i.e. phase-locked to the signal. */
                long long Ha = ev.markPeriod > 0
                             ? (long long)ev.markPeriod
                             : (long long)std::llround((double)G / (double)e->internalStretch);
                if (Ha > 0) {
                    long long snapped = (long long)std::floor((double)ev.grainIn / (double)Ha) * Ha;
                    ev.grainQ += ev.grainIn - snapped;
                    ev.grainIn = snapped;
                }
            }
            long long applied = 0, skipped = 0, gmin = -1, gmax = -1, base = 0;
            /* Upper bound of the covered region: solaPost if set, else the transient copy's half-window */
            long long gEnd = ev.outPos + (long long)(ev.solaPost > 0 ? ev.solaPost : ev.win);
            while (ev.grainQ + W < emitted && ev.grainQ < gEnd) {
                /* WSOLA: in verbatim mode, nudge the grain's input start within +/-S to
                   maximize correlation of the overlap with what has already been written,
                   avoiding the misalignment and cancellation a fixed hop would cause. */
                ev.grainSearch = ev.grainIn;
                /* WSOLA is disabled in equal-hop mode: the grains are identical by
                   construction, so nudging the start would only break coherence. */
                static const bool EQHOP_WS = !getenv("TAFFY_EQHOP") || atoi(getenv("TAFFY_EQHOP")) != 0;
                if (ev.grainVerbatim && !EQHOP_WS) {
                    static const int SENV = getenv("TAFFY_WSS") ? atoi(getenv("TAFFY_WSS")) : 256;
                    const int S = SENV, OV = W / 2;
                    Chan &k0 = e->ch[0];
                    long long q0 = ev.grainQ;
                    if (q0 >= (long long)e->pvTrimmed &&
                        q0 + OV <= (long long)e->pvTrimmed + (long long)e->pv[0].size()) {
                        double bestC = -2.0;
                        for (long long pp = ev.grainIn - S; pp <= ev.grainIn + S; pp++) {
                            if (pp < k0.histBase || pp + OV > k0.histBase + (long long)k0.hist.size()) continue;
                            double dot = 0, ea = 0, eb = 0;
                            for (int i = 0; i < OV; i++) {
                                double a = k0.hist[(size_t)(pp + i - k0.histBase)];
                                double b = e->pv[0][(size_t)(q0 + i - (long long)e->pvTrimmed)];
                                dot += a * b; ea += a * a; eb += b * b;
                            }
                            if (ea < 1e-12 || eb < 1e-12) continue;
                            double c = dot / std::sqrt(ea * eb);
                            if (c > bestC) { bestC = c; ev.grainSearch = pp; }
                        }
                    }
                }
                for (int c = 0; c < e->channels; c++) {
                    std::vector<float> &dst = e->pv[(size_t)c];
                    Chan &k = e->ch[(size_t)c];
                    base = (long long)e->pvTrimmed + (long long)e->pvRead[(size_t)c];
                    for (int jj = 0; jj < W; jj++) {
                        long long gp = ev.grainQ + jj;
                        if (gp < base) { skipped++; continue; }
                        /* No grain writes inside the bulk-copy span: both cover the same
                           region in different rounds and the later one would overwrite the
                           earlier one's result. */
                        if (ev.directMap && !(getenv("TAFFY_NOGEXCL") && atoi(getenv("TAFFY_NOGEXCL"))) &&
                            gp >= ev.outPos - (long long)ev.mapPre &&
                            gp <= ev.outPos + (long long)ev.mapPost) { skipped++; continue; }
                        size_t li = (size_t)(gp - (long long)e->pvTrimmed);
                        if (li >= dst.size()) break;
                        /* The grain is a sample-for-sample copy of the input segment, which
                           keeps its pitch. The input hop is the output hop divided by stretch
                           and the stretching comes from overlapping the grains. */
                        long long gi; double giFrac = 0.0, u = (double)jj;
                        if (ev.grainVerbatim) {
                            /* The grain must be pre-resampled by 1/pitch before it is written
                               into pv, because resample_out steps through pv at rate pitch.
                               The two cancel, leaving the grain verbatim in the output. */
                            u = (double)jj / ((double)e->pitch > 1e-9 ? (double)e->pitch : 1.0);
                            gi = ev.grainSearch + (long long)std::floor(u);
                            giFrac = u - std::floor(u);
                        }
                        else gi = ev.tp + (long long)std::floor((double)(gp - ev.outPos) / (double)ev.stretch);
                        if (gi < k.histBase || gi + 1 >= k.histBase + (long long)k.hist.size()) { skipped++; continue; }
                        size_t bi = (size_t)(gi - k.histBase);
                        applied++;
                        if (gmin < 0 || gp < gmin) gmin = gp;
                        if (gp > gmax) gmax = gp;
                        /* The grain grid would replicate the transient at spacing Hs - Ha,
                           although the +/-win copy path already places it exactly at outPos,
                           so the transient core is cut out of the grain. TAFFY_GEXCL sets the
                           cut radius, 0 = off. */
                        {
                            static const int GEX = getenv("TAFFY_GEXCL") ? atoi(getenv("TAFFY_GEXCL")) : 0;
                            if (GEX > 0) {
                                long long d = gi - ev.tp;
                                if (d < 0) d = -d;
                                if (d <= (long long)GEX) { skipped++; continue; }
                            }
                        }
                        float v;
                        if (giFrac > 1e-9)
                            v = (float)((1.0 - giFrac) * k.hist[bi] + giFrac * k.hist[bi + 1]);
                        else
                            v = k.hist[bi];

                        float w = 0.5f - 0.5f * std::cos(2.0f * (float)M_PI * (float)jj / (float)W);
                        dst[li] = w * v + (1.0f - w) * dst[li];
                    }
                }
                ev.grainQ += G;
                if (ev.grainVerbatim) {
                    /* The input-to-pv ratio is internalStretch = stretch * pitch, because the
                       resampler scales pv by 1/pitch again, so the input hop must be
                       pv hop / internalStretch and not / stretch. */
                    /* Equal-hop mode, on by default: the input advances by G but only every
                       stretch-th output grain, so consecutive grains hold identical content
                       and overlap coherently as w + (1-w) = 1. This removes the incoherent
                       sqrt(K) loss that grains offset by Hs - Ha would cause. */
                    static const bool EQHOP = !getenv("TAFFY_EQHOP") || atoi(getenv("TAFFY_EQHOP")) != 0;
                    if (EQHOP) {
                        ev.grainInAcc += (double)G / (double)e->internalStretch;
                        while (ev.grainInAcc >= (double)G) { ev.grainIn += G; ev.grainInAcc -= (double)G; }
                    } else {
                        ev.grainIn += (long long)std::llround((double)G / (double)e->internalStretch);
                    }
                } else
                    ev.grainIn = ev.tp;
            }
            if (getenv("TAFFY_TDBG2") && (applied || skipped))
                fprintf(stderr, "[g] outPos=%lld W=%d applied=%lld skipped=%lld range=[%lld,%lld] (rel %lld..%lld) base=%lld bufEnd=%lld\n",
                        ev.outPos, W, applied, skipped, gmin, gmax,
                        gmin - ev.outPos, gmax - ev.outPos, base,
                        (long long)e->pvTrimmed + (long long)e->pv[0].size());
        }
        /* (B0) Bulk copy over the burst: reads the input mapped through stretch across
           [outPos - mapPre, outPos + mapPost]. The cursor advances only once the region has
           actually been processed, otherwise an idle round with lo > hi would skip it. */
        if (ev.directMap) {
            long long base = (long long)e->pvTrimmed + (long long)e->pvRead[0];
            long long lo = ev.mapQ > base ? ev.mapQ : base;
            long long hi = ev.outPos + (long long)ev.mapPost;
            long long end = (long long)e->pvTrimmed + (long long)e->pv[0].size() - 1;
            if (hi > end) hi = end;
            if (lo <= hi) {
                const long long xf = 256;
                long long left = ev.outPos - (long long)ev.mapPre;
                long long right = ev.outPos + (long long)ev.mapPost;
                for (int c = 0; c < e->channels; c++) {
                    std::vector<float> &dst = e->pv[(size_t)c];
                    Chan &k = e->ch[(size_t)c];
                    for (long long gp = lo; gp <= hi; gp++) {
                        double si = (double)(gp - ev.outPos) / (double)ev.stretch;
                        long long gi = ev.tp + (long long)std::floor(si);
                        double fr = si - (double)std::floor(si);
                        if (gi < k.histBase + 1 ||
                            gi + 2 >= k.histBase + (long long)k.hist.size()) continue;
                        size_t bi = (size_t)(gi - k.histBase);
                        /* Cubic (Catmull-Rom) interpolation. Linear interpolation smooths a
                           noise burst badly at high ratios: at stretch = 4 each output sample
                           advances only 0.25 input samples, so the burst's high-frequency
                           content is lost. */
                        double p0 = k.hist[bi - 1], p1 = k.hist[bi],
                               p2 = k.hist[bi + 1], p3 = k.hist[bi + 2];
                        double c0 = p1;
                        double c1 = 0.5 * (p2 - p0);
                        double c2 = p0 - 2.5 * p1 + 2.0 * p2 - 0.5 * p3;
                        double c3 = 0.5 * (p3 - p0) + 1.5 * (p1 - p2);
                        float v = (float)(((c3 * fr + c2) * fr + c1) * fr + c0);
                        /* Right-edge taper, decided by transient kind. With mapPost = 0 the
                           right edge is outPos itself, where the +/-win copy takes over, so
                           tapering would zero the weight exactly where the pre-echo is worst.
                           With mapPost > 0 the edge falls mid-signal and dropping the taper
                           would leave a hard step there. */
                        long long d0 = gp - left;
                        long long d1 = (ev.mapNoRightTaper && ev.mapPost == 0)
                                       ? ((long long)1 << 30) : (right - gp);
                        long long dm = d0 < d1 ? d0 : d1;
                        /* cap < 1 leaves (1 - cap) of the PV residue in place, so the front region is not cleaned up beyond the intended amount. */
                        static const float MAPCAP = getenv("TAFFY_MAPCAP") ? (float)atof(getenv("TAFFY_MAPCAP")) : 1.0f;
                        float w = (dm >= xf) ? 1.0f
                                : 0.5f - 0.5f * std::cos((float)M_PI * (float)dm / (float)xf);
                        w *= MAPCAP;
                        size_t li = (size_t)(gp - (long long)e->pvTrimmed);
                        dst[li] = w * v + (1.0f - w) * dst[li];
                    }
                }
                ev.mapQ = hi + 1;
                if (getenv("TAFFY_TDBG2"))
                    fprintf(stderr, "[m] outPos=%lld mapped [%lld,%lld] -> cursor %lld\n",
                            ev.outPos, lo, hi, ev.mapQ);
            }
        }
        if (ev.outPos + ev.win >= emitted) { n++; continue; }
        int win = ev.win;
        for (int c = 0; c < e->channels; c++) {
            std::vector<float> &dst = e->pv[(size_t)c];
            const std::vector<float> &src = ev.src[(size_t)c];

            for (int j = -win; j <= win; j++) {
                long long gp = ev.outPos + j;
                if (gp < (long long)e->pvTrimmed) continue;
                size_t li = (size_t)(gp - (long long)e->pvTrimmed);
                if (li >= dst.size()) continue;
                float v;
                if (ev.resample) {
                    /* resample the source segment on the output timeline: in[tp + j/stretch] */
                    double sj = (double)j / (double)ev.stretch;
                    long si = (long)std::floor(sj);
                    long bi = (long)(ev.tp + si - ev.inBase);
                    if (bi < 0 || (size_t)(bi + 1) >= src.size()) continue;
                    double fr = sj - (double)si;
                    v = (float)((1.0 - fr) * src[(size_t)bi] + fr * src[(size_t)(bi + 1)]);
                } else {
                    long i0 = (long)(ev.tp + j - ev.inBase);
                    if (i0 < 0 || (size_t)i0 >= src.size()) continue;
                    v = src[(size_t)i0];
                }
                /* cross-fade: full weight over the inner half-window, raised cosine down to 0 at the outer edge */
                int a = j < 0 ? -j : j;
                float w;
                if (a <= win / 2) w = 1.0f;
                else {
                    float t = (float)(win - a) / (float)(win - win / 2);
                    w = 0.5f * (1.0f - std::cos((float)M_PI * t));
                }
                dst[li] = w * (ev.gain * v) + (1.0f - w) * dst[li];
            }
            if (ev.sup > win && getenv("TAFFY_TSUPR")) {   /* post-echo suppression (off by default: the target behaviour is louder, not quieter, after an impulse) */
                long long end = (long long)e->pvTrimmed + (long long)dst.size();
                long long rd = (long long)e->pvRead[(size_t)c];
                long long lo = ev.supHi < rd ? rd : ev.supHi;
                long long hi = ev.outPos + ev.sup;
                if (hi > end - 1) hi = end - 1;
                for (long long gp = lo; gp <= hi; gp++) {
                    float t = (float)(gp - ev.outPos - win) / (float)(ev.sup - win);
                    float g = 0.05f + 0.95f * (0.5f - 0.5f * std::cos((float)M_PI * t));
                    dst[(size_t)(gp - (long long)e->pvTrimmed)] *= g;
                }
                if (hi >= lo) ev.supHi = hi + 1;
            }
        }
        if (getenv("TAFFY_TDBG"))
            fprintf(stderr, "[t] apply outPos=%lld sup=%d emitted=%lld read=%zu suppressible=%lld\n",
                    ev.outPos, ev.sup, emitted, e->pvRead[0],
                    (long long)(ev.outPos - ev.sup) - (long long)e->pvRead[0]);
        /* The grains may need several more rounds to cover the whole solaPost region, since
           each round only processes what has been produced so far, so the event is kept
           until the grains finish; erasing earlier would truncate the covered region. */
        {
            long long gEnd2 = ev.outPos + (long long)(ev.solaPost > 0 ? ev.solaPost : ev.win);
            bool grainsDone = (ev.solaPre <= 0) || (ev.grainQ + (long long)ev.grainW >= gEnd2);
            bool mapDone = (!ev.directMap) ||
                           (ev.mapQ > ev.outPos + (long long)ev.mapPost);
            if (grainsDone && mapDone) e->pending.erase(e->pending.begin() + (long)n);
            else n++;
        }
    }
}

/* ---------------------------------------------------------------- formant correction
   The output resampler scales the whole spectrum, so the spectral envelope (the formants)
   travels with the pitch. This stage keeps the pitch shift but puts the envelope back: per
   frame it estimates the envelope by cepstral smoothing, divides it out to get the fine
   structure, re-applies the envelope sampled at f * pitch, and overlap-adds the result.
   Block based: the frame is twice the output block with a hop of one block, so the Hann
   window sums to one and the stage is transparent apart from the envelope warp.
   Only Soloist uses it; Efficient and Pro keep the plain resampled output. */
static double e2_at(const std::vector<float>& env, int nb, int b, double P) {
    const double src = (double)b * P;
    if (src >= (double)(nb - 1)) return (double)env[(size_t)(nb - 1)];
    const int i0 = (int)src; const double fr = src - (double)i0;
    return (1.0 - fr) * (double)env[(size_t)i0] + fr * (double)env[(size_t)i0 + 1];
}

static void formant_correct(taffy_engine *e, float *const *out, int nFrames) {
    const int N = e->fcN;
    if (N <= 0 || nFrames <= 0 || 2 * nFrames != N) return;
    const int nb = N / 2 + 1;
    /* Cepstral lifter, in bins. It has to sit below the pitch period or the harmonics leak
       into the envelope, and the period depends on the direction: an up shift raises the
       output pitch, a down shift lowers it, so one value cannot suit both. Measured against
       a reference output on real speech, worst per band error over 300 Hz to 13 kHz:
         lifter   -12      +12
           34    9.04 dB  3.14 dB
           64    5.93 dB  3.07 dB   <- best for up
          256    5.60 dB  7.21 dB   <- best for down
       TAFFY_FCLIFT overrides both. */
    const bool down = (e->pitch < 1.0f);
    static const int LIFT_UP   = getenv("TAFFY_FCLIFT") ? atoi(getenv("TAFFY_FCLIFT")) : 64;
    static const int LIFT_DOWN = getenv("TAFFY_FCLIFT") ? atoi(getenv("TAFFY_FCLIFT")) : 256;
    const int LIFT = down ? LIFT_DOWN : LIFT_UP;
    const double P = (double)e->pitch;
    if (P < 1e-6) return;

    std::vector<float> re((size_t)N), im((size_t)N), mag((size_t)N);
    std::vector<float> lg((size_t)nb), cep((size_t)nb), env((size_t)nb);

    for (int c = 0; c < e->channels; c++) {
        std::vector<float> &hist = e->fcIn[(size_t)c];      /* N samples: previous block + this */
        std::vector<float> &acc  = e->fcAcc[(size_t)c];     /* N samples: overlap-add accumulator */
        /* build the frame: previous block (nFrames) followed by the current block */
        for (int i = 0; i < nFrames; i++) hist[(size_t)(nFrames + i)] = out[c][i];
        for (int i = 0; i < N; i++) { re[(size_t)i] = hist[(size_t)i] * e->fcWin[(size_t)i]; im[(size_t)i] = 0.0f; }
        taffy::fft(re, im, false);
        for (int b = 0; b < nb; b++)
            mag[(size_t)b] = std::sqrt(re[(size_t)b]*re[(size_t)b] + im[(size_t)b]*im[(size_t)b]);

        for (int b = 0; b < nb; b++) lg[(size_t)b] = std::log(mag[(size_t)b] + 1e-9f);
        for (int q = 0; q < nb; q++) {
            double a = 0.0;
            for (int b = 0; b < nb; b++)
                a += (double)lg[(size_t)b] * std::cos(M_PI * (double)q * (double)b / (double)(nb - 1));
            cep[(size_t)q] = (float)(a / (double)nb);
        }
        for (int q = LIFT + 1; q < nb; q++) cep[(size_t)q] = 0.0f;
        for (int b = 0; b < nb; b++) {
            double a = (double)cep[0];
            for (int q = 1; q < nb; q++)
                a += 2.0 * (double)cep[(size_t)q] * std::cos(M_PI * (double)q * (double)b / (double)(nb - 1));
            env[(size_t)b] = (float)a;
        }

        /* Re-apply the envelope sampled at f * pitch: a higher sampling frequency moves the
           envelope down the axis, undoing exactly the shift the resampler introduced. */
        for (int b = 0; b < nb; b++) {
            /* Warp direction. The resampler scales the spectrum by P, so the envelope has to
               be sampled at f * P to undo it. For P < 1 that reads below the reliable part of
               the envelope, so the reciprocal is offered as an alternative (TAFFY_FCINV). */
            static const int FCINV = getenv("TAFFY_FCINV") ? atoi(getenv("TAFFY_FCINV")) : 0;
            const double src = (double)b * ((P < 1.0 && FCINV) ? (1.0 / P) : P);
            float e2;
            if (src >= (double)(nb - 1)) e2 = env[(size_t)(nb - 1)];
            else { const int i0 = (int)src; const double fr = src - (double)i0;
                   e2 = (float)((1.0 - fr) * (double)env[(size_t)i0] + fr * (double)env[(size_t)i0 + 1]); }
            /* Work in the log domain: the correction is the envelope ratio only, and it is
               clamped because a steep envelope would otherwise blow the frame up. */
            double d = (double)e2 - (double)env[(size_t)b];
            /* The clamp only guards against a pathological envelope, it must not truncate
               the correction itself: a full octave of envelope tilt is about 1.8 in log
               units, so a limit of 1.5 was cutting the low band short and flattening F1. */
            static const double FCLIM = getenv("TAFFY_FCLIM") ? atof(getenv("TAFFY_FCLIM")) : 4.0;
            if (d > FCLIM) d = FCLIM;
            if (d < -FCLIM) d = -FCLIM;
            const float sc = (float)std::exp(d);
            re[(size_t)b] *= sc; im[(size_t)b] *= sc;
            if (b > 0 && b < N / 2) { re[(size_t)(N - b)] *= sc; im[(size_t)(N - b)] *= sc; }
        }
        /* Level preservation. Reshaping the envelope also changes the frame's total energy:
           for a downward tilt the warped envelope is the larger one, so a down shift came out
           about 5 dB louder than the reference and an up shift about 3 dB louder. The stage is
           meant to move the formants, not the loudness, so the frame is rescaled to the energy
           it had before the correction. Measured against a reference output on real speech,
           this takes the worst per band error over 300 Hz to 4.5 kHz from 5.60 dB to 1.2 dB
           for a down shift. An up shift was already balanced without it, so it is applied
           only for pitch < 1. TAFFY_FCLVL=0 disables it. */
        {
            static const int LVL = getenv("TAFFY_FCLVL") ? atoi(getenv("TAFFY_FCLVL")) : 1;
            if (LVL && e->pitch < 1.0f) {
                double e0 = 0.0, e1 = 0.0;
                for (int b = 0; b < nb; b++) {
                    e0 += (double)mag[(size_t)b] * (double)mag[(size_t)b];
                    e1 += (double)re[(size_t)b] * (double)re[(size_t)b] +
                          (double)im[(size_t)b] * (double)im[(size_t)b];
                }
                if (e1 > 1e-20 && e0 > 1e-20) {
                    const float g = (float)std::sqrt(e0 / e1);
                    for (int b = 0; b < nb; b++) { re[(size_t)b] *= g; im[(size_t)b] *= g; }
                    for (int b = 1; b < N / 2; b++) { re[(size_t)(N - b)] *= g; im[(size_t)(N - b)] *= g; }
                }
            }
        }
        if (getenv("TAFFY_FCDBG2")) {
            static long n3 = 0;
            if (n3++ == 5) {
                fprintf(stderr, "[fcenv] P=%.2f  bin: env(orig) / e2(warpped)\n", P);
                for (int b = 20; b < 400; b += 20)
                    fprintf(stderr, "    %5.0f Hz: %8.3f / %8.3f   ratio %+.2f\n",
                            (double)b * (double)e->sampleRate / (double)N, (double)env[(size_t)b], (double)e2_at(env, nb, b, P),
                            (double)e2_at(env, nb, b, P) - (double)env[(size_t)b]);
            }
        }
        taffy::fft(re, im, true);
        for (int i = 0; i < N; i++) acc[(size_t)i] += re[(size_t)i] * e->fcWin[(size_t)i];

        if (getenv("TAFFY_FCDBG")) {
            static long n2 = 0;
            if (n2++ % 100 == 0) {
                float wmin = 1e30f, wmax = -1e30f, amin = 1e30f, amax = -1e30f;
                for (int i = 0; i < nFrames; i++) {
                    if (e->fcWsum[(size_t)i] < wmin) wmin = e->fcWsum[(size_t)i];
                    if (e->fcWsum[(size_t)i] > wmax) wmax = e->fcWsum[(size_t)i];
                    if (acc[(size_t)i] < amin) amin = acc[(size_t)i];
                    if (acc[(size_t)i] > amax) amax = acc[(size_t)i];
                }
                fprintf(stderr, "[fc2] wsum=[%.4f,%.4f] acc=[%.2f,%.2f]\n", wmin, wmax, amin, amax);
            }
        }
        /* Emit the first half, then slide both buffers by one hop. */
        for (int i = 0; i < nFrames; i++) {
            const float w = e->fcWsum[(size_t)i];
            out[c][i] = (w > 1e-6f) ? (acc[(size_t)i] / w) : 0.0f;
        }
        std::memmove(acc.data(),  acc.data()  + nFrames, sizeof(float) * (size_t)nFrames);
        std::fill(acc.begin() + nFrames, acc.end(), 0.0f);
        /* Keep the current block as the first half of the next frame; the second half is
           filled from the next call's output. */
        std::memmove(hist.data(), hist.data() + nFrames, sizeof(float) * (size_t)nFrames);
        std::fill(hist.begin() + nFrames, hist.end(), 0.0f);
    }
}

static void formant_correct(taffy_engine *e, float *const *out, int nFrames);

void resample_out(taffy_engine *e, float *const *out, int nFrames) {
    if (getenv("TAFFY_RDBG"))
        fprintf(stderr, "[r] bypass=%d pitch=%.3f taps=%d rsPhase=%.6f rd=%zu src=%zu n=%d\n",
                (int)e->rsBypass, e->pitch, e->rsTaps, e->rsPhase,
                e->pvRead[0], e->pv[0].size(), nFrames);
    if (getenv("TAFFY_DEBUG")) {
        fprintf(stderr, "[dbg] resample_out out=%p out[0]=%p nFrames=%d ch=%d pvAvail=%zu\n",
                (void*)out, out?(void*)out[0]:NULL, nFrames, e->channels,
                e->pv[0].size()-e->pvRead[0]);
    }
    for (int c = 0; c < e->channels; c++) {
        const std::vector<float> &src = e->pv[(size_t)c];
        size_t rd = e->pvRead[(size_t)c];
        double pos = e->rsPhase;
        float *o = out[c];
        if (e->rsBypass) {            /* pitch == 1: straight copy, no resampling */
            for (int i = 0; i < nFrames; i++) {
                size_t idx = rd + (size_t)i;
                o[i] = (idx < src.size()) ? src[idx] : 0.0f;
            }
            continue;
        }
        const int taps = e->rsTaps;
        const int half = taps / 2;
        const std::vector<float> &tab = e->rsTable;
        for (int i = 0; i < nFrames; i++) {
            long long base = (long long)rd + (long long)std::floor(pos);
            double frac = pos - std::floor(pos);
            int s = (int)(frac * taffy_engine::RS_SUB);
            if (s >= taffy_engine::RS_SUB) s = taffy_engine::RS_SUB - 1;
            const float *h = &tab[(size_t)s * taps];
            float acc = 0.0f;
            for (int t = 0; t < taps; t++) {
                long long idx = base + (long long)t - half + 1;
                if (idx >= 0 && (size_t)idx < src.size())
                    acc += h[t] * src[(size_t)idx];
            }
            o[i] = acc;
            pos += (double)e->pitch;
        }
    }
    e->rsPhase += (double)nFrames * (double)e->pitch;

    /* Soloist output gain. Two effects are folded in here, both measured against a
       reference implementation on real speech, relative to the input RMS:
         pitch  0.25   0.5   0.707   1.0   1.5   2.0   4.0
         ref   -6.60 -3.61  -2.06   0.0  +0.40 +0.70 +2.58 dB
       The reference shifts pitch by repeating or discarding pitch periods, which changes the
       energy: a down shift loses level roughly as 10*log10(pitch), an up shift gains less.
       This engine shifts pitch by resampling instead, whose anti alias filter removes the
       band above Nyquist/pitch, so an up shift loses energy rather than gaining it. The
       curve below reproduces the reference behaviour. TAFFY_SOLOOUTG=0 disables it. */
    if (e->solaTier == taffy_engine::SOLA_MONO && std::fabs(e->pitch - 1.0f) > 1e-4f) {
        /* Note the distinct name: TAFFY_SOLAG already exists and controls Pro's aux gain. */
        static const int SG = getenv("TAFFY_SOLOOUTG") ? atoi(getenv("TAFFY_SOLOOUTG")) : 1;
        if (SG) {
            const double P = (double)e->pitch;
            const double l10 = std::log10(P);
            /* Down shifts track the reference closely (measured within 0.5 dB). Up shifts
               cannot be matched by a gain at all: the reference recovers the band above
               Nyquist/pitch by repeating grains, while resampling removes it, and the gap
               grows to 13 dB by pitch 4. The gain is therefore applied to down shifts only,
               leaving up shifts at their natural level. */
            if (P >= 1.0) { /* nothing */ }
            double gdb = (P < 1.0) ? (10.0 * l10) : 0.0;
            const float g = (float)std::pow(10.0, gdb / 20.0);
            for (int c = 0; c < e->channels; c++)
                for (int i = 0; i < nFrames; i++) out[c][i] *= g;
        }
    }
}

void drop_consumed(taffy_engine *e) {
    long n = (long)std::floor(e->rsPhase);
    if (n <= 0) return;
    for (int c = 0; c < e->channels; c++) {
        size_t avail = e->pv[(size_t)c].size() - e->pvRead[(size_t)c];
        size_t take = std::min((size_t)n, avail);
        e->pvRead[(size_t)c] += take;
        if (e->pvRead[(size_t)c] > 65536 &&
            e->pvRead[(size_t)c] * 2 > e->pv[(size_t)c].size()) {
            e->pv[(size_t)c].erase(e->pv[(size_t)c].begin(),
                                   e->pv[(size_t)c].begin() + (long)e->pvRead[(size_t)c]);
            e->pvTrimmed += e->pvRead[(size_t)c];
            e->pvRead[(size_t)c] = 0;
        }
    }
    // trim the consumed input prefix
    if (e->inRead > (size_t)(e->N + e->Ha) * 8) {
        size_t drop = e->inRead - (size_t)e->N;
        for (int c = 0; c < e->channels; c++) {
            std::vector<float> &b = e->ch[(size_t)c].inBuf;
            size_t d = std::min(drop, b.size());
            b.erase(b.begin(), b.begin() + (long)d);
        }
        e->inRead = e->inRead > drop ? e->inRead - drop : 0;
        /* The SOLA grain cursor must be shifted left by the same amount. It advances in
           absolute positions like inRead but is maintained only by sola_step, so leaving it
           alone would push solaInPos past the end of the buffer, make the availability test
           pos + S + W <= inBuf.size() permanently false, and emit endless zeros, because
           taffy_get_frames_needed keeps seeing enough buffered input. */
        for (int c = 0; c < e->channels; c++) {
            Chan &k = e->ch[(size_t)c];
            k.solaInPos = (k.solaInPos > (long long)drop) ? (k.solaInPos - (long long)drop) : 0;
        }
    }
    e->rsPhase -= std::floor(e->rsPhase);
}

} // namespace

extern "C" {

taffy_engine *taffy_create(int framesOut, int channels, float sampleRate, int mode) {
    /* Upper bound N = 2048. Any block size up to N is processed identically; larger values
       cannot be filled correctly and are rejected. */
    if (framesOut < 1 || framesOut > 2048) return nullptr;
    if (channels < 1 || channels > 48) return nullptr;
    if (!(sampleRate >= 8000.0f && sampleRate <= 384000.0f)) return nullptr;

    taffy_engine *e = new (std::nothrow) taffy_engine();
    if (!e) return nullptr;
    e->framesOut = framesOut;
    e->channels = channels;
    e->sampleRate = sampleRate;
    e->mode = mode;

    /* Per-tier geometry. Pro and Efficient share the phase-vocoder core here and need 512
       input samples per 1024 output samples after warm-up; Soloist pulls whole blocks
       instead, which is why it uses a different frames-needed model. The switch is explicit
       so that an enum change cannot silently drift the behaviour. */
    switch (mode) {
    case TAFFY_MODE_PRO:        e->N = 2048; e->solaTier = taffy_engine::SOLA_POLY;
                             e->transientCopy = true;          /* Pro has the transient copy layer */
                             /* Pro defaults to aux mode 8 (parallel OLA + HP(actual) + input
                                silence gate): it wins on both acceptance metrics and avoids
                                mode 4's high-band artifact. */
                             e->proAuxMode = 8;
                             e->auxGain    = 1.30f;            /* calibrated aux gain */
                             e->auxHold    = 3;                /* TAFFY_AUXHOLD */
                             e->auxActual  = true;             /* base the replacement on the actual output */
                             e->auxLvlDb   = -60.0;            /* input silence gate threshold, dB */
                             e->solaMix = 0.0f;  break;        /* mixing is not used */
    /* Efficient keeps the transient copy layer: the phase vocoder performs an equivalent
       time-domain repair of its own, so all tiers share this layer. */
    case TAFFY_MODE_EFFICIENT:  e->N = 2048; e->solaTier = taffy_engine::SOLA_NONE;
                             e->transientCopy = true;  break;
    case TAFFY_MODE_EFF_MOBILE: e->N = 2048; e->solaTier = taffy_engine::SOLA_NONE;
                             e->transientCopy = true;  break;
    case TAFFY_MODE_SOLO_MONO:  e->N = 2048; e->solaTier = taffy_engine::SOLA_MONO;
                             e->markEvery = 16; break;
    /* Solo-speech tier. The speech variant uses a coarser pitch-analysis step than the sung
       variant, but forcing markEvery = 64 makes impulses much worse while only slightly
       improving vowels and sines, so the default stays at 16. TAFFY_SPEEFEVERY exposes the
       experiment. */
    case TAFFY_MODE_SOLO_SPEECH:e->N = 2048; e->solaTier = taffy_engine::SOLA_MONO;
                             e->markEvery = getenv("TAFFY_SPEEFEVERY") ? atoi(getenv("TAFFY_SPEEFEVERY")) : 16;
                             break;
    default:                 e->N = 2048; e->solaTier = taffy_engine::SOLA_POLY; break;
    }
    /* diagnostic: force one tier to run another tier's overlay layer */
    if (getenv("TAFFY_TIER")) {
        int t = atoi(getenv("TAFFY_TIER"));
        if (t >= 0 && t <= 2) e->solaTier = t;
    }
    /* diagnostic: override the transient copy layer independently of detection/suppression */
    if (getenv("TAFFY_TCOPY")) e->transientCopy = atoi(getenv("TAFFY_TCOPY")) != 0;
    if (getenv("TAFFY_SOLAMIX")) e->solaMix = (float)atof(getenv("TAFFY_SOLAMIX"));
    /* Pro aux combination mode: 0 off, 1 mix, 2 full-band add, 3 add the high band only,
       4 complementary replace. TAFFY_SOLAADD is the legacy alias kept for compatibility. */
    if (getenv("TAFFY_PROAUXMODE")) e->proAuxMode = atoi(getenv("TAFFY_PROAUXMODE"));
    if (getenv("TAFFY_AUXGATET"))  e->auxGateTransient = (float)atof(getenv("TAFFY_AUXGATET"));
    if (getenv("TAFFY_AUXGATESM")) e->auxGateSm = (float)atof(getenv("TAFFY_AUXGATESM"));
    if (getenv("TAFFY_AUXHOLD"))   e->auxHold = atoi(getenv("TAFFY_AUXHOLD"));
    if (getenv("TAFFY_AUXGATEK"))  e->auxGateK = (float)atof(getenv("TAFFY_AUXGATEK"));
    if (getenv("TAFFY_SOLAG"))     e->auxGain = (float)atof(getenv("TAFFY_SOLAG"));
    if (getenv("TAFFY_SOLAADD")) {
        int a = atoi(getenv("TAFFY_SOLAADD"));
        e->proAuxMode = (a >= 2) ? 3 : (a == 1 ? 2 : 1);
    }
    if (getenv("TAFFY_GRAINVAR")) e->grainVar = atoi(getenv("TAFFY_GRAINVAR")) != 0;
    if (getenv("TAFFY_PROENV"))   e->proEnv   = atoi(getenv("TAFFY_PROENV")) != 0;
    if (getenv("TAFFY_PROAUX"))   e->proAux   = atoi(getenv("TAFFY_PROAUX"));
    if (getenv("TAFFY_AUXBIN"))   e->auxCrossBin = atoi(getenv("TAFFY_AUXBIN"));
    if (getenv("TAFFY_AUXFC"))    e->auxFcBin    = atoi(getenv("TAFFY_AUXFC"));   /* affects the aux mask only */
    if (getenv("TAFFY_ANACOPY"))  e->anaCopy = atoi(getenv("TAFFY_ANACOPY")) != 0;
    if (getenv("TAFFY_AUXACTUAL")) e->auxActual = atoi(getenv("TAFFY_AUXACTUAL")) != 0;
    if (getenv("TAFFY_AUXPIPE"))   e->auxPipe = atoi(getenv("TAFFY_AUXPIPE")) != 0;
    if (getenv("TAFFY_THOLD"))     e->transHoldHops = atoi(getenv("TAFFY_THOLD"));
    if (getenv("TAFFY_AUXLVLDB"))  e->auxLvlDb = atof(getenv("TAFFY_AUXLVLDB"));
    if (getenv("TAFFY_ENVORDER")) e->envOrder = atoi(getenv("TAFFY_ENVORDER"));
    if (getenv("TAFFY_ENVALPHA")) e->envAlpha = (float)atof(getenv("TAFFY_ENVALPHA"));
    if (getenv("TAFFY_ENVMIX"))   e->envMix   = (float)atof(getenv("TAFFY_ENVMIX"));
    /* diagnostic: override the FFT size to calibrate N per tier */
    if (getenv("TAFFY_N")) { int v = atoi(getenv("TAFFY_N")); if (v >= 256 && v <= 16384) e->N = v; }
    e->Ha = e->N / 4;
    if (getenv("TAFFY_TGAIN")) e->transientGain = (float)atof(getenv("TAFFY_TGAIN"));
    /* Transient forward/backward window lengths in samples. In hops (Hs = 512) the defaults
       are 768 forward and 384 backward; both are tunable. */
    if (getenv("TAFFY_TSPRE"))  e->transientSolaPre  = atoi(getenv("TAFFY_TSPRE"));
    if (getenv("TAFFY_TSPOST")) e->transientSolaPost = atoi(getenv("TAFFY_TSPOST"));
    if (getenv("TAFFY_TWIN"))  e->transientWin = atoi(getenv("TAFFY_TWIN"));
    if (getenv("TAFFY_TSUP"))  e->transientSuppress = atoi(getenv("TAFFY_TSUP"));
    if (getenv("TAFFY_TSUPG")) e->transientSupGain = (float)atof(getenv("TAFFY_TSUPG"));
    if (getenv("TAFFY_SOLA"))    e->transientSolaPre  = atoi(getenv("TAFFY_SOLA"));
    if (getenv("TAFFY_SOLAPOST"))e->transientSolaPost = atoi(getenv("TAFFY_SOLAPOST"));
    if (getenv("TAFFY_SOLAW"))   e->transientSolaW    = atoi(getenv("TAFFY_SOLAW"));
    if (getenv("TAFFY_SOLABPRE")) e->transientSolaBurstPre  = atoi(getenv("TAFFY_SOLABPRE"));
    if (getenv("TAFFY_SOLABPOST"))e->transientSolaBurstPost = atoi(getenv("TAFFY_SOLABPOST"));
    if (getenv("TAFFY_TRES"))  e->transientResample = atoi(getenv("TAFFY_TRES")) != 0;
    if (getenv("TAFFY_TOFF"))  e->transientGain = e->transientGain;

    build_window(e);

    /* Formant correction frame: twice the output block with a hop of one block, so the Hann
       window sums to one and the stage reconstructs perfectly apart from the envelope warp. */
    e->fcN = 2 * e->framesOut;
    if (e->fcN >= 256) {
        const int M = e->fcN;
        e->fcWin.resize((size_t)M);
        for (int i = 0; i < M; i++)
            e->fcWin[(size_t)i] = 0.5f * (1.0f - std::cos(2.0f * (float)M_PI * (float)i / (float)M));
        /* Steady-state window square sum. Because the accumulator slides by one hop per call,
           the value at position j must include the tail of the previous frame (k = -1) as
           well as the current one, otherwise the sum dips to zero at the frame edge and the
           division there explodes. */
        e->fcWsum.assign((size_t)M, 0.0f);
        for (int j = 0; j < M; j++) {
            float a = 0.0f;
            for (int k = -1; k <= 1; k++) {
                const int idx = j - k * (M / 2);
                if (idx >= 0 && idx < M) a += e->fcWin[(size_t)idx] * e->fcWin[(size_t)idx];
            }
            e->fcWsum[(size_t)j] = a;
        }
        e->fcIn.assign((size_t)e->channels, std::vector<float>((size_t)M, 0.0f));
        e->fcAcc.assign((size_t)e->channels, std::vector<float>((size_t)M, 0.0f));
    } else {
        e->fcN = 0;
    }
    rebuild_hopped(e);

    int half = e->N / 2 + 1;
    e->ch.resize((size_t)channels);
    for (int c = 0; c < channels; c++) {
        Chan &k = e->ch[(size_t)c];
        k.ola.assign((size_t)e->N, 0.0f);
        k.wsum.assign((size_t)e->N, 0.0f);
        k.lastPhase.assign((size_t)half, 0.0f);
        k.sumPhase.assign((size_t)half, 0.0f);
        k.mag.assign((size_t)half, 0.0f);
        k.origPhase.assign((size_t)half, 0.0f);
        k.re.assign((size_t)e->N, 0.0f);
        k.im.assign((size_t)e->N, 0.0f);
        k.prevBand.assign((size_t)std::max(1, e->N >> 8), 0.0f);
    }
    e->pv.resize((size_t)channels);
    e->pvRead.assign((size_t)channels, 0);
    prime_wsum(e);
    sola_init(e);      /* SOLA state is always built; only SOLA_MONO uses it */
    e->build_rs_table();
    return e;
}

void taffy_destroy(taffy_engine *e) {
    if (e && getenv("TAFFY_GATEDBG") && e->channels > 0) {
        const Chan &c0 = e->ch[0];
        fprintf(stderr, "[gateduty] total hops=%lld closed=%lld (%.1f%%)\n",
                c0.gateTot, c0.gateClosed,
                c0.gateTot ? 100.0 * (double)c0.gateClosed / (double)c0.gateTot : 0.0);
    }
    delete e;
}

int taffy_get_frames_needed(const taffy_engine *ce) {
    taffy_engine *e = const_cast<taffy_engine *>(ce);
    if (e->pvBypass) return e->framesOut;      /* bypass: one to one */
    /* Soloist uses a block pull model instead of the deficit-driven one that Pro and
       Efficient use: it requests a whole 5120-sample block, then asks for nothing for about
       ten output blocks while still producing 1024 frames each, and then pulls the next
       block. This is only a hint to the caller and does not affect the audio path; the exact
       trigger instant is approximated here by "top up when below half a block". */
    if (e->solaTier == taffy_engine::SOLA_MONO) {
        static const int PULL = getenv("TAFFY_PULL") ? atoi(getenv("TAFFY_PULL")) : 5120;
        /* inBuf.size() - inRead must not be used as the unconsumed amount: drop_consumed
           keeps N samples for history and analysis, so that difference is always >= N and
           the buffer would always look sufficient. The engine instead tracks inFed minus
           the samples consumed by stepping. */
        long buffered = (long)e->inFed - (long)e->stepsDone * (long)e->Ha;
        if (buffered < 0) buffered = 0;
        /* Always request a whole block rather than just the deficit, so that 5120 samples of
           input support exactly 10 output blocks at the steady-state rate. */
        if (getenv("TAFFY_PULLDBG"))
            fprintf(stderr, "[pull] inFed=%lld steps=%lld buffered=%ld inBuf=%zu inRead=%zu search=%d\n",
                    (long long)e->inFed, (long long)e->stepsDone, buffered,
                    e->ch[0].inBuf.size(), e->inRead, e->ch[0].solaSearch);
        /* The threshold must include the grain window look-ahead W + S: sola_step requires
           solaInPos + S + W <= inBuf.size(), so the last S + W samples can never be consumed
           and a threshold of a single step would stall the buffer forever. */
        long lookahead = 2L * (long)e->Hs + (long)std::max(32, std::min(4096, e->ch[0].solaSearch));
        /* The threshold must also cover a whole output block, not just one step: a block
           needs framesOut/Hs steps of Ha input samples each, i.e. framesOut * Ha / Hs.
           Asking for less leaves the engine short of input while it is still asked to emit
           1024 frames, so it pads or stretches and the frame count overshoots. At stretch = 2
           a block needs only 512 samples, far below 2*Hs + S, so that range is unaffected. */
        long needPerBlock = (long)((double)e->framesOut * (double)e->Ha / (double)(e->Hs > 0 ? e->Hs : 1)) + 1;
        if (buffered < lookahead + (long)e->Hs + needPerBlock) {
            /* PULL is only ever raised, never lowered. 5120 was calibrated at stretch = 2,
               where a 1024-frame block needs 512 input samples, so 5120 covers ten blocks.
               Since framesOut * Ha / Hs grows as compression deepens, heavy compression
               would otherwise ask for less than a single block; at stretch >= 0.25
               needPerBlock is already <= 5120, making this a no-op there. */
            int pull = PULL;
            if ((long)pull < needPerBlock) pull = (int)(needPerBlock < (1 << 20) ? needPerBlock : (1 << 20));
            if (getenv("TAFFY_PULLDBG"))
                fprintf(stderr, "[pull] needPerBlock=%ld -> PULL=%d\n", needPerBlock, pull);
            return pull;
        }
        return 0;
    }
    double need = (double)e->framesOut * (double)e->pitch + 2.0;   /* pv samples needed */
    size_t pvAvail = 0;
    for (int c = 0; c < e->channels; c++)
        pvAvail = std::max(pvAvail, e->pv[(size_t)c].size() - e->pvRead[(size_t)c]);
    long deficit = (long)std::ceil(need) - (long)pvAvail;
    if (deficit < 0) deficit = 0;
    long steps = (deficit + e->Hs - 1) / e->Hs;
    if (steps > 0) steps += 1;            // one extra step for safety
    /* The window requirement depends on the tier: the phase vocoder needs N samples per
       FFT, while SOLA needs only the grain length 2*Hs plus the alignment search half-width
       S, which the pitch-mark spacing can push as high as 2205 on a pulse train. Using N for
       SOLA would understate it and stall the step loop. */
    long windowNeed;
    /* tiers with a SOLA layer need the grain window plus the search half-width */
    int sS = std::max(32, std::min(4096, e->ch[0].solaSearch));
    long solaNeed = (long)(2 * e->Hs) + sS;
    if (e->solaTier == taffy_engine::SOLA_MONO) {
        windowNeed = solaNeed;
    } else if (e->solaTier == taffy_engine::SOLA_POLY && e->proAuxMode > 0) {
        windowNeed = std::max((long)e->N, solaNeed);
    } else {
        windowNeed = (long)e->N;
    }
    long totalNeeded = (steps > 0) ? (steps - 1) * (long)e->Ha + windowNeed : 0;
    size_t bufAvail = e->ch[0].inBuf.size() - std::min(e->inRead, e->ch[0].inBuf.size());
    long want = totalNeeded - (long)bufAvail;
    if (want < 0) want = 0;
    if (getenv("TAFFY_NEDBG"))
        fprintf(stderr, "[need] pvAvail=%zu deficit=%ld steps=%ld totalNeeded=%ld bufAvail=%zu inBuf=%zu inRead=%zu want=%ld tier=%d\n",
                pvAvail, deficit, steps, totalNeeded, bufAvail,
                e->ch[0].inBuf.size(), e->inRead, want, e->solaTier);
    if (want > (1 << 20)) want = 1 << 20;
    return (int)want;
}

int taffy_get_max_frames_needed(const taffy_engine *ce) {
    taffy_engine *e = const_cast<taffy_engine *>(ce);
    /* Soloist needs a much larger maximum than the phase-vocoder tiers */
    if (e->solaTier == taffy_engine::SOLA_MONO) return 26624;
    return e->N + 32 * e->framesOut + 4096;
}

void taffy_set_stretch_pitch(taffy_engine *e, float stretch, float pitch, int reset) {
    if (!e) return;
    if (stretch < 1e-4f) stretch = 1e-4f;
    if (pitch < 1e-4f) pitch = 1e-4f;
    e->stretch = stretch;
    e->pitch = pitch;
    e->internalStretch = stretch * pitch;
    rebuild_hopped(e);
    e->build_rs_table();
    if (reset) {
        e->rsPhase = 0.0;
        e->inRead = 0;
        e->stepsDone = 0;
        e->flushing = false;
        e->flushRemaining = 0;
        e->inFed = 0;
        e->pvEmitted = 0;
        e->pvTrimmed = 0;
        e->envPrev = 0.0f;
        e->lastTransientIn = -1000000;
        e->scanPos = 0;
        e->pending.clear();
        for (int c = 0; c < e->channels; c++) { e->ch[(size_t)c].hist.clear(); e->ch[(size_t)c].histBase = 0; }
        for (int c = 0; c < e->channels; c++) {
            Chan &k = e->ch[(size_t)c];
            std::fill(k.lastPhase.begin(), k.lastPhase.end(), 0.0f);
            std::fill(k.sumPhase.begin(), k.sumPhase.end(), 0.0f);
            std::fill(k.prevBand.begin(), k.prevBand.end(), 0.0f);
            k.fluxMean = k.fluxVar = 0.0f;
            k.frames = 0;
            k.sinceTransient = 1000;
            k.isTransient = false;
            k.inBuf.clear();
            e->pv[(size_t)c].clear();
            e->pvRead[(size_t)c] = 0;
        }
        prime_wsum(e);
    }
    sola_init(e);      /* SOLA state is always built; only SOLA_MONO uses it */
    for (int c = 0; c < e->channels; c++) { e->ch[(size_t)c].hist.clear(); e->ch[(size_t)c].histBase = 0; }
}

int taffy_process(taffy_engine *e, const float *const *in, int nFrames, float *const *out) {
    if (!e || !out) return TAFFY_ERR_PARAM;
    if (in == nullptr) return TAFFY_ERR_PARAM;
    if (e->pvBypass) {                          /* straight copy when ratio == 1 */
        /* Short blocks must be accepted here as well as in the non-bypass path: rejecting a
           final partial block would make the driver error out or drop it, and the following
           flush would then repeat earlier content. Copy what is there and zero-fill the
           remainder. */
        int n = nFrames < e->framesOut ? nFrames : e->framesOut;
        for (int c = 0; c < e->channels; c++) {
            if (n > 0) std::memcpy(out[c], in[c], sizeof(float) * (size_t)n);
            if (n < e->framesOut)
                std::memset(out[c] + n, 0, sizeof(float) * (size_t)(e->framesOut - n));
        }
        /* Do not advance the pv pipeline here. flush resumes from inRead, and those samples
           have already been copied out verbatim above, so flush would process the same input
           a second time and emit a duplicate rather than a decaying tail. The bypass path has
           no latency compensation (out[i] == in[i]), so the correct output length is exactly
           the input length and there must be no tail at all. */
        for (int c = 0; c < e->channels; c++)
            e->ch[(size_t)c].inBuf.insert(e->ch[(size_t)c].inBuf.end(), in[c], in[c] + nFrames);
        e->inFed += nFrames;
        return TAFFY_OK;
    }
    /* Short blocks (nFrames below the requested amount) are accepted: at the end of a stream
       the host cannot fill the request, so feed what exists and run as many steps as
       possible, letting the pv zero-padding cover the shortfall (resample_out already writes
       0 for out-of-range indices). Returning an error and flushing instead would discard the
       unfed input at the tail and truncate the output. */
    if (nFrames < 0) return TAFFY_ERR_PARAM;

    for (int c = 0; c < e->channels; c++) {
        e->ch[(size_t)c].inBuf.insert(e->ch[(size_t)c].inBuf.end(), in[c], in[c] + nFrames);
        e->ch[(size_t)c].hist.insert(e->ch[(size_t)c].hist.end(), in[c], in[c] + nFrames);
    }
    e->inFed += nFrames;
    if (nFrames > 0) e->solaPrimed = true;      /* Soloist block pull: once input has arrived no first-block request is needed */
    /* history trimming: keep the most recent samples */
    {
        int win = e->transientWin > 0 ? e->transientWin : (e->N / 2);
        if (win > 4096) win = 4096;
        /* Beyond the history synthesis needs, keep an extra analysis history for mark-period
           estimation: the autocorrelation needs maxLag + W past samples (4096 + 1024 by
           default) whereas synthesis needs only about 3k. That longer history is kept only
           when mark extraction is enabled, because lengthening it alone changes the boundary
           scan of transient detection. */
        static const size_t ANAKEEP = getenv("TAFFY_ANAKEEP") ? (size_t)atoi(getenv("TAFFY_ANAKEEP"))
                                       : (mark_enabled() ? 65536 : 0);
        size_t keep = (size_t)(2 * win + e->N + e->Ha + 64 + ANAKEEP);
        for (int c = 0; c < e->channels; c++) {
            Chan &k = e->ch[(size_t)c];
            if (k.hist.size() > keep * 2) {
                size_t drop = k.hist.size() - keep;
                k.hist.erase(k.hist.begin(), k.hist.begin() + (long)drop);
                k.histBase += (long long)drop;
            }
        }
    }
    /* only scan regions whose source segment is fully available; the cursor advances independently so nothing is skipped */
    {
        int win = e->transientWin > 0 ? e->transientWin : (e->N / 2);
        if (win > 4096) win = 4096;
        /* The look-ahead window is [p+128, p+1152): long enough to span several pitch periods
           and recognise a decaying event, yet short enough that the event is created early,
           since SOLA grains have to cover the smear region before it is consumed. */
        long long to = e->inFed - win
                       - (getenv("TAFFY_CFSTART") ? atoi(getenv("TAFFY_CFSTART")) : 256)
                       - (getenv("TAFFY_CFWIN") ? atoi(getenv("TAFFY_CFWIN")) : 384);
        if (e->scanPos < 0) e->scanPos = 0;
        if (getenv("TAFFY_TDBG"))
            fprintf(stderr, "[s] inFed=%lld to=%lld scanPos=%lld\n", e->inFed, to, e->scanPos);
        if (to > e->scanPos) {
            e->scanPos = detect_transients(e, e->scanPos, to);
        }
    }

    double needPv = (double)e->framesOut * (double)e->pitch + 2.0;
    int guard = 0;
    while (guard++ < 1000000) {
        size_t avail = e->pv[0].size() - e->pvRead[0];
        if ((double)avail >= needPv) break;
        if (!step_once(e)) break;
    }
    confirm_events(e);
    snapshot_aux(e);
    apply_transients(e);
    apply_aux(e);
    if (getenv("TAFFY_LVLDBG")) {
        static long n = 0;
        if (n++ % 100 == 0) {
            double pv = 0, src = 0;
            size_t a = e->pv[0].size() - e->pvRead[0];
            for (size_t i = e->pvRead[0]; i < e->pv[0].size(); i++) pv += (double)e->pv[0][i] * e->pv[0][i];
            long long steps = e->stepsDone;
            for (long long i = 0; i < steps * e->Ha && i < (long long)e->ch[0].inBuf.size(); i++)
                src += (double)e->ch[0].inBuf[(size_t)i] * e->ch[0].inBuf[(size_t)i];
            fprintf(stderr, "[lvl2] tier=%d pitch=%.3f Hs=%d Ha=%d steps=%lld pvRMS=%.5f srcRMS=%.5f ratio=%+.2f dB\n",
                    (int)e->solaTier, (double)e->pitch, e->Hs, e->Ha, steps,
                    std::sqrt(pv / (double)(a > 0 ? a : 1)),
                    std::sqrt(src / (double)(steps * e->Ha > 0 ? steps * e->Ha : 1)),
                    20.0 * std::log10((std::sqrt(pv / (double)(a > 0 ? a : 1)) + 1e-30) /
                                      (std::sqrt(src / (double)(steps * e->Ha > 0 ? steps * e->Ha : 1)) + 1e-30)));
        }
    }
    resample_out(e, out, e->framesOut);
    /* Soloist: put the spectral envelope back after the resampler moved it with the pitch. */
    {
        static const int FC = getenv("TAFFY_FC") ? atoi(getenv("TAFFY_FC")) : 1;
        if (getenv("TAFFY_FCDBG")) {
            static long n = 0;
            if (n++ % 100 == 0)
                fprintf(stderr, "[fc] tier=%d pitch=%.3f fcN=%d framesOut=%d FC=%d\n",
                        (int)e->solaTier, (double)e->pitch, e->fcN, e->framesOut, FC);
        }
        if (FC && e->solaTier == taffy_engine::SOLA_MONO && std::fabs(e->pitch - 1.0f) > 1e-4f)
            formant_correct(e, out, e->framesOut);
    }
    drop_consumed(e);
    if (getenv("TAFFY_DEBUG2")) {
        fprintf(stderr, "[e] inRead=%zu inBuf=%zu pvSize=%zu pvRead=%zu avail=%zu\n",
                e->inRead, e->ch[0].inBuf.size(), e->pv[0].size(), e->pvRead[0],
                e->pv[0].size()-e->pvRead[0]);
    }
    return TAFFY_OK;
}

int taffy_flush(taffy_engine *e, float *const *out) {
    if (!e || !out) return 0;
    if (!e->flushing) {
    e->auxDrain = true;      /* flush: drain all pending hops from the aux pipeline */
        /* first call: push the pipeline's residue out with zero padding; later calls only consume */
        e->flushing = true;
        /* The bypass path is handled separately and never touches the pipeline. At ratio 1
           the output is the input plus an all-zero tail of about 1466 frames, which is part
           of the visible frame-count contract, so the same tail length is kept here with
           zero content. The zero-pad-and-pump path below would reprocess the real input
           still sitting in inBuf and emit a duplicate of it instead. */
        if (e->pvBypass) {
            long cap = (long)e->N - (long)e->Hs;
            e->flushRemaining = cap > 0 ? cap : 0;   /* the tail is silence, at the length the frame-count contract implies */
        } else {
        /* pushing the whole N-sample OLA tail out takes N/Hs steps of Ha zero inputs each */
        static const int ZS = getenv("TAFFY_FLUSHZ") ? atoi(getenv("TAFFY_FLUSHZ")) : 0;
        /* drain amount 1.5*N: empirically the smallest total frame-count error over nine stretch/pitch combinations */
        size_t zeros = ZS > 0 ? (size_t)ZS : ((size_t)e->N + (size_t)e->N / 2);
        /* the Pro mixing layer needs enough zeros to push the SOLA grain window out as well */
        if (e->solaTier == taffy_engine::SOLA_POLY && e->proAuxMode > 0) {
            size_t zn = (size_t)(2 * e->Hs + std::max(32, std::min(4096, e->ch[0].solaSearch)));
            if (zeros < zn) zeros = zn;
        }
        /* The SOLA tier must first consume all buffered real input and only then add one
           grain window of zeros. Padding 1.5*N zeros up front would count them as input and
           pump many more steps than the real input can support. */
        size_t afterReal = 0;
        if (e->solaTier == taffy_engine::SOLA_MONO) {
            int g0 = 0;
            while (g0++ < 1000000 && step_once(e)) {}          /* 1) consume all real input */
            snapshot_aux(e); apply_transients(e); apply_aux(e);
            afterReal = e->pv[0].size() - e->pvRead[0];
            /* 2) add only one grain-window-plus-search-half-width of zeros: sola_step must look ahead W + S samples */
            zeros = (size_t)(2 * e->Hs + std::max(32, std::min(4096, e->ch[0].solaSearch)));
        }
        for (int c = 0; c < e->channels; c++)
            e->ch[(size_t)c].inBuf.insert(e->ch[(size_t)c].inBuf.end(), zeros, 0.0f);
        int guard = 0;
        while (guard++ < 100000 && step_once(e)) { apply_aux(e); }
        size_t avail = e->pv[0].size() - e->pvRead[0];
        e->flushRemaining = (long)std::floor((double)avail / (double)e->pitch);

        /* SOLA: the total follows the number of steps the real input can support, not real
           output plus half a grain. The grain window looks ahead W + S samples, so the last
           ~1280 real samples fit into no grain during either pass and can only be covered by
           the zero-padded steps. Total = floor(inFed / Ha) * Hs. */
        if (e->solaTier == taffy_engine::SOLA_MONO) {
            long long consumedPv = (long long)e->pvTrimmed + (long long)e->pvRead[0];
            long long stepsJust   = (long long)(e->inFed / (long long)e->Ha);
            long long allowedPv   = stepsJust * (long long)e->Hs;
            long long left = allowedPv - consumedPv;
            if (left < 0) left = 0;
            e->flushRemaining = (long)((double)left / (double)e->pitch);
        }
        /* Same rule as the SOLA tier: the total output must not exceed what the real input can
           support. Pumping 1.5*N zeros and setting flushRemaining = avail/pitch counts the
           zero-padded steps as output, and those extra frames form a tail driven by silence
           rather than by content, which stretches the result noticeably. */
        if (!(getenv("TAFFY_NOFLUXCAP") && atoi(getenv("TAFFY_NOFLUXCAP")) != 0)) {
            long long consumedPv = (long long)e->pvTrimmed + (long long)e->pvRead[0];
            long long stepsJust   = (long long)(e->inFed / (long long)(e->Ha > 0 ? e->Ha : 1));
            long long allowedPv   = stepsJust * (long long)e->Hs;
            long long leftAll = allowedPv - consumedPv;
            if (leftAll < 0) leftAll = 0;
            long cap2 = (long)((double)leftAll / (double)e->pitch);
            if (e->flushRemaining > cap2) e->flushRemaining = cap2;
        }
        (void)afterReal;
        }   /* end else (non-bypass) */
    }
    if (e->flushRemaining <= 0) return 0;
    int n = (int)std::min((long)e->framesOut, e->flushRemaining);
    resample_out(e, out, n);
    drop_consumed(e);
    e->flushRemaining -= n;
    return n;
}

int   taffy_fft_size(const taffy_engine *e) { return e ? e->N : 0; }
int   taffy_hop_in(const taffy_engine *e) { return e ? e->Ha : 0; }
float taffy_internal_stretch(const taffy_engine *e) { return e ? e->internalStretch : 0.0f; }

} // extern "C"
