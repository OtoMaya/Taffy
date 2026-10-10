// taffy.h - public C API for the Taffy time stretch / pitch shift engine.
//
// -- Usage -------------------------------------------------------------------
//   taffy_engine *e = taffy_create(1024, 2, 44100.0f, TAFFY_MODE_EFFICIENT);
//   taffy_set_stretch_pitch(e, /*stretch*/ 2.0f, /*pitch*/ 1.0f, /*reset*/ 1);
//   for (;;) {
//       int need = taffy_get_frames_needed(e);
//       if (need > 0) {
//           if (no more input) break;              // switch to draining
//           int n = min(need, remaining);          // short blocks are allowed
//           taffy_process(e, in + fed, n, out + produced);
//           fed += n;
//       } else {
//           taffy_process(e, in + fed, 0, out + produced);   // 0 is a valid call
//       }
//       produced += framesOut;
//   }
//   while (taffy_flush(e, out + produced) > 0) produced += result;   // drain
//
// -- Contract ----------------------------------------------------------------
//   * create(framesOut, channels, sampleRate, mode)
//       framesOut   output frames produced by each process call, 1..2048.
//                   Values above the internal FFT size are rejected with NULL.
//       channels    1..48
//       sampleRate  8000..384000
//       mode        see the TAFFY_MODE_* enum below
//   * get_frames_needed()   Suggested number of input frames for the next call.
//                           It is a hint, not a requirement: feeding fewer frames
//                           (including 0) is valid and the engine advances with
//                           whatever is available. A return of 0 means no new
//                           input is needed, but process must still be called to
//                           collect output.
//   * process(in, nFrames, out)
//       Returns an error code, 0 on success. On success out always receives
//       exactly framesOut frames. nFrames may be smaller than framesOut, or 0.
//   * flush(out)            Drains the tail. Returns the number of frames written,
//                           which is less than framesOut, or 0 when fully drained.
//                           The amount is determined by how much real input the
//                           engine received, not by how often flush is called.
//   * Channel pointers are planar, one contiguous buffer per channel, not interleaved.
//   * The input type is `const float *const *`, so the host must pass a const
//     qualified pointer array:
//         const float *ip[2] = { left, right };   taffy_process(e, ip, n, op);
//     Passing a plain `float **` triggers -Wincompatible-pointer-types-discards-qualifiers.
//     The output side is the same, using `float *op[2]` of type `float *const *`.
//   * Threading: a single taffy_engine is not thread safe. Separate instances may
//     be used in parallel.
//
// -- Known limitations -------------------------------------------------------
//   * Mode 0 (Pro) rebuilds the high band through a parallel path that is blended
//     in with a per bin complementary switch, crossing over at 1500 Hz. This makes
//     Pro's output differ from Efficient's, but its absolute accuracy against the
//     intended behaviour is lower than Efficient's. The band switch itself is
//     accurate to within about 0.8 dB on drums, noise and vowel material; very
//     short impulses remain the weakest case.
//   * Mode 3 with pitch == 1 takes a pass through path with no delay compensation,
//     so the tail is silent.
//   * Transient overshoot: the phase vocoder in modes 0 and 1 can overshoot peaks
//     by about 1.20x on transients, independently of the transient handling layer.
//     Input peaks above 0.833 can therefore clip on output. This is inherent to
//     the method, not specific to this implementation. Mode 3 has unity gain and
//     does not overshoot. Leave at least 2 dB of headroom when using modes 0 or 1
//     with stretch >= 1.5.
#ifndef TAFFY_H
#define TAFFY_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Error codes. */
enum {
    TAFFY_OK              = 0,
    TAFFY_ERR_ALLOC       = 1000001,
    TAFFY_ERR_PARAM       = 5000003,
    TAFFY_ERR_SAMPLERATE  = 5000004,
    TAFFY_ERR_CHANNELS    = 5000005
};

/* Processing modes.
 *
 *   0  Pro          phase vocoder plus the parallel high band path
 *   1  Efficient    phase vocoder only
 *   2  Efficient mobile   lighter variant of mode 1
 *   3  Soloist mono       grain synthesis only, no phase vocoder, monophonic material
 *   4  Soloist speech     same as mode 3 with a different analysis step
 */
enum {
    TAFFY_MODE_PRO         = 0,   /* phase vocoder + parallel high band path */
    TAFFY_MODE_EFFICIENT   = 1,   /* phase vocoder only */
    TAFFY_MODE_EFF_MOBILE  = 2,   /* lighter variant of mode 1 */
    TAFFY_MODE_SOLO_MONO   = 3,   /* grain synthesis, monophonic */
    TAFFY_MODE_SOLO_SPEECH = 4    /* grain synthesis, speech tuned */
};

typedef struct taffy_engine taffy_engine;

taffy_engine *taffy_create(int framesOut, int channels, float sampleRate, int mode);
void       taffy_destroy(taffy_engine *e);

int  taffy_get_frames_needed(const taffy_engine *e);
int  taffy_get_max_frames_needed(const taffy_engine *e);

/* stretch: duration factor, above 1 makes it longer.
   pitch:   frequency factor, above 1 makes it higher. */
void taffy_set_stretch_pitch(taffy_engine *e, float stretch, float pitch, int reset);

/* in and out are planar channel pointer arrays. Returns 0 or an error code. */
int  taffy_process(taffy_engine *e, const float *const *in, int nFrames, float *const *out);

/* Drains the tail. Returns the number of frames written, 0 when finished. */
int  taffy_flush(taffy_engine *e, float *const *out);

/* Internal parameters, for diagnostics. */
int   taffy_fft_size(const taffy_engine *e);
int   taffy_hop_in(const taffy_engine *e);
float taffy_internal_stretch(const taffy_engine *e);

#ifdef __cplusplus
}
#endif

#endif /* TAFFY_H */
