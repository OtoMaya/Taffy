/* taffy_stretch.c - minimal example and command line tool for the Taffy engine.
 *
 * Uses only the public API in include/taffy.h and touches no internal state.
 *
 * Usage:
 *   taffy_stretch <in.f32> <out.f32> <stretch> <pitch> [mode] [outBlock] [channels]
 *
 *   mode     0=Pro 1=Efficient 2=Eff-mobile 3=Soloist(m) 4=Soloist(s)   (default 1)
 *   outBlock output frames per process call, 1..2048                    (default 1024)
 *   channels 1..48                                                      (default 1)
 *
 * Input and output are raw float32 and planar: one contiguous run of samples per
 * channel, channels stored one after another.
 *
 * Build (from the impl/ directory). The example is C, so compile it to a C object
 * file first and link it with the C++ engine, otherwise the C++ compiler treats
 * the .c file as C++ and emits deprecation warnings:
 *   clang -O2 -std=c11 -Iinclude -c examples/taffy_stretch.c -o /tmp/taffy_stretch.o
 *   c++   -O2 -std=c++20 -Iinclude /tmp/taffy_stretch.o src/engine.cpp src/fft.cpp -o taffy_stretch
 * Or simply:  make taffy_stretch
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "taffy.h"

static float *read_f32(const char *path, long *nOut) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(2); }
    if (fseek(f, 0, SEEK_END) != 0) { perror("fseek"); exit(2); }
    long bytes = ftell(f);
    fseek(f, 0, SEEK_SET);
    long n = bytes / (long)sizeof(float);
    float *p = (float *)malloc(sizeof(float) * (size_t)(n ? n : 1));
    if (!p) { fprintf(stderr, "out of memory\n"); exit(2); }
    if (n && fread(p, sizeof(float), (size_t)n, f) != (size_t)n) { perror("fread"); exit(2); }
    fclose(f);
    *nOut = n;
    return p;
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr,
            "usage: %s <in.f32> <out.f32> <stretch> <pitch> [mode] [outBlock] [channels]\n"
            "  mode 0=Pro 1=Efficient 2=Eff-mobile 3=Soloist(mono) 4=Soloist(speech)\n"
            "  note: Pro (0) uses the full parallel high band path, Efficient (1) is single path\n",
            argv[0]);
        return 1;
    }
    const char *inPath  = argv[1];
    const char *outPath = argv[2];
    float stretch = (float)atof(argv[3]);
    float pitch   = (float)atof(argv[4]);
    int   mode    = argc > 5 ? atoi(argv[5]) : TAFFY_MODE_EFFICIENT;
    int   blk     = argc > 6 ? atoi(argv[6]) : 1024;
    int   ch      = argc > 7 ? atoi(argv[7]) : 1;

    long nIn = 0;
    float *src = read_f32(inPath, &nIn);
    if (nIn % ch != 0) {
        fprintf(stderr, "input length %ld is not a multiple of the channel count %d\n", nIn, ch);
        return 2;
    }
    long framesIn = nIn / ch;

    taffy_engine *e = taffy_create(blk, ch, 44100.0f, mode);
    if (!e) { fprintf(stderr, "taffy_create failed (check outBlock <= 2048 and channels <= 48)\n"); return 3; }
    taffy_set_stretch_pitch(e, stretch, pitch, 1);

    /* Output capacity: twice the ideal length plus a few blocks, enough for the tail. */
    long cap = (long)((double)framesIn * (stretch > 1.0f ? stretch : 1.0f) * 2.0) + 8L * blk + 65536;
    float *dst = (float *)calloc((size_t)cap * (size_t)ch, sizeof(float));
    /* taffy_process takes `const float *const *`, so the pointer array itself must be
       const qualified, otherwise the compiler warns about discarding nested qualifiers. */
    const float **ip = (const float **)calloc((size_t)ch, sizeof(float *));
    float **op = (float **)calloc((size_t)ch, sizeof(float *));
    if (!dst || !ip || !op) { fprintf(stderr, "out of memory\n"); return 3; }

    long fed = 0, made = 0;
    int guard = 0;
    while (fed < framesIn && guard++ < 100000000) {
        int need = taffy_get_frames_needed(e);
        if (need < 0) need = 0;
        if (need > framesIn - fed) need = framesIn - fed;   /* short blocks are allowed */
        for (int c = 0; c < ch; c++) {
            ip[c] = src + (size_t)c * (size_t)framesIn + fed;
            op[c] = dst + (size_t)c * (size_t)cap + made;
        }
        if (taffy_process(e, ip, need, op) != TAFFY_OK) { fprintf(stderr, "taffy_process failed\n"); break; }
        fed  += need;
        made += blk;
        if (made + blk > cap) break;
    }
    for (int k = 0; k < 100000 && made + blk <= cap; k++) {
        for (int c = 0; c < ch; c++) op[c] = dst + (size_t)c * (size_t)cap + made;
        int n = taffy_flush(e, op);
        if (n <= 0) break;
        made += n;
    }

    /* Statistics: peak, RMS, non-finite count, and content length at a -60 dBFS gate. */
    double peak = 0.0, sq = 0.0;
    long bad = 0, cnt = 0;
    for (int c = 0; c < ch; c++)
        for (long i = 0; i < made; i++) {
            float v = dst[(size_t)c * (size_t)cap + i];
            if (!isfinite(v)) bad++;
            double a = fabs((double)v);
            if (a > peak) peak = a;
            sq += (double)v * v;
            cnt++;
        }
    double rms = cnt ? sqrt(sq / cnt) : 0.0;
    double thr = peak * 1e-3;
    long last = 0;
    for (int c = 0; c < ch; c++)
        for (long i = 0; i < made; i++)
            if (fabs((double)dst[(size_t)c * (size_t)cap + i]) > thr) last = i;

    FILE *of = fopen(outPath, "wb");
    if (!of) { perror(outPath); return 2; }
    for (int c = 0; c < ch; c++)
        fwrite(dst + (size_t)c * (size_t)cap, sizeof(float), (size_t)made, of);
    fclose(of);

    fprintf(stderr,
            "mode=%d stretch=%.3f pitch=%.4f outBlock=%d ch=%d\n"
            "  input %ld frames -> output %ld frames (content %ld frames, ratio %.4f, ideal %.4f)\n"
            "  peak %.4f  RMS %.4f  non-finite %ld  clipped %s\n",
            mode, stretch, pitch, blk, ch, framesIn, made, last + 1,
            framesIn ? (double)(last + 1) / (double)framesIn : 0.0,
            (double)stretch, peak, rms, bad, peak > 1.0 ? "yes" : "no");

    free(dst); free(ip); free(op); free(src);
    taffy_destroy(e);
    return 0;
}
