#include "fft.h"

#include <cmath>

namespace taffy {

static void make_tables(int n, std::vector<float> &cos_t, std::vector<float> &sin_t) {
    cos_t.resize(n / 2);
    sin_t.resize(n / 2);
    for (int i = 0; i < n / 2; i++) {
        double a = 2.0 * M_PI * i / n;
        cos_t[i] = (float)std::cos(a);
        sin_t[i] = (float)std::sin(a);
    }
}

void fft(std::vector<float> &re, std::vector<float> &im, bool inverse) {
    const int n = (int)re.size();
    if (n < 2) return;

    static int cached_n = 0;
    static std::vector<float> cos_t, sin_t;
    if (cached_n != n) {
        make_tables(n, cos_t, sin_t);
        cached_n = n;
    }

    // Bit reversal permutation.
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            std::swap(re[i], re[j]);
            std::swap(im[i], im[j]);
        }
    }

    // Butterflies.
    for (int len = 2; len <= n; len <<= 1) {
        int half = len >> 1;
        int step = n / len;
        for (int i = 0; i < n; i += len) {
            for (int k = 0; k < half; k++) {
                float wr = cos_t[k * step];
                float wi = inverse ? sin_t[k * step] : -sin_t[k * step];
                int a = i + k, b = i + k + half;
                float xr = re[b] * wr - im[b] * wi;
                float xi = re[b] * wi + im[b] * wr;
                re[b] = re[a] - xr;
                im[b] = im[a] - xi;
                re[a] += xr;
                im[a] += xi;
            }
        }
    }

    if (inverse) {
        float s = 1.0f / (float)n;
        for (int i = 0; i < n; i++) {
            re[i] *= s;
            im[i] *= s;
        }
    }
}

} // namespace taffy
