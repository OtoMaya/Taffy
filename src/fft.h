// fft.h - minimal iterative radix-2 complex FFT.
#ifndef TAFFY_FFT_H
#define TAFFY_FFT_H

#include <vector>

namespace taffy {

// In-place FFT. re and im both hold n elements; n must be a power of two.
void fft(std::vector<float> &re, std::vector<float> &im, bool inverse);

} // namespace taffy

#endif
