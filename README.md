# Taffy

A phase vocoder for time stretching and pitch shifting, with transient handling.
Written from scratch in C++.

## Features

* Phase vocoder with a fixed synthesis hop and a varying analysis hop
* Transient handling: detection, phase reset, time domain relocation, optional SOLA grain overlay
* Polyphase sinc resampling with a Kaiser window for pitch shifting
* Four quality modes: `0` Pro, `1` Efficient, `2` Efficient mobile, `3` Soloist (mono)
* Block based streaming API, usable for both offline and real time processing

## Build

```bash
make          # builds the library and the example CLI into build/
make test     # runs the example once
```

Or with CMake:

```bash
cmake -B build-cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build-cmake
```

Requires a C++20 compiler (clang or gcc). No third party libraries.

## Example CLI

```bash
./taffy_stretch <in.f32> <out.f32> <stretch> <pitch> [mode] [outBlock] [channels]
```

* Input and output are raw 32 bit floats (`.f32`), interleaved for multiple channels
* `stretch` is the duration factor, so `2.0` doubles the length
* `pitch` is the frequency factor, so `2^(n/12)` shifts by `n` semitones
* `mode` is one of `0` `1` `2` `3` as listed above

Examples:

```bash
./taffy_stretch in.f32 out.f32 2.0  1.0        # twice as long
./taffy_stretch in.f32 out.f32 1.0  1.189207   # up 3 semitones
./taffy_stretch in.f32 out.f32 0.5  1.0  1     # half as long, Efficient
```

## API

The public interface is in [`include/taffy.h`](include/taffy.h).

```c
taffy_engine *taffy_create(int framesOut, int channels, float sampleRate, int mode);
void          taffy_destroy(taffy_engine *e);

void          taffy_set_stretch_pitch(taffy_engine *e, float stretch, float pitch, int reset);
int           taffy_get_frames_needed(const taffy_engine *e);
int           taffy_get_max_frames_needed(const taffy_engine *e);

int           taffy_process(taffy_engine *e, const float *const *in, int nFrames, float *const *out);
int           taffy_flush(taffy_engine *e, float *const *out);
```

A typical offline loop feeds `taffy_get_frames_needed()` frames at a time and collects
the produced output until `taffy_flush()` returns zero.

## Implementation notes

* The analysis hop follows the synthesis hop divided by the stretch factor, which keeps
  the overlap constant across stretch ratios.
* Phase is advanced per bin from the measured instantaneous frequency, with a reset at
  detected transients so that attacks stay sharp.
* Transients are additionally relocated in the time domain, so their position does not
  drift with the analysis grid.
* Pitch shifting is done by resampling the output, which is why the internal stretch is
  the product of the requested stretch and pitch.
* The high band is rebuilt through a parallel path in the Pro mode and blended back with
  a content adaptive gate, so that smeared high frequencies are replaced without changing
  the overall level.

## References

The implementation follows published work in this area:

* Flanagan, J. L., and Golden, R. M. (1966). *Phase Vocoder*. Bell System Technical Journal, 45(9), 1493-1509.
* Portnoff, M. R. (1976). *Implementation of the digital phase vocoder using the fast Fourier transform*. IEEE Trans. ASSP, 24(3), 243-248.
* Dolson, M. (1986). *The phase vocoder: A tutorial*. Computer Music Journal, 10(4), 14-27.
* Laroche, J., and Dolson, M. (1999). *Improved phase vocoder time scale modification of audio*. IEEE Trans. Speech and Audio Processing, 7(3), 323-332.
* Driedger, J., and Müller, M. (2016). *A review of time scale modification of music signals*. Applied Sciences, 6(2), 57.
* Charpentier, F., and Stella, M. (1986). *Diphone synthesis using an overlap-add technique for speech waveforms concatenation*. ICASSP.
* Moulines, E., and Charpentier, F. (1990). *Pitch-synchronous waveform processing techniques for text-to-speech synthesis using diphones*. Speech Communication, 9(5-6), 453-467.
* Kaiser, J. F. (1974). *Nonrecursive digital filter design using the I0-sinh window function*. IEEE ISCAS.

## License

MIT. See [`LICENSE`](LICENSE).
