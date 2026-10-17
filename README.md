# FrameLab — video enhancement, complexity analysis & content-aware encoding

FrameLab is a small media-engineering project: a **C++17 / pybind11 core** (motion estimation, spatial
complexity, LUT kernels) driven from **Python** (OpenCV, ffmpeg/x264, VMAF, Streamlit).

What it does:

1. **Enhance** video (CLAHE, denoise, sharpen, brightness LUT) and measure what actually changed
   (blur/noise/brightness before vs. after, PSNR vs. input, enhancement ms/frame, output bitrate). Audio is kept.
2. **Analyze content complexity** per frame: spatial detail + motion-compensated temporal residual
   (block-matching motion estimation in C++), plus scene-cut detection.
3. **Content-aware encoding study:** sweep x264 CRF values, score each encode (**VMAF** when your ffmpeg has it,
   otherwise SSIM/PSNR), plot the rate–distortion curve, and recommend the cheapest CRF that meets a quality target.
4. **Benchmark** native C++ against NumPy/OpenCV/pure-Python references, with bit-exactness checks.

> Scope: this is a learning/portfolio project, **not** a codec. The motion estimator is classic block matching
> (SAD, full + diamond search) to study the idea; blur/noise values are heuristics, not perceptual quality.

---

## 1. Quick start (works without any C++ compiler)

Python 3.10+ (tested on 3.12). From the `FrameLab` folder:

**Windows (PowerShell)**
```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope Process Bypass
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/run_demo.py            # generates a sample video and runs everything (~1 min)
streamlit run app/dashboard.py        # the web UI
```

**Linux / macOS**
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_demo.py
streamlit run app/dashboard.py
```

`requirements.txt` includes `imageio-ffmpeg`, which ships an ffmpeg binary with x264, so you do **not** need to
install ffmpeg yourself. If you already have ffmpeg on your PATH it is used instead; point `FFMPEG_BINARY` at a
specific build to override. Run `python -m framelab.cli info` to see what was detected.

Without the C++ module everything still works through pure-Python fallbacks (slower motion estimation).

## 2. Build the C++ core (recommended — this is the interesting part)

```bash
python setup.py build_ext --inplace
python -m framelab.cli info            # should print  "native_module": true
```

This produces `framelab/_cpp*.so` (or `.pyd` on Windows) and is picked up automatically.
You need a C++17 compiler: **Windows** → "Build Tools for Visual Studio" (Desktop development with C++),
**Linux** → `sudo apt install build-essential python3-dev`, **macOS** → `xcode-select --install`.

CMake alternative (writes the module into `framelab/` too):
```bash
cmake -S cpp -B cpp/build -Dpybind11_DIR=$(python -m pybind11 --cmakedir)
cmake --build cpp/build --config Release
```

## 3. Command line

```bash
python -m framelab.cli info
python scripts/make_sample_video.py                      # samples/sample.mp4 (synthetic, 2 scenes, with audio)

python -m framelab.cli process  samples/sample.mp4 --method clahe --output outputs/clahe.mp4
python -m framelab.cli complexity samples/sample.mp4
python -m framelab.cli sweep    samples/sample.mp4 --max-frames 150          # CRF sweep + RD curve
python -m framelab.cli compare  reference.mp4 processed.mp4                  # PSNR / SSIM
python -m framelab.cli kernel-benchmark
python -m framelab.cli motion-benchmark --include-python-full
```

Methods: `identity`, `clahe`, `denoise`, `sharpen`, `brightness` (uses the native LUT kernel), `combined`.
`process` flags: `--crf` (output x264 quality), `--backend auto|ffmpeg|opencv`, `--no-analyze` (faster).
Audio is kept when the whole video is processed through ffmpeg (not with `--max-frames` or `--backend opencv`).

`sweep` metric: `auto` picks VMAF if your ffmpeg has libvmaf (the bundled Linux build did when I tested; on other platforms check `python -m framelab.cli info`), else SSIM.
Defaults for the quality target are VMAF 93, SSIM 0.95, PSNR 38 dB; override with `--target`.

## 4. Tests

```bash
python -m pytest
```
153 test cases: native-vs-Python equivalence, motion recovery of a planted shift, LUT bit-exactness, scene-cut detection,
audio preservation, odd-size videos, CRF-sweep monotonicity, VMAF sanity, CLI error paths. Tests that need an
unavailable component (native build, libvmaf, scikit-image) are **skipped with a reason**, never silently passed.

## 5. Measured results

Measured once in a Linux sandbox (x86-64, single thread, synthetic data). **Your numbers will differ — rerun
`python scripts/run_demo.py`.** Treat these as examples of what the benchmarks report, not as claims.

**Brightness/contrast kernel, 1920×1080×3** (all three outputs are bit-identical)

| Implementation | ms / call | vs NumPy |
|---|---|---|
| NumPy float32 | 10.3 | 1× |
| OpenCV `cv2.LUT` | 2.7 | 3.9× |
| C++ 256-entry LUT (unrolled) | 2.4 | 4.3× |

The C++ LUT is on par with OpenCV's already-optimised LUT: for this simple per-pixel op there is little left to win.

**Motion estimation, 640×360, 16×16 blocks, ±16 px** (Python and C++ return *identical* vectors)

| Implementation | ms / frame pair | mean SAD / block |
|---|---|---|
| Python diamond | 109 | 39.2 |
| **C++ diamond** | **6.0** (18× faster) | 39.2 |
| Python full search | 4412 | 37.5 |
| C++ full search | 207 (21× faster) | 37.5 |

Diamond search is ~35× faster than full search here, at a cost of ~4.6 % higher residual SAD (the classic speed/quality trade-off).

**Enhancement cost at 1280×720** (ms per frame, single thread)

| identity | brightness | sharpen | CLAHE | denoise (NLM) | combined |
|---|---|---|---|---|---|
| 0.3 | 1.1 | 7.6 | 17.3 | 1932 | 1772 |

Non-local-means denoising is ~0.5 fps at 720p, i.e. **not real-time**; the dashboard shows a real-time flag per run.

**CRF sweep on the bundled synthetic sample (150 frames, VMAF target 93):** recommended **CRF 26**, ~76 % lower
bitrate than CRF 18. With SSIM (luma-only, more lenient) the same target logic picks CRF 30 — the choice of metric
changes the answer, which is exactly why the tool reports several.

## 6. How it works

- **`cpp/image_ops.cpp`** — `adjust_brightness_contrast` (LUT built in float32 with FMA contraction prevented, so it is bit-exact with
  the NumPy reference), `motion_estimate` (SAD block matching; full search or large/small-diamond search; GIL released),
  `gradient_energy`. Tie-breaking is "strictly lower SAD wins, starting from the zero vector" so Python and C++ agree exactly.
- **`framelab/kernels.py`** — native fast path + `*_python` reference for each kernel.
- **`framelab/complexity.py`** — per frame: spatial = mean |∇| of the downscaled luma; temporal = mean per-pixel SAD after motion
  compensation; scene cut = residual spike above `max(cut_min, cut_ratio × median)`. `complexity_index = spatial + temporal`
  is an explainable proxy in 8-bit intensity units, **not** a calibrated predictor.
- **`framelab/sweep.py`** — encode with x264 at each CRF, measure bitrate and quality against the source, pick the highest
  CRF whose quality ≥ target.
- **`framelab/ffmpeg_tools.py`** — ffmpeg discovery, raw-frame piping to x264 with audio muxing, VMAF via `libvmaf`.
- **`framelab/pipeline.py`** — decode → enhance → analyze → encode, with enhancement timing reported separately from end-to-end FPS.

```
FrameLab/
├── app/dashboard.py          Streamlit UI (Enhance · Complexity · Encoding sweep · Benchmarks)
├── cpp/                      C++17 core (image_ops.cpp) + CMakeLists.txt
├── framelab/                 Python package (kernels, complexity, sweep, pipeline, ffmpeg tools, CLI)
├── scripts/                  make_sample_video.py · run_demo.py
├── tests/                    pytest suite
├── samples/ outputs/         sample video (generated) · results
├── setup.py  requirements.txt
```

## 7. Known limitations (honest list)

- Block matching is integer-pel and single reference; real encoders use sub-pel, multiple references and variable block sizes.
- The complexity index and scene-cut detector are heuristics; the threshold adapts to each video but is not tuned on a dataset.
- CRF sweep quality is measured on a (default) 150-frame excerpt; per-scene or per-title ladders are future work.
- SSIM here is luma-only; PSNR/SSIM decode through OpenCV, VMAF through ffmpeg.
- Tested on Linux only. Windows/macOS paths use standard `subprocess`/`pathlib`, but I could not run them here — please report anything odd.
- The sample video is synthetic; results on real footage will differ (that is the point of running your own clips).

## 8. Ideas to extend it

1. Run `sweep` + `complexity` over 20+ clips and fit/validate a model that predicts the CRF needed for a VMAF target from the complexity features.
2. ROI-based encoding: detect faces/people and feed it to x264 as regions of interest (ffmpeg's `addroi` filter) — measure bitrate at equal perceived quality.
3. Sub-pel and variable block-size motion estimation, or SIMD/OpenMP in the C++ kernels with before/after benchmarks.
4. Multi-resolution (bitrate ladder) sweep with a BD-rate summary.
