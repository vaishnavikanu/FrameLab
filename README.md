
# FrameLab — Video Processing and Quality Analysis

FrameLab is a video-processing project I built using Python and C++. I wanted to explore video enhancement, motion estimation, image complexity analysis, and the trade-off between video quality and compression.

The project combines a Python processing pipeline with a C++17 extension using pybind11. I also built a Streamlit dashboard to run experiments and view results.

## Features

- **Video Enhancement:** Apply CLAHE, denoising, sharpening, brightness adjustment, and combined enhancement methods.
- **Complexity Analysis:** Measure spatial detail, motion-compensated temporal residuals, and possible scene changes.
- **Content-Aware Encoding Study:** Sweep x264 CRF values, compare video quality, generate rate-distortion curves, and recommend a CRF that meets a quality target.
- **C++ Optimization:** Implement image-processing kernels and block-matching motion estimation, then benchmark them against Python and OpenCV implementations.
- **Interactive Dashboard:** Explore video enhancement, complexity analysis, encoding sweeps, and benchmarks through Streamlit.
- **Automated Testing:** Use pytest to test core functionality and compare native implementations against reference implementations.

> **Note:** FrameLab is an experimental video-processing toolkit, not a video codec. Complexity scores and noise/blur measurements are heuristic indicators rather than calibrated measures of perceptual quality.

## Tech Stack

- **Languages:** Python, C++17
- **Libraries:** OpenCV, NumPy, pybind11
- **Video Processing:** FFmpeg, x264, VMAF
- **Dashboard:** Streamlit
- **Testing:** pytest

## Project Structure

```text
FrameLab/
├── app/
│   └── dashboard.py
├── cpp/
│   ├── image_ops.cpp
│   └── CMakeLists.txt
├── framelab/
│   ├── analysis.py
│   ├── benchmarks.py
│   ├── cli.py
│   ├── complexity.py
│   ├── enhancement.py
│   ├── ffmpeg_tools.py
│   ├── kernels.py
│   ├── metrics.py
│   ├── pipeline.py
│   ├── sweep.py
│   └── video_io.py
├── scripts/
│   ├── make_sample_video.py
│   └── run_demo.py
├── tests/
├── requirements.txt
├── setup.py
└── README.md
```

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/vaishnavikanu/FrameLab.git
cd FrameLab
```

### 2. Create a Virtual Environment

**Windows (PowerShell)**

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Build the C++ Extension (Optional)

The project supports Python fallbacks when the native extension is unavailable. Building the extension enables the C++ implementations.

A C++17 compiler is required.

```bash
python setup.py build_ext --inplace --force
python -m framelab.cli info
```

Check the output for `native_module: true` to confirm that the extension loaded successfully.

### 4. Run the Demo

```bash
python scripts/run_demo.py
```

### 5. Launch the Dashboard

```bash
python -m streamlit run app/dashboard.py
```

## Command-Line Usage

Display environment and dependency information:

```bash
python -m framelab.cli info
```

Generate a synthetic sample video:

```bash
python scripts/make_sample_video.py
```

Enhance a video using CLAHE:

```bash
python -m framelab.cli process samples/sample.mp4 --method clahe --output outputs/clahe.mp4
```

Analyze video complexity:

```bash
python -m framelab.cli complexity samples/sample.mp4
```

Run a CRF sweep and generate a rate-distortion curve:

```bash
python -m framelab.cli sweep samples/sample.mp4 --max-frames 150
```

Compare a reference video with a processed video:

```bash
python -m framelab.cli compare reference.mp4 processed.mp4
```

Run the kernel and motion-estimation benchmarks:

```bash
python -m framelab.cli kernel-benchmark
python -m framelab.cli motion-benchmark
```

Available enhancement methods include `identity`, `clahe`, `denoise`, `sharpen`, `brightness`, and `combined`.

Audio preservation depends on the processing backend and options. Check the command output for the behavior of each run.

## Implementation Overview

### 1. Video Enhancement

The pipeline decodes video frames, applies the selected enhancement method, analyzes the output, and encodes the processed video. It reports enhancement timing separately from end-to-end processing speed.

### 2. Native C++ Kernels

I used pybind11 to connect the C++ implementation to Python. The native code includes:

- Brightness and contrast adjustment using a lookup table (LUT).
- Gradient-energy calculation for image-detail analysis.
- Block-matching motion estimation using the sum of absolute differences (SAD).
- Full-search and diamond-search motion estimation.

The native implementations are benchmarked against Python and OpenCV alternatives to compare performance and verify correctness.

### 3. Complexity Analysis

The complexity analysis combines spatial image detail with a motion-compensated temporal residual. A residual-based heuristic is also used to identify possible scene cuts.

The resulting complexity index is an explainable indicator for experimentation, not a trained predictor of video complexity.

### 4. Rate-Distortion Analysis

The CRF sweep encodes a video at different x264 quality settings and measures the resulting bitrate and quality. It plots the rate-distortion curve and selects the highest CRF that satisfies the configured quality target.

The quality metric depends on the available FFmpeg build. VMAF is used when supported; otherwise, the configured fallback metric is used.

## Benchmark Results

The following measurements were obtained in my Windows environment. Results depend on the machine, compiler, input, and configuration.

### Brightness/Contrast Kernel

**Input:** 1920 × 1080 × 3 image, 20 repetitions.

| Implementation | Average Time per Call |
|---|---:|
| NumPy float32 | 41.86 ms |
| OpenCV LUT | 1.18 ms |
| C++ LUT | 1.95 ms |

All three implementations produced bit-identical outputs in this benchmark. OpenCV was the fastest in this particular run.

### Motion Estimation

**Input:** 640 × 360, 16 × 16 blocks, search range of ±16 pixels.

| Implementation | Time per Frame Pair |
|---|---:|
| Python diamond search | 99.07 ms |
| C++ diamond search | 3.53 ms |
| C++ full search | 110.33 ms |

The C++ diamond-search implementation was approximately **28× faster** than the Python diamond-search implementation in this run. The Python and C++ diamond-search implementations produced matching results in the benchmark.

Diamond search evaluates fewer candidate positions than full search, reducing computation but potentially selecting different motion vectors.

### CRF Sweep

On the bundled synthetic sample, using 150 frames and a VMAF target of 93, the sweep recommended **CRF 26**.

The resulting bitrate was approximately **75.5% lower than CRF 18** while meeting the configured VMAF target in this experiment.

This result is specific to the tested sample and settings. Other videos may produce different recommendations.

### Video Enhancement

In a CLAHE processing run on the sample video:

- Mean enhancement time: approximately 2.77 ms per frame.
- End-to-end processing speed: approximately 50 FPS.
- Audio preservation: reported as successful for the tested FFmpeg pipeline.

Enhancement time and end-to-end FPS measure different parts of the pipeline.

## Tests

Run the test suite using:

```bash
python -m pytest
```

The latest recorded test run completed with **152 passed and 1 skipped**.

The tests cover the processing pipeline, enhancement methods, native kernels, complexity analysis, CLI behavior, and encoding-sweep functionality.

## Limitations

- Motion estimation uses integer-pixel block matching and a single reference frame.
- Complexity analysis and scene-cut detection rely on heuristics that have not been calibrated against a large dataset.
- The default CRF sweep evaluates a limited frame excerpt rather than optimizing every scene independently.
- The bundled sample video is synthetic, so results may differ on real-world footage.
- Performance depends on hardware, compiler, input resolution, and installed FFmpeg capabilities.

## Future Improvements

- Evaluate the pipeline on a larger collection of real-world videos.
- Predict CRF settings from measured video-complexity features.
- Add sub-pixel motion estimation and variable block sizes.
- Explore SIMD and multithreaded C++ optimizations.
- Extend encoding analysis to multiple resolutions and bitrate ladders.

## What I Learned

Through this project, I explored Python and C++ integration, native-code benchmarking, video-quality metrics, motion estimation, rate-distortion analysis, and automated testing. I also gained experience building a command-line interface and a dashboard around a reusable processing pipeline.
