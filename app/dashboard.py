"""FrameLab dashboard:  streamlit run app/dashboard.py"""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:          # lets `streamlit run app/dashboard.py` find the framelab package
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from framelab import kernels  # noqa: E402
from framelab.benchmarks import kernel_benchmark, motion_benchmark  # noqa: E402
from framelab.complexity import analyze_complexity  # noqa: E402
from framelab.enhancement import METHODS  # noqa: E402
from framelab.ffmpeg_tools import capabilities  # noqa: E402
from framelab.pipeline import process_video  # noqa: E402
from framelab.sweep import crf_sweep  # noqa: E402

OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

st.set_page_config(page_title="FrameLab", page_icon="🎞️", layout="wide")
st.title("🎞️ FrameLab")
st.caption("Video enhancement, content-complexity analysis and content-aware encoding study "
           "(Python + OpenCV + ffmpeg, with a C++ motion-estimation core).")

caps = capabilities()
with st.sidebar:
    st.subheader("System")
    st.write("C++ native module:", "✅ loaded" if kernels.native_available() else "⚠️ not built (Python fallback)")
    st.write("ffmpeg / x264:", "✅" if caps["libx264"] else "⚠️ missing (OpenCV fallback)")
    st.write("libvmaf:", "✅" if caps["libvmaf"] else "➖ not in this ffmpeg (SSIM/PSNR used)")
    if not kernels.native_available():
        st.info("Build it: `python setup.py build_ext --inplace`")

uploaded = st.file_uploader("Upload a short video (5-30 s works best)", type=["mp4", "avi", "mov", "mkv"])
input_path = None
if uploaded:
    if st.session_state.get("upload_id") != uploaded.file_id:
        tmp = Path(tempfile.mkdtemp(prefix="framelab_")) / uploaded.name
        tmp.write_bytes(uploaded.getbuffer())
        st.session_state.update(upload_id=uploaded.file_id, input_path=str(tmp), results={})
    input_path = Path(st.session_state["input_path"])
    st.video(uploaded)
results = st.session_state.setdefault("results", {})

tab_enh, tab_cx, tab_sweep, tab_bench = st.tabs(["Enhance", "Complexity", "Encoding sweep", "Benchmarks"])

# ------------------------------------------------------------------ Enhance
with tab_enh:
    c1, c2, c3 = st.columns(3)
    method = c1.selectbox("Method", METHODS, index=1)
    max_frames = c1.number_input("Max frames (0 = all)", min_value=0, value=0, step=50)
    clip_limit = c2.slider("CLAHE clip limit", 1.0, 4.0, 2.0, 0.25)
    denoise = c2.slider("Denoise strength", 1, 12, 5)
    crf = c3.slider("Output x264 CRF (lower = better)", 14, 30, 18)
    analyze = c3.checkbox("Before/after quality metrics", value=True)
    if st.button("Process video", type="primary", disabled=input_path is None):
        out = OUT / f"{input_path.stem}_{method}.mp4"
        bar, status = st.progress(0.0), st.empty()

        def update(value, count):
            if value is not None:
                bar.progress(float(value))
            status.write(f"Processed {count} frames…")

        try:
            results["enhance"] = process_video(
                input_path, out, method=method, progress_callback=update, analyze=analyze, crf=crf,
                max_frames=int(max_frames) or None,
                params={"clip_limit": clip_limit, "denoise_strength": denoise})
        except Exception as exc:  # show, don't crash the app
            st.error(f"Processing failed: {exc}")
    r = results.get("enhance")
    if r:
        m = st.columns(5)
        m[0].metric("Frames", r["frames_processed"])
        m[1].metric("End-to-end FPS", f'{r["fps_processed"]:.1f}')
        m[2].metric("Enhance time", f'{r["mean_enhance_ms"]:.1f} ms/frame')
        m[3].metric("Output bitrate", f'{r["output_bitrate_kbps"]:.0f} kbps')
        m[4].metric("Real-time?", "yes" if r["realtime_capable"] else "no")
        st.caption(f'Writer: {r["writer_backend"]} · audio kept: {r["audio_preserved"]}')
        if "mean_psnr_vs_input_db" in r:
            st.table(pd.DataFrame({
                "metric": ["brightness", "blur proxy (Laplacian var)", "noise proxy"],
                "input": [r["mean_brightness_mean_in"], r["mean_blur_laplacian_variance_in"], r["mean_noise_residual_std_in"]],
                "output": [r["mean_brightness_mean_out"], r["mean_blur_laplacian_variance_out"], r["mean_noise_residual_std_out"]],
            }).round(2))
            st.caption(f'PSNR of output vs. input: {r["mean_psnr_vs_input_db"]:.1f} dB (how far the enhancement moved the pixels). '
                       "Blur/noise are heuristics, not perceptual quality.")
        st.video(r["output"])
        d1, d2 = st.columns(2)
        d1.download_button("Download video", Path(r["output"]).read_bytes(), Path(r["output"]).name, "video/mp4")
        d2.download_button("Download metrics CSV", Path(r["csv"]).read_bytes(), Path(r["csv"]).name, "text/csv")
        chart_df = pd.DataFrame(r["frame_metrics"]).set_index("frame_index")
        chart_df = chart_df[["enhance_ms"]]

        # Exclude the first frame from the display to avoid startup-time distortion.
        if len(chart_df) > 1:
            chart_df = chart_df.iloc[1:]

        st.line_chart(chart_df)

# ------------------------------------------------------------------ Complexity
with tab_cx:
    st.write("Spatial detail + motion-compensated temporal residual per frame, and scene cuts. "
             "Runs on the C++ kernels when built.")
    use_native = st.checkbox("Use native C++ kernels", value=kernels.native_available(),
                             disabled=not kernels.native_available())
    cx_frames = st.number_input("Max frames (0 = all)", 0, value=300, step=50, key="cxf")
    if st.button("Analyze complexity", disabled=input_path is None):
        with st.spinner("Analyzing…"):
            try:
                results["cx"] = analyze_complexity(input_path, max_frames=int(cx_frames) or None,
                                                   prefer_native=use_native)
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")
    if "cx" in results:
        s, df = results["cx"]
        m = st.columns(4)
        m[0].metric("Complexity index", f'{s["complexity_index"]:.1f}')
        m[1].metric("Spatial", f'{s["mean_spatial_complexity"]:.1f}')
        m[2].metric("Temporal residual", f'{s["mean_temporal_residual"]:.1f}')
        m[3].metric("Scene cuts", s["scene_cuts"])
        st.caption(f'Cut frames: {s["scene_cut_frames"] or "none"} · analysis took {s["analysis_seconds"]:.2f}s '
                   f'({"C++" if s["native_kernels"] else "Python"} kernels)')
        st.line_chart(df.set_index("frame_index")[["spatial", "temporal_residual"]])

# ------------------------------------------------------------------ Sweep
with tab_sweep:
    st.write("Encode with x264 at several CRFs, score each against the source, and recommend the "
             "cheapest setting that still meets a quality target.")
    if not caps["libx264"]:
        st.warning("This tab needs ffmpeg with libx264 (`pip install imageio-ffmpeg`).")
    c1, c2 = st.columns(2)
    crfs = c1.multiselect("CRF values", list(range(14, 41, 2)), default=[18, 22, 26, 30, 34])
    metrics = (["vmaf"] if caps["libvmaf"] else []) + ["ssim", "psnr"]
    metric = c2.selectbox("Quality metric", metrics)
    defaults = {"vmaf": 93.0, "ssim": 0.95, "psnr": 38.0}
    target = c2.number_input("Target", value=defaults[metric], format="%.3f")
    sw_frames = c1.number_input("Frames to encode", 10, value=100, step=10)
    if st.button("Run sweep", disabled=input_path is None or not caps["libx264"] or not crfs):
        with st.spinner("Encoding and measuring…"):
            try:
                results["sweep"] = crf_sweep(input_path, OUT / "sweep", crfs=crfs, max_frames=int(sw_frames),
                                             metric=metric, target=target)
            except Exception as exc:
                st.error(f"Sweep failed: {exc}")
    r = results.get("sweep")
    if r:
        st.success(r["note"])
        if "bitrate_saving_vs_lowest_crf_pct" in r:
            st.metric("Bitrate saved vs. lowest CRF tested", f'{r["bitrate_saving_vs_lowest_crf_pct"]:.0f}%')
        st.image(r["plot"])
        st.dataframe(pd.DataFrame(r["table"]).round(3), hide_index=True)
        if "complexity" in r:
            st.caption(f'Content complexity index: {r["complexity"]["complexity_index"]:.1f} '
                       "(compare across clips: busier content needs more bitrate for the same quality).")

# ------------------------------------------------------------------ Benchmarks
with tab_bench:
    st.write("Honest micro-benchmarks. Timings depend on your hardware; the *identical output* flags matter most.")
    b1, b2 = st.columns(2)
    if b1.button("Brightness/contrast kernel"):
        with st.spinner("Running…"):
            results["kb"] = kernel_benchmark(1280, 720, 10)
    if b2.button("Motion estimation (Python vs C++)"):
        with st.spinner("Running…"):
            results["mb"] = motion_benchmark(320, 176, 3)
    if "kb" in results:
        k = results["kb"]
        st.dataframe(pd.DataFrame({"ms/call": k["ms_per_call"], "speedup vs NumPy": k["speedup_vs_numpy"],
                                   "identical to NumPy": k["identical_to_numpy"]}).round(2))
    if "mb" in results:
        k = results["mb"]
        st.dataframe(pd.DataFrame({"ms/frame pair": k["ms_per_frame_pair"],
                                   "mean SAD/block": k["mean_sad_per_block"]}).round(2))
        if "speedup_cpp_diamond_vs_python_diamond" in k:
            st.write(f'C++ diamond is {k["speedup_cpp_diamond_vs_python_diamond"]:.1f}× faster than the Python '
                     f'reference (identical vectors: {k["python_equals_cpp_diamond"]}); diamond is '
                     f'{k["speedup_diamond_vs_full_search"]:.0f}× faster than full search at '
                     f'{(k["diamond_sad_vs_full_sad_ratio"] - 1) * 100:.1f}% higher residual.')
        else:
            st.info(k.get("note", ""))
