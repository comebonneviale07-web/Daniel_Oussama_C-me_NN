"""Local museum guide. Start from the project root with python main.py demo."""
import argparse
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

# Streamlit also executes this file as a standalone app script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from src import predict as inference

MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 25_000_000


def read_image(content):
    """Decode an upload in memory, respecting camera orientation and image limits."""
    if not content or len(content) > MAX_BYTES:
        raise ValueError("Choose an image smaller than 10 MB.")
    try:
        with Image.open(io.BytesIO(content)) as image:
            if image.format not in ("JPEG", "PNG", "WEBP"):
                raise ValueError("Please use a JPG, PNG or WebP image.")
            if image.width * image.height > MAX_PIXELS:
                raise ValueError("Please resize this image to fewer than 25 million pixels.")
            image.load()
            return ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError("This image could not be opened. Try a different JPG, PNG or WebP file.") from error


def load_descriptions():
    return json.loads((ROOT / "data/style_descriptions.json").read_text(encoding="utf-8"))


def example_path():
    """Use a training image, clearly labelled as a collection example, when data exists."""
    split = ROOT / "data/splits/style5.json"
    if split.exists():
        for relative in json.loads(split.read_text(encoding="utf-8"))["train"]:
            path = ROOT / "data/wikiart_style5" / relative
            if path.is_file():
                return path
    return None


@st.cache_resource(show_spinner=False, max_entries=2)
def cached_model(run_path, checkpoint_mtime, threads):
    """Read-only CPU model shared across reruns; invalidate when weights change."""
    torch.set_num_threads(threads)
    return inference.load_run(Path(run_path))


def analyse_image(image, loaded_model):
    model, model_name, classes = loaded_model
    started = time.perf_counter()
    ranking = inference.predict(model, model_name, classes, image, top=3)
    return {
        "model": model_name,
        "predictions": [{"style": style, "score": score} for style, score in ranking],
        "tentative": ranking[0][1] < .5,
        "inference_ms": (time.perf_counter() - started) * 1000,
    }


def render_header():
    stylesheet = Path(__file__).with_suffix(".css").read_text(encoding="utf-8")
    st.markdown(f"<style>{stylesheet}</style>", unsafe_allow_html=True)
    st.markdown("""<div class="school-banner">
      <div class="school-lockup">
        <div class="school-albert">ALBERT <span>SCHOOL</span></div>
        <span class="school-cross" aria-hidden="true">×</span>
        <div class="school-mines">MINES PARIS<span>PSL</span></div>
      </div><div class="project-pill">STUDENT PROJECT · ART &amp; AI</div>
    </div><div class="hero-eyebrow">MUSEUM AI GUIDE</div>""", unsafe_allow_html=True)
    st.title("Look closer at a painting.")
    st.markdown('<div class="hero-rule"></div>', unsafe_allow_html=True)
    st.write("Upload a painting to explore its possible style and the visual ideas behind it.")
    st.markdown("""<div class="flow-row">
      <span class="flow-chip"><b>01</b> Choose a painting</span>
      <span class="flow-chip"><b>02</b> Explore its style</span>
      <span class="flow-chip"><b>03</b> Look a little closer</span>
    </div>""", unsafe_allow_html=True)


def selected_run():
    requested = os.environ.get("MUSEUM_DEMO_RUN")
    run = Path(requested).resolve() if requested else inference.default_run(ROOT / "results")
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    model_kind = inference.get_version(config["model"])["kind"]
    checkpoint = run / ("best.joblib" if model_kind == "logistic" else "best.pt")
    if not checkpoint.is_file():
        raise FileNotFoundError("Missing model weights")
    return run, config, checkpoint


def use_uploaded_image():
    st.session_state["use_example"] = False


def render_upload():
    st.subheader("Your painting")
    uploaded = st.file_uploader(
        "Choose a painting",
        type=["jpg", "jpeg", "png", "webp"],
        key="painting",
        on_change=use_uploaded_image,
    )
    sample = example_path()
    if st.button("Try a collection example", disabled=sample is None):
        st.session_state["use_example"] = True

    if st.session_state.get("use_example", False) and sample:
        content, label = sample.read_bytes(), "Collection example"
        st.caption("An image from the training collection; this is a demonstration, not an independent test.")
    elif uploaded is not None:
        content, label = uploaded.getvalue(), uploaded.name
    else:
        content, label = None, ""

    st.caption("JPG, PNG or WebP · up to 10 MB · images stay in this local session")
    if content is None:
        return None, label
    image = read_image(content)
    st.image(image, caption=label, width="stretch")
    return image, label


def render_style_list(descriptions):
    st.subheader("A new way to look")
    st.write("Choose a painting on the left. The guide will compare it with five artistic movements.")
    st.divider()
    for info in descriptions.values():
        st.markdown(f"**{info['label']}**")
        st.caption(" · ".join(info["look_for"]))


def render_prediction(result, image_label, descriptions):
    leading_style = result["predictions"][0]["style"]
    info = descriptions[leading_style]
    st.caption("TENTATIVE SUGGESTION" if result["tentative"] else "LEADING STYLE SUGGESTION")
    st.header(info["label"])
    if result["tentative"]:
        st.info("The scores are spread across styles. Treat this match as a starting point for exploration.")
    st.write(info["description"])
    st.markdown("**Look for**")
    st.write(" · ".join(info["look_for"]))
    st.divider()
    st.markdown("**Three leading matches**")
    for prediction in result["predictions"]:
        style_label = descriptions[prediction["style"]]["label"]
        st.progress(prediction["score"], text=f"{style_label} — {prediction['score']:.1%}")
    st.caption("Relative model scores, not calibrated certainty. Only the top three of five styles are shown.")
    st.download_button(
        "Save this prediction",
        json.dumps({**result, "image": image_label}, indent=2),
        file_name="museum-style-prediction.json",
        mime="application/json",
    )


def render_footer(run, config):
    st.divider()
    st.markdown("""<div class="museum-footer">
      <strong>ALBERT SCHOOL × MINES PARIS – PSL</strong>
      <span>Computer vision meets art history · Student demonstration</span>
    </div>""", unsafe_allow_html=True)
    st.caption("A coursework exploration of five styles. Artworks outside these movements may receive misleading matches.")
    with st.expander("About this guide"):
        st.write("The descriptions are curated text. Predictions come from the locally saved classifier; no image is sent to an external prediction service.")
        st.code(config["model"])
        metrics_path = run / "test_metrics.json"
        if metrics_path.is_file():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            st.write(f"Reported accuracy on 225 held-out WikiArt images: {metrics['accuracy']:.1%}.")
        st.caption("One image-level split; simulated photo conditions differ from real museum photos. Uploads are processed in memory and are not written to disk by the demo.")


def render():
    st.set_page_config(page_title="Museum AI Guide", page_icon="🖼️", layout="wide")
    render_header()
    descriptions = load_descriptions()
    try:
        run, config, checkpoint = selected_run()
    except (KeyError, ValueError, OSError):
        st.error("The guide needs a trained model. Place its best.pt beside its config.json, then restart the demo.")
        st.caption("Setup instructions: docs/DEMO.md")
        return

    left, right = st.columns([1.15, 1], gap="large")
    with left, st.container(border=True, key="painting_panel"):
        try:
            image, label = render_upload()
        except ValueError as error:
            st.error(str(error))
            return

    with right, st.container(border=True, key="prediction_panel"):
        if image is None:
            render_style_list(descriptions)
        else:
            with st.spinner("Looking at colour, form and texture…"):
                try:
                    threads = int(os.environ.get("MUSEUM_DEMO_THREADS", "2"))
                    loaded_model = cached_model(str(run), checkpoint.stat().st_mtime_ns, threads)
                    result = analyse_image(image, loaded_model)
                except (OSError, ValueError, KeyError, RuntimeError) as error:
                    st.error("The trained model could not be loaded or used. Check the selected run and its weights.")
                    with st.expander("Setup details"):
                        st.code(str(error))
                    return
            render_prediction(result, label, descriptions)
    render_footer(run, config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, help="Use this run instead of the preferred locally available model")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or args.threads < 1:
        parser.error("Use a port from 1 to 65535 and a positive thread count")
    environment = os.environ.copy()
    environment["MUSEUM_DEMO_THREADS"] = str(args.threads)
    environment.pop("MUSEUM_DEMO_RUN", None)
    if args.run:
        environment["MUSEUM_DEMO_RUN"] = str(args.run.resolve())
    print(f"Open the museum demo at http://127.0.0.1:{args.port} (Ctrl+C to stop)", flush=True)
    try:
        code = subprocess.call([sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve()),
                                "--server.address", "127.0.0.1", "--server.port", str(args.port),
                                "--server.headless", "true", "--server.maxUploadSize", "10",
                                "--browser.gatherUsageStats", "false"], cwd=ROOT, env=environment)
    except KeyboardInterrupt:
        return
    if code:
        raise SystemExit(code)


if __name__ == "__main__":
    render()
