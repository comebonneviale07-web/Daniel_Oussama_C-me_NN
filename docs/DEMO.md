# Local museum demo

The interface uses a navy, royal-blue and pale-blue palette inspired by the student's
Albert School and Mines Paris–PSL context. School names are text labels, not official
logo assets. The theme is in `.streamlit/config.toml` and the layout styling in
`src/demo.css`. No external fonts or image assets are required. Restart the server
after changing Streamlit theme settings; ordinary layout changes reload on refresh.
Visual references: [Albert School](https://www.albertschool.com/en/about/) and
[Mines Paris–PSL](https://www.minesparis.psl.eu/). The chosen colours are an inspired
project palette, not a claim to reproduce either institution's official brand guide.

## Start

From the repository root in Git Bash:

```bash
source .venv/Scripts/activate
python -m pip install -r requirements-lock.txt
python main.py demo
```

Open **http://127.0.0.1:8501**. Keep the terminal running. Stop with Ctrl+C.
The server listens only on this computer. No hosting account or external inference API is used.
Once dependencies and weights are installed, the demo can run offline.

If the port is occupied, use `python main.py demo --port 8502`.
To choose a particular trained run:

```bash
python main.py demo --run results/M3/v3_augmented/m3_augmented_1793378930229182800
```

By default the demo reuses `src/predict.py`'s preferred locally available checkpoint.
On this laptop that is the verified `m3_augmented` checkpoint (79.11% on the original
225-image test split). If its weights are absent, the existing pipeline can select another
available model; the "About this guide" section shows which one is being used.
The code does not retrain models, download weights, or change model versions.

## Use

1. Upload a JPG, PNG or WebP painting (up to 10 MB and 25 million pixels).
2. Read the leading style suggestion, the top three model scores and the curated explanation.
3. Optionally download the prediction as JSON.

"Try a collection example" uses a local training image if the dataset is present. It
is explicitly labelled and must not be presented as unseen-image evaluation. Uploads
work without the WikiArt dataset. Camera EXIF orientation is applied before prediction.
Uploaded bytes stay in memory; the app does not save them to disk. Streamlit keeps
session state in memory while the app runs. Usage-statistics collection is disabled by
the launcher. The read-only model is cached to avoid reloading on each interaction.

## Presentation checklist

- Confirm the correct model name in "About this guide" and start the server before presenting.
- Bring a few new painting photos and an out-of-scope image to discuss limitations.
- Keep screenshots or a recording as a fallback; the example button is another fallback.
- Explain that scores are uncalibrated relative outputs, not probabilities of being correct.
- The "tentative" message uses a simple top-score threshold of 0.5; it is not a validated
  rejection rule or an out-of-distribution detector. High scores can also be wrong.
- This classifier covers five styles. Its test accuracy on cropped scans is not an estimate
  of reliability for arbitrary visitor photographs. Curated descriptions are generic visual
  context, not generated artwork-specific facts.

## Implementation and verification

- `src/demo.py`: launcher, in-memory image validation, UI and cached inference.
- `data/style_descriptions.json`: the five curated descriptions and visual cues.
- `tests/prediction/test_demo.py`: image errors and orientation, score handling, UI interaction,
  JSON download presence, and missing-checkpoint state.
- `main.py demo`: launches Streamlit in the same Python environment on localhost.

```bash
python -m unittest -v tests.prediction.test_demo
```

UI checks use [Streamlit AppTest](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest).
The first demo covers upload, classification and style explanations. Similar-painting
retrieval, semantic text search and a public deployment are separate features.
