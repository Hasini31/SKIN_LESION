import html
from pathlib import Path

import joblib
import streamlit as st
from PIL import Image

from src.feature_extraction import extract_features

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"
SCALER_PATH = BASE_DIR / "models" / "scaler.joblib"

CLASS_NAMES = {
    "akiec": "Actinic keratoses / intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}

# General dataset-level grouping (informational only, not a diagnosis).
CLASS_GROUP = {
    "akiec": ("Pre-malignant / malignant type", "#d97706", "#fef3c7"),
    "bcc": ("Malignant type", "#dc2626", "#fee2e2"),
    "bkl": ("Benign type", "#059669", "#d1fae5"),
    "df": ("Benign type", "#059669", "#d1fae5"),
    "mel": ("Malignant type", "#dc2626", "#fee2e2"),
    "nv": ("Benign type", "#059669", "#d1fae5"),
    "vasc": ("Benign type", "#059669", "#d1fae5"),
}

st.set_page_config(
    page_title="Skin Lesion Classification",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif; }

.stApp {
    background:
        radial-gradient(900px 400px at 8% -5%, #e0f2fe 0%, transparent 60%),
        radial-gradient(800px 400px at 100% 0%, #ccfbf1 0%, transparent 55%),
        #f6f9fc;
}
#MainMenu, footer, header[data-testid="stHeader"] { background: transparent; }
footer { visibility: hidden; }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1150px; }

/* Hero */
.hero {
    background: linear-gradient(135deg, #ffffff 0%, #f0fdfa 100%);
    border: 1px solid #e2e8f0;
    border-radius: 22px;
    padding: 2.2rem 2.4rem;
    box-shadow: 0 10px 30px rgba(15, 118, 110, 0.07);
    margin-bottom: 1.4rem;
}
.hero .pill {
    display: inline-block; font-size: .75rem; font-weight: 600; letter-spacing: .06em;
    text-transform: uppercase; color: #0f766e; background: #ccfbf1;
    padding: .3rem .75rem; border-radius: 999px; margin-bottom: .9rem;
}
.hero h1 { font-size: 2.3rem; font-weight: 800; color: #0f172a; margin: 0 0 .5rem 0; line-height: 1.15; }
.hero h1 span { background: linear-gradient(90deg, #0d9488, #2563eb); -webkit-background-clip: text; background-clip: text; color: transparent; }
.hero p { color: #475569; font-size: 1.02rem; margin: 0; max-width: 760px; line-height: 1.6; }
.chips { margin-top: 1.1rem; display: flex; gap: .5rem; flex-wrap: wrap; }
.chip { font-size: .8rem; font-weight: 500; color: #334155; background: #fff; border: 1px solid #e2e8f0; padding: .3rem .7rem; border-radius: 999px; }

/* Disclaimer */
.disclaimer {
    display: flex; gap: .8rem; align-items: flex-start;
    background: #fffbeb; border: 1px solid #fde68a; color: #92400e;
    padding: .9rem 1.1rem; border-radius: 14px; font-size: .9rem; line-height: 1.5; margin-bottom: 1.4rem;
}
.disclaimer b { color: #78350f; }

/* Cards */
.card {
    background: #ffffff; border: 1px solid #e2e8f0; border-radius: 18px;
    padding: 1.4rem 1.5rem; box-shadow: 0 4px 18px rgba(15, 23, 42, 0.04);
}
.card-title { font-size: 1.05rem; font-weight: 700; color: #0f172a; margin: 0 0 .2rem 0; }
.card-sub { font-size: .86rem; color: #64748b; margin: 0 0 1rem 0; }

/* Uploader */
[data-testid="stFileUploader"] section {
    background: #f8fafc; border: 2px dashed #99f6e4; border-radius: 16px; padding: 1.4rem;
    transition: all .2s ease;
}
[data-testid="stFileUploader"] section:hover { background: #f0fdfa; border-color: #14b8a6; }
[data-testid="stFileUploader"] button { border-radius: 10px; }

/* Image */
[data-testid="stImage"] img { border-radius: 14px; border: 1px solid #e2e8f0; box-shadow: 0 6px 20px rgba(15,23,42,.08); }

/* Button */
div.stButton > button {
    width: 100%; border: 0; border-radius: 12px; padding: .75rem 1rem; font-weight: 600; font-size: 1rem;
    background: linear-gradient(90deg, #0d9488, #0ea5e9); color: #fff;
    box-shadow: 0 8px 18px rgba(13, 148, 136, .25); transition: transform .15s ease, box-shadow .15s ease;
}
div.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 12px 24px rgba(13, 148, 136, .32); color: #fff; border: 0; }
div.stButton > button:focus { color: #fff; }

/* Result */
.result-top { display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem; }
.result-code {
    font-size: 1.6rem; font-weight: 800; color: #0f766e; background: #ccfbf1;
    border-radius: 16px; padding: .6rem 1rem; text-transform: uppercase; letter-spacing: .03em;
}
.result-name { font-size: 1.15rem; font-weight: 700; color: #0f172a; line-height: 1.3; }
.badge { display: inline-block; font-size: .75rem; font-weight: 600; padding: .2rem .6rem; border-radius: 999px; margin-top: .35rem; }
.conf { font-size: .85rem; color: #64748b; margin-top: .1rem; }

/* Bars */
.bar-row { margin: .65rem 0; }
.bar-label { display: flex; justify-content: space-between; font-size: .86rem; color: #334155; margin-bottom: .25rem; }
.bar-label .pct { font-weight: 600; color: #0f172a; }
.bar-track { height: 10px; background: #eef2f7; border-radius: 999px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #5eead4, #0d9488); }
.bar-fill.top { background: linear-gradient(90deg, #0d9488, #2563eb); }

/* Placeholder */
.placeholder { text-align: center; padding: 2.5rem 1rem; color: #64748b; }
.placeholder .emoji { font-size: 2.4rem; margin-bottom: .5rem; }

/* Sidebar */
section[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e2e8f0; }
.side-h { font-weight: 700; color: #0f172a; font-size: .95rem; margin: .4rem 0 .6rem 0; }
.cls-item { display: flex; gap: .6rem; align-items: baseline; font-size: .84rem; color: #475569; padding: .25rem 0; }
.cls-item code { background: #f0fdfa; color: #0f766e; padding: .05rem .4rem; border-radius: 6px; font-weight: 600; }
.step { display: flex; gap: .7rem; align-items: flex-start; font-size: .86rem; color: #475569; margin: .5rem 0; }
.step .n { flex: 0 0 24px; height: 24px; border-radius: 50%; background: #ccfbf1; color: #0f766e; font-weight: 700; font-size: .75rem; display: flex; align-items: center; justify-content: center; }

.foot { text-align: center; color: #94a3b8; font-size: .8rem; margin-top: 2.2rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file not found. Put your final trained model at "
            "models/best_model.joblib."
        )
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH) if SCALER_PATH.exists() else None
    return model, scaler


# ---------------------------------------------------------------- Sidebar
with st.sidebar:
    st.markdown("### 🔬 DermaScope")
    st.caption("Academic ML demonstration")

    st.markdown('<div class="side-h">How it works</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="step"><div class="n">1</div><div>Upload a lesion image (JPG or PNG).</div></div>
        <div class="step"><div class="n">2</div><div>HOG + HSV features are extracted from the image.</div></div>
        <div class="step"><div class="n">3</div><div>The trained classifier scores all 7 classes.</div></div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="side-h" style="margin-top:1.2rem">The 7 classes</div>', unsafe_allow_html=True)
    st.markdown(
        "".join(
            f'<div class="cls-item"><code>{k}</code><span>{html.escape(v)}</span></div>'
            for k, v in CLASS_NAMES.items()
        ),
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------- Hero
st.markdown(
    """
    <div class="hero">
        <div class="pill">Educational prototype</div>
        <h1>Skin Lesion <span>Classification</span></h1>
        <p>An academic machine-learning demonstration that analyses texture (HOG) and colour (HSV)
        features of a dermoscopic image and scores it against seven lesion categories.</p>
        <div class="chips">
            <span class="chip">HAM10000</span>
            <span class="chip">HOG + HSV features</span>
            <span class="chip">7 classes</span>
            <span class="chip">Classical ML</span>
        </div>
    </div>
    <div class="disclaimer">
        <div style="font-size:1.2rem">⚠️</div>
        <div><b>Not a medical device.</b> This tool must not be used to diagnose, rule out, or treat any
        medical condition. Please consult a qualified dermatologist for medical advice.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- Layout
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown(
        '<div class="card-title">1 · Upload image</div>'
        '<div class="card-sub">A clear, well-lit close-up works best.</div>',
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "Upload a skin lesion image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    image = None
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption=uploaded_file.name, use_container_width=True)
        except Exception as exc:
            st.error(f"Could not read this image: {exc}")

    classify = st.button("Classify image", type="primary", disabled=image is None)

with right:
    st.markdown(
        '<div class="card-title">2 · Results</div>'
        '<div class="card-sub">Model output appears here.</div>',
        unsafe_allow_html=True,
    )
    result_box = st.container()

    if not classify:
        with result_box:
            st.markdown(
                """
                <div class="card"><div class="placeholder">
                    <div class="emoji">🧬</div>
                    <div style="font-weight:600;color:#334155">No results yet</div>
                    <div style="font-size:.88rem;margin-top:.3rem">Upload an image and press
                    <b>Classify image</b> to see the prediction.</div>
                </div></div>
                """,
                unsafe_allow_html=True,
            )

if classify and image is not None:
    with result_box:
        try:
            with st.spinner("Extracting image features and predicting..."):
                model, scaler = load_artifacts()
                features = extract_features(image)
                if features.shape[1] != 6132:
                    raise ValueError(
                        f"Expected 6,132 features, but extracted "
                        f"{features.shape[1]}. Check the original HPC HOG/HSV "
                        "settings in src/feature_extraction.py."
                    )
                model_input = scaler.transform(features) if scaler is not None else features
                prediction = model.predict(model_input)[0]

            label = str(prediction)
            full_name = CLASS_NAMES.get(label, label)
            group, fg, bg = CLASS_GROUP.get(label, ("Unknown group", "#475569", "#e2e8f0"))

            probs = None
            conf_html = ""
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(model_input)[0]
                classes = getattr(model, "classes_", range(len(proba)))
                probs = sorted(
                    ((str(c), float(p)) for c, p in zip(classes, proba)),
                    key=lambda x: x[1],
                    reverse=True,
                )
                conf_html = f'<div class="conf">Model score: {probs[0][1] * 100:.1f}%</div>'

            st.markdown(
                f"""
                <div class="card">
                    <div class="result-top">
                        <div class="result-code">{html.escape(label)}</div>
                        <div>
                            <div class="result-name">{html.escape(full_name)}</div>
                            <span class="badge" style="color:{fg};background:{bg}">{group}</span>
                            {conf_html}
                        </div>
                    </div>
                """,
                unsafe_allow_html=True,
            )

            if probs:
                bars = ""
                for i, (c, p) in enumerate(probs):
                    cls = "bar-fill top" if i == 0 else "bar-fill"
                    bars += (
                        f'<div class="bar-row"><div class="bar-label">'
                        f"<span>{html.escape(CLASS_NAMES.get(c, c))}</span>"
                        f'<span class="pct">{p * 100:.1f}%</span></div>'
                        f'<div class="bar-track"><div class="{cls}" style="width:{max(p * 100, 0.6):.1f}%"></div></div></div>'
                    )
                st.markdown(
                    f'<div class="side-h" style="margin-top:.4rem">Class scores</div>{bars}'
                    '<div style="font-size:.78rem;color:#94a3b8;margin-top:.9rem">'
                    "Model scores / probabilities, not calibrated medical risk estimates.</div></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div style="font-size:.85rem;color:#64748b">This model does not expose '
                    "predict_proba().</div></div>",
                    unsafe_allow_html=True,
                )

        except FileNotFoundError as exc:
            st.warning(f"📦 {exc}")
        except Exception as exc:
            st.error(f"Prediction could not be completed: {exc}")

st.markdown(
    '<div class="foot">Model: classical machine learning · Features: HOG + HSV · '
    "Dataset: HAM10000 · Classes: 7<br>Educational use only</div>",
    unsafe_allow_html=True,
)
