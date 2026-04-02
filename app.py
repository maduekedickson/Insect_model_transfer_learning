import streamlit as st
import tensorflow as tf
import keras
import numpy as np
from PIL import Image
import gdown
import os

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Insect AI - Transfer Model", layout="centered")

st.title("🧠 Insect Classification AI (Transfer Learning)")
st.caption("High-accuracy model using MobileNetV2")
st.divider()

# =========================
# MODEL DOWNLOAD
# =========================

FILE_ID = "1rEOLOTp6rChtflDyYA0OMtK8jweuEjS2"
MODEL_URL = f"https://drive.google.com/uc?id={FILE_ID}"
MODEL_PATH = "transfer_model.keras"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

    return keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

# =========================
# CLASS NAMES
# =========================

class_names = [
    "Aedes_aegypti","Aedes_albopictus","Aedes_vexans",
    "Amblyomma_americanum","Anopheles_stephensi",
    "Anopheles_tessellatus","Cimex_lectularius",
    "Ctenocephalides_canis","Ctenocephalides_felis",
    "Culex_quinquefasciatus","Culex_vishnui",
    "Ixodes_ricinus","Ixodes_scapularis",
    "Pediculus_humanus_capitis",
    "Pediculus_humanus_corporis",
    "Rhipicephalus_sanguineus"
]

# =========================
# UI
# =========================

uploaded_file = st.file_uploader("📤 Upload an insect image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    st.divider()

    img = image.resize((192,192))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("🔍 Classifying..."):
        preds = model.predict(img_array)[0]

    pred_class = class_names[np.argmax(preds)]
    confidence = float(np.max(preds))

    st.success(f"✅ Prediction: {pred_class}")
    st.info(f"Confidence: {confidence:.2f}")

    st.progress(confidence)

    st.subheader("🔝 Top 3 Predictions")
    top_indices = np.argsort(preds)[-3:][::-1]

    for i in top_indices:
        st.write(f"{class_names[i]} — {preds[i]:.2f}")
