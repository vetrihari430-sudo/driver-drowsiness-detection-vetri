# app.py - Driver Drowsiness Detection
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Driver Drowsiness Detection", layout="centered")
st.title("Driver Drowsiness Detection")
st.write("Upload an eye / face image to check Drowsiness")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("drowsiness_final_model.h5")
    return model

model = load_model()
labels = ["Closed", "Open", "no_yawn", "yawn"]

uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array, verbose=0)
    class_id = int(np.argmax(pred))
    confidence = float(np.max(pred) * 100)
    predicted = labels[class_id]

    st.write(f"Predicted: **{predicted}**")
    st.write(f"Confidence: **{confidence:.1f}%**")

    if predicted in ["Closed", "yawn"]:
        st.error("DROWSY - ALERT! Please Take a Break")
    else:
        st.success("ALERT - SAFE to Drive")

    st.write("Confidence for all classes:")
    for i, label in enumerate(labels):
        st.write(f"{label}: {float(pred[i][0]) * 100:.1f}%")
