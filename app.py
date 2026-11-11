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

def predict_and_show(img):
    st.image(img, caption="Test Image", use_container_width=True)
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
        st.write(f"{label}: {float(pred[0][i]) * 100:.1f}%")

st.write("### Try Demo Images - No Upload Needed")
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("Show Closed"):
        predict_and_show(Image.open("demo_Closed.jpg").convert("RGB"))
with col2:
    if st.button("Show Open"):
        predict_and_show(Image.open("demo_Open.jpg").convert("RGB"))
with col3:
    if st.button("Show Yawn"):
        predict_and_show(Image.open("demo_yawn.jpg").convert("RGB"))
with col4:
    if st.button("Show No Yawn"):
        predict_and_show(Image.open("demo_no_yawn.jpg").convert("RGB"))

st.write("---")
st.write("### Or Upload Your Own Image")
uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    predict_and_show(img)
