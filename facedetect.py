import streamlit as st
import cv2
from deepface import DeepFace
from PIL import Image
import tempfile
import numpy as np
import os

st.set_page_config(page_title="AI Face Analyzer", layout="centered")

st.title("🧠 AI Face Analyzer")
st.markdown("Analyze **Emotion**, **Gender**, **Age**, and **Race** from a human face using webcam or image upload.")

# Sidebar controls
st.sidebar.header("🔧 Settings")
input_mode = st.sidebar.radio("Choose Input Mode:", ["Upload Image", "Use Webcam"])
analysis_mode = st.sidebar.selectbox(
    "Select Analysis Mode",
    options=["emotion", "age", "gender", "race", "all"]
)

def analyze_face(img, actions):
    try:
        result = DeepFace.analyze(img_path=img, actions=actions, enforce_detection=False)
        return result[0]
    except Exception as e:
        return {"error": str(e)}

def draw_info_on_frame(frame, analysis_result):
    y0 = 30
    dy = 30
    for i, (key, value) in enumerate(analysis_result.items()):
        if key != "region":
            text = f"{key.capitalize()}: {value}"
            cv2.putText(frame, text, (10, y0 + i*dy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 0), 2)
    return frame

# Upload mode
if input_mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            image.save(temp.name)
            actions = [analysis_mode] if analysis_mode != "all" else ['emotion', 'age', 'gender', 'race']
            result = analyze_face(temp.name, actions)

        if "error" in result:
            st.error(f"Error: {result['error']}")
        else:
            st.subheader("🧾 Analysis Result:")
            for k, v in result.items():
                if k != "region":
                    st.write(f"**{k.capitalize()}**: {v}")

# Webcam mode
else:
    run = st.checkbox("🎥 Start Webcam")
    stop = st.button("🛑 Stop Webcam")

    FRAME_WINDOW = st.image([])
    camera = None

    if run and not stop:
        camera = cv2.VideoCapture(0)

    if stop and camera:
        camera.release()
        st.warning("Webcam Stopped.")
        st.stop()

    if run and not stop:
        actions = [analysis_mode] if analysis_mode != "all" else ['emotion', 'age', 'gender', 'race']

        while camera.isOpened():
            success, frame = camera.read()
            if not success:
                st.error("Failed to read from webcam.")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Save temporary frame to analyze
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
                cv2.imwrite(temp.name, cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
                result = analyze_face(temp.name, actions)

            if "error" not in result:
                rgb_frame = draw_info_on_frame(rgb_frame, result)
            else:
                cv2.putText(rgb_frame, "Face Not Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            FRAME_WINDOW.image(rgb_frame)

    elif not run:
        st.info("Click **Start Webcam** to begin.")
