
import streamlit as st
import cv2
from PIL import Image
import numpy as np
from ultralytics import YOLO

# Load your PPE detection model
@st.cache_resource
def load_model():
    model = YOLO('/Users/abc/Downloads/PPE_YOLOsnippet_Oct20_1351pm/best-2.pt')  # Replace with your model path
    return model

model = load_model()

st.title("PPE Detection System")
st.write("Detect whether people are wearing proper personal protective equipment")

# Sidebar for mode selection
mode = st.sidebar.selectbox("Choose Mode", ["Image Upload", "Video Upload", "Webcam (Real-time)"])

if mode == "Image Upload":
    uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Convert to OpenCV format
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        # Run detection
        results = model(img_array)
        
        # Display results
        annotated_img = results[0].plot()
        st.image(annotated_img, caption="Detection Results")
        
        # Show detection details
        for box in results[0].boxes:
            class_name = model.names[int(box.cls)]
            confidence = float(box.conf)
            st.write(f"Detected: {class_name} (Confidence: {confidence:.2f})")

elif mode == "Video Upload":
    uploaded_video = st.file_uploader("Upload a video", type=['mp4', 'avi', 'mov'])
    
    if uploaded_video is not None:
        # Save uploaded video temporarily
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_video.read())
        
        # Process video
        cap = cv2.VideoCapture("temp_video.mp4")
        stframe = st.empty()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run detection
            results = model(frame)
            annotated_frame = results[0].plot()
            
            # Display frame
            stframe.image(annotated_frame, channels="BGR")
        
        cap.release()

elif mode == "Webcam (Real-time)":
    st.write("Real-time webcam detection")
    run_webcam = st.checkbox("Start Webcam")
    
    if run_webcam:
        cap = cv2.VideoCapture(0)  # 0 for built-in webcam
        stframe = st.empty()
        stop_button = st.button("Stop")
        
        while cap.isOpened() and not stop_button:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access webcam")
                break
            
            # Run detection
            results = model(frame)
            annotated_frame = results[0].plot()
            
            # Display frame
            stframe.image(annotated_frame, channels="BGR")
        
        cap.release()