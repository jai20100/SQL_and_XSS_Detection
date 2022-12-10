import cv2
import mediapipe as mp
import streamlit as st
from PIL import Image
import numpy as np

# streamlit run face_detection.py

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def main(image_file):
    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5) as face_detection:

        image = image_file
        # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


        annotated_image = image.copy()
        for detection in results.detections:
            print(mp_face_detection.get_key_point(
                detection, mp_face_detection.FaceKeyPoint.NOSE_TIP))
            mp_drawing.draw_detection(annotated_image, detection)

    return annotated_image

def app():
    html_temp = """
        <body style="background-color:red;">
        <div style="background-color:DeepPink ;padding:10px">
        <h2 style="color:white;text-align:center;">Face Detection for Images</h2>
        </div>
        </body>
        """
    st.markdown(html_temp, unsafe_allow_html=True)


    image_file = st.file_uploader("Upload Image", type=['jpg'])
    if image_file is not None:
        our_image = np.array(Image.open(image_file))
        st.text("DONE")
    if st.button("Find"):  
        output = main(our_image)
        st.image(output)



