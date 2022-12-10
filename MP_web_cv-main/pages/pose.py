import cv2
import mediapipe as mp
import streamlit as st
from PIL import Image
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def main(image_file):
    with mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    min_detection_confidence=0.5) as pose:
        image = image_file
    # Convert the BGR image to RGB before processing.
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        annotated_image = image.copy()
        mp_drawing.draw_landmarks(
            annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        mp_drawing.plot_landmarks(
        results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

    return annotated_image


def app():
    html_temp = """
        <body style="background-color:red;">
        <div style="background-color:MidnightBlue ;padding:10px">
        <h2 style="color:white;text-align:center;">Pose estimation</h2>
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
