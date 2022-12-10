import cv2
import mediapipe as mp
import streamlit as st
from PIL import Image
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic


def main(image_file):
    with mp_holistic.Holistic(
    static_image_mode=True,
    model_complexity=2) as holistic:
        image = image_file
        image_height, image_width, _ = image.shape
        results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.pose_landmarks:
      print(
          f'Nose coordinates: ('
          f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x * image_width}, '
          f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y * image_height})'
      )
    # Draw pose, left and right hands, and face landmarks on the image.
    annotated_image = image.copy()
    mp_drawing.draw_landmarks(
        annotated_image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)
    mp_drawing.draw_landmarks(
        annotated_image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(
        annotated_image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(
        annotated_image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    # Plot pose world landmarks.
    mp_drawing.plot_landmarks(
        results.pose_world_landmarks, mp_holistic.POSE_CONNECTIONS)

    return annotated_image


def app():
    html_temp = """
        <body style="background-color:red;">
        <div style="background-color:DeepSkyBlue ;padding:10px">
        <h2 style="color:white;text-align:center;">Hostilic image</h2>
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
