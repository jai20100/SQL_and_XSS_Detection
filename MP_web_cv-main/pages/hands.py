import cv2
import mediapipe as mp
import streamlit as st
import numpy as np
from PIL import Image

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
drawing_styles = mp.solutions.drawing_styles

def main(image_file):
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0) as hands:
        image = image_file
        # Convert the BGR image to RGB before processing.
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_hand_landmarks:
            st.text("No HANDS")
        # Print handedness and draw hand landmarks on the image.
        print('Handedness:', results.multi_handedness)
        image_height, image_width, _ = image.shape
        annotated_image = image.copy()
        for hand_landmarks in results.multi_hand_landmarks:
            print('hand_landmarks:', hand_landmarks)
            print(
            f'Index finger tip coordinates: (',
            f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
            f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
        )
            mp_drawing.draw_landmarks(
                annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                drawing_styles.get_default_hand_landmark_style(),
                drawing_styles.get_default_hand_connection_style())
    return annotated_image


def app():
    html_temp = """
        <body style="background-color:red;">
        <div style="background-color:Teal ;padding:10px">
        <h2 style="color:white;text-align:center;">Hand Finder</h2>
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

