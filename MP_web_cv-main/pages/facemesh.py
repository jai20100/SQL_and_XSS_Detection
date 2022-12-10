import cv2
import mediapipe as mp
import streamlit as st
from PIL import Image
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
def main(image_file):
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=0) as face_mesh:
            image = image_file
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            annotated_image = image.copy()
            for face_landmarks in results.multi_face_landmarks:
                print('face_landmarks:', face_landmarks)
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACE_CONNECTIONS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)

    return annotated_image


def app():
    html_temp = """
        <body style="background-color:red;">
        <div style="background-color:GreenYellow ;padding:10px">
        <h2 style="color:white;text-align:center;">Face mesh</h2>
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
