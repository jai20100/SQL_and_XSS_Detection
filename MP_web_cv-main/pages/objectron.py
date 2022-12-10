import cv2
import mediapipe as mp
import streamlit as st
import numpy as np
from PIL import Image


mp_drawing = mp.solutions.drawing_utils
mp_objectron = mp.solutions.objectron

def main(image_file):
    with mp_objectron.Objectron(static_image_mode=True,
                            max_num_objects=5,
                            min_detection_confidence=0,
                            model_name='Shoe') as objectron:
        image = image_file
        results = objectron.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        annotated_image = image.copy()
        for detected_object in results.detected_objects:
            mp_drawing.draw_landmarks(
                annotated_image, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
            mp_drawing.draw_axis(annotated_image, detected_object.rotation,
                                detected_object.translation)
    return annotated_image



def app():
    html_temp = """
        <body style="background-color:red;">
        <div style="background-color:Teal ;padding:10px">
        <h2 style="color:white;text-align:center;">3D Objectron</h2>
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
