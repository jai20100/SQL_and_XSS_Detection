import cv2
import mediapipe as mp
import streamlit as st
from PIL import Image
import numpy as np

# streamlit run face_detection.py
mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation


def main(image_file):
    with mp_selfie_segmentation.SelfieSegmentation(
            model_selection=0) as selfie_segmentation:
            image = image_file
            BG_COLOR = (192, 192, 192) # gray
            MASK_COLOR = (255, 255, 255) # white
            image_height, image_width, _ = image.shape
    # Convert the BGR image to RGB before processing.
            results = selfie_segmentation.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Draw selfie segmentation on the background image.
            # To improve segmentation around boundaries, consider applying a joint
            # bilateral filter to "results.segmentation_mask" with "image".
            condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
            # Generate solid color images for showing the output selfie segmentation mask.
            fg_image = np.zeros(image.shape, dtype=np.uint8)
            fg_image[:] = MASK_COLOR
            bg_image = np.zeros(image.shape, dtype=np.uint8)
            bg_image[:] = BG_COLOR
            output_image = np.where(condition, fg_image, bg_image)
        
    return output_image

def app():
    html_temp = """
        <body style="background-color:red;">
        <div style="background-color:Teal ;padding:10px">
        <h2 style="color:white;text-align:center;">Selfie Segementation</h2>
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
