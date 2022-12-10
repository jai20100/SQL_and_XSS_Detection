import streamlit as st
from multiapp import MultiApp 
from pages import face_detection_image,facemesh,hands,hostilic,objectron,pose,selfie_segmentation,home
import warnings
warnings.filterwarnings("ignore")
st.markdown("Tool Bar")
app = MultiApp()
app.add_app("Home",home.app)
app.add_app("Face detection", face_detection_image.app)
app.add_app("Face Mesh", facemesh.app)
app.add_app("Hands", hands.app)
app.add_app("Hostilic", hostilic.app)
app.add_app("Pose ", pose.app)
app.add_app("selfie_segmentation", selfie_segmentation.app)
app.add_app("3D Objectron",objectron.app)
app.run()
