import streamlit as st

def app():
    html_temp = """
        <body style="background-color:red;">
        <div style="background-color:Crimson ;padding:10px">
        <h2 style="color:white;text-align:center;">Advanced Computer Vision</h2>
        </div>
        </body>
        """
    st.markdown(html_temp, unsafe_allow_html=True)
    
    st.success("Developed by G Sai Krishna")
    if st.button("How To Use"):
        st.text("First,You must know about all the features in it")
        st.success("Visit GITHUB Repository")
    if st.button("Input format and Handling errors"):
        st.success("Input format must be JPG for accurate estimations")
        st.success("NoneType error was due to Improper image input or the algo is unable to find what you need")
        st.success("RGB error - this was because the algo is not framed for these inputs")
    if st.button("Architecture"):
        st.success("This was made by mediapipe,Opencv-Python,Streamlit and Heroku")
    if st.button("Github Repository"):
        link1 = '[GithubRepo](https://github.com/sai-krishna-ghanta/MP_web_cv)'
        st.markdown(link1, unsafe_allow_html=True)
    if st.button("About Developer"):
        link2 = '[Linkedin Profile](https://www.linkedin.com/in/ghanta-sai-krishna-320ab0211/)'
        st.markdown(link2, unsafe_allow_html=True)
