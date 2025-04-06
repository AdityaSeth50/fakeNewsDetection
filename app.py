import streamlit as st
import joblib
import base64

# Load the model and vectorizer
vectorizer = joblib.load("vectorizer.jb")
model = joblib.load("lr_model.jb")

# Set page configuration
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# Custom background function
def set_background(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set background image (change 'news_bg.jpg' to your own image path)
set_background("news_bg.jpg")

# Custom CSS for better UI
st.markdown(
    """
    <style>
    body {
        font-family: 'Arial', sans-serif;
        color: #333;
    }
    .main-title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: #fff;
        text-shadow: 2px 2px 4px #000;
    }
    .subtitle {
        text-align: center;
        font-size: 20px;
        color: #ddd;
    }
    .stTextArea textarea {
        background: white !important; /* White */
        border: none !important;
        color: black !important;
        font-size: 16px !important;
        padding: 10px !important;
        border-radius: 10px !important;
    }
    .stButton>button {
        background: #ff4b4b;
        color: white;
        font-size: 18px;
        padding: 10px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and description
st.markdown("<h1 class='main-title'>📰 Fake News Detector</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Enter a news article below to verify its authenticity.</p>", unsafe_allow_html=True)

# Text input for news article
inputn = st.text_area("News Article:", "", height=200)

# Button to check news authenticity
if st.button("Check News 🕵️"):
    if inputn.strip():
        transform_input = vectorizer.transform([inputn])
        prediction = model.predict(transform_input)

        if prediction[0] == 1:
            st.success("✅ The News is Real!")
        else:
            st.error("🚨 The News is Fake!")
    else:
        st.warning("⚠️ Please enter some text to analyze.")
