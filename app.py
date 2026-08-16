import streamlit as st
import numpy as np

from PIL import Image

from tensorflow.keras.models import load_model

from tensorflow.keras.applications.efficientnet import (
    preprocess_input as efficientnet_preprocess
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Fake vs Real Face Detector",
    page_icon="🔍",
    layout="centered"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main page */

    .stApp {
        background: linear-gradient(
            135deg,
            #f8f9ff 0%,
            #eef1ff 100%
        );
    }


    /* Main title */

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 800;
        margin-top: 20px;
        margin-bottom: 5px;
        color: #262a3b;
    }


    /* Subtitle */

    .subtitle {
        text-align: center;
        font-size: 17px;
        color: #6b7280;
        margin-bottom: 35px;
    }


    /* Cards */

    .card {
        background: white;
        padding: 25px;
        border-radius: 18px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.06);
        margin-bottom: 20px;
    }


    /* Section title */

    .section-title {
        font-size: 20px;
        font-weight: 700;
        color: #303442;
        margin-bottom: 8px;
    }


    /* Description */

    .description {
        font-size: 15px;
        color: #6b7280;
        line-height: 1.6;
    }


    /* Result */

    .result-box {
        background: white;
        padding: 25px;
        border-radius: 18px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.06);
        text-align: center;
        margin-top: 20px;
    }


    /* Footer */

    .footer {
        text-align: center;
        color: #8a8f9e;
        font-size: 13px;
        margin-top: 35px;
        margin-bottom: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🔍 Fake vs Real Face Image Detector</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Detect whether a face image is real or AI-generated using deep learning'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_cnn_model():

    return load_model(
        "cnn_model.h5"
    )


@st.cache_resource
def load_efficientnet_model():

    return load_model(
        "efficientnet_model.h5"
    )


# ============================================================
# MODEL SELECTION
# ============================================================

st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-title">🧠 Select Detection Model</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="description">'
    'Choose the deep learning model you want to use for image analysis.'
    '</div>',
    unsafe_allow_html=True
)

model_choice = st.selectbox(
    "Select Model",
    [
        "CNN",
        "EfficientNetB0"
    ],
    label_visibility="collapsed"
)

st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-title">📤 Upload Face Image</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="description">'
    'Upload a JPG, JPEG, or PNG image for analysis.'
    '</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload an Image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ],
    label_visibility="collapsed"
)

st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:

    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    img = Image.open(
        uploaded_file
    ).convert("RGB")


    # --------------------------------------------------------
    # DISPLAY IMAGE
    # --------------------------------------------------------

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">🖼️ Uploaded Image</div>',
        unsafe_allow_html=True
    )

    # Compatible with older Streamlit versions
    st.image(
        img,
        caption="Uploaded Image",
        width=500
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # SELECT MODEL
    # --------------------------------------------------------

    if model_choice == "CNN":

        model = load_cnn_model()

        image_size = (
            128,
            128
        )

    else:

        model = load_efficientnet_model()

        image_size = (
            224,
            224
        )


    # --------------------------------------------------------
    # RESIZE IMAGE
    # --------------------------------------------------------

    resized_img = img.resize(
        image_size
    )


    # --------------------------------------------------------
    # CONVERT IMAGE TO NUMPY ARRAY
    # --------------------------------------------------------

    img_array = np.array(
        resized_img
    )


    # --------------------------------------------------------
    # ADD BATCH DIMENSION
    # --------------------------------------------------------

    img_array = np.expand_dims(
        img_array,
        axis=0
    )


    # --------------------------------------------------------
    # PREPROCESS IMAGE
    # --------------------------------------------------------

    if model_choice == "CNN":

        img_array = (
            img_array / 255.0
        )

    else:

        img_array = efficientnet_preprocess(
            img_array
        )


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    prediction = model.predict(
        img_array,
        verbose=0
    )


    # --------------------------------------------------------
    # GET CONFIDENCE
    # --------------------------------------------------------

    confidence = float(
        prediction[0][0]
    )


    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------

    if confidence > 0.5:

        result = "Real"

        conf = confidence * 100

    else:

        result = "Fake"

        conf = (
            1 - confidence
        ) * 100


    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    st.markdown(
        '<div class="result-box">',
        unsafe_allow_html=True
    )


    if result == "Real":

        st.success(
            f"Prediction: {result}"
        )

    else:

        st.error(
            f"Prediction: {result}"
        )


    st.metric(
        "Confidence",
        f"{conf:.2f}%"
    )


    st.info(
        f"Model Used: {model_choice}"
    )


    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="footer">'
    'Fake vs Real Face Image Detector • Deep Learning Project'
    '</div>',
    unsafe_allow_html=True
)
