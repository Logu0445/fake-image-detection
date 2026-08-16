```python
# ============================================================
# FAKE VS REAL FACE IMAGE DETECTOR
# CNN + EfficientNetB0 + ResNet50
# ============================================================

import streamlit as st
import numpy as np

from tensorflow.keras.models import load_model

from tensorflow.keras.applications.efficientnet import (
    preprocess_input as efficientnet_preprocess
)

from tensorflow.keras.applications.resnet50 import (
    preprocess_input as resnet_preprocess
)

from PIL import Image


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Fake vs Real Face Detector",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main background */

    .stApp {
        background: linear-gradient(
            135deg,
            #f8f9ff 0%,
            #eef2ff 100%
        );
    }


    /* Hide Streamlit default elements */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }


    /* Main container */

    .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* Main title */

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 800;
        color: #1f2937;
        margin-bottom: 5px;
    }


    /* Subtitle */

    .subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 17px;
        margin-bottom: 35px;
    }


    /* Badge */

    .badge {
        display: inline-block;
        background: #e0e7ff;
        color: #4338ca;
        padding: 7px 16px;
        border-radius: 20px;
        font-size: 14px;
        font-weight: 600;
        margin-bottom: 15px;
    }


    /* Cards */

    .card {
        background: white;
        padding: 25px;
        border-radius: 18px;
        box-shadow:
            0 8px 25px rgba(0, 0, 0, 0.07);
        margin-bottom: 20px;
        border: 1px solid #e5e7eb;
    }


    /* Card title */

    .card-title {
        font-size: 20px;
        font-weight: 700;
        color: #111827;
        margin-bottom: 8px;
    }


    /* Card description */

    .card-description {
        color: #6b7280;
        font-size: 14px;
        margin-bottom: 15px;
    }


    /* Result card */

    .result-card {
        background: white;
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        box-shadow:
            0 10px 30px rgba(0, 0, 0, 0.08);
        margin-top: 25px;
        border: 1px solid #e5e7eb;
    }


    /* Result title */

    .result-title {
        font-size: 16px;
        color: #6b7280;
        margin-bottom: 8px;
    }


    /* Real result */

    .result-real {
        font-size: 38px;
        font-weight: 800;
        color: #16a34a;
    }


    /* Fake result */

    .result-fake {
        font-size: 38px;
        font-weight: 800;
        color: #dc2626;
    }


    /* Confidence */

    .confidence {
        font-size: 22px;
        font-weight: 700;
        color: #374151;
        margin-top: 8px;
    }


    /* Model information */

    .model-info {
        background: #eef2ff;
        padding: 15px;
        border-radius: 12px;
        color: #3730a3;
        text-align: center;
        font-weight: 600;
        margin-top: 15px;
    }


    /* Upload area */

    [data-testid="stFileUploader"] {
        background: #f9fafb;
        border-radius: 15px;
        padding: 10px;
    }


    /* Footer */

    .footer {
        text-align: center;
        color: #9ca3af;
        font-size: 13px;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid #e5e7eb;
    }


    /* Buttons */

    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
    }

    </style>
    """,
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


@st.cache_resource
def load_resnet_model():

    return load_model(
        "resnet50_model.h5"
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div style="text-align:center;">
        <span class="badge">
            AI IMAGE ANALYSIS
        </span>
    </div>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """
    <div class="main-title">
        🔍 Fake vs Real Face Detector
    </div>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """
    <div class="subtitle">
        Detect whether a face image is real or AI-generated
        using deep learning
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MODEL SELECTION
# ============================================================

st.markdown(
    """
    <div class="card">

        <div class="card-title">
            🧠 Select Detection Model
        </div>

        <div class="card-description">
            Choose the deep learning model you want
            to use for image analysis.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


model_choice = st.selectbox(
    "Model",
    [
        "CNN",
        "EfficientNetB0",
        "ResNet50"
    ],
    label_visibility="collapsed"
)


# ============================================================
# LOAD SELECTED MODEL
# ============================================================

if model_choice == "CNN":

    model = load_cnn_model()

    image_size = (
        128,
        128
    )


elif model_choice == "EfficientNetB0":

    model = load_efficientnet_model()

    image_size = (
        224,
        224
    )


else:

    model = load_resnet_model()

    image_size = (
        224,
        224
    )


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.markdown(
    """
    <div class="card">

        <div class="card-title">
            📤 Upload Face Image
        </div>

        <div class="card-description">
            Upload a JPG, JPEG, or PNG image for analysis.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


uploaded_file = st.file_uploader(
    "Upload Image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ],
    label_visibility="collapsed"
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
    # DISPLAY IMAGE + INFORMATION
    # --------------------------------------------------------

    col1, col2 = st.columns(2)


    # --------------------------------------------------------
    # IMAGE COLUMN
    # --------------------------------------------------------

    with col1:

        st.markdown(
            """
            <div class="card-title">
                🖼️ Uploaded Image
            </div>
            """,
            unsafe_allow_html=True
        )

        # width=350 is used for compatibility
        # with your current Streamlit version.

        st.image(
            img,
            width=350
        )


    # --------------------------------------------------------
    # INFORMATION COLUMN
    # --------------------------------------------------------

    with col2:

        st.markdown(
            """
            <div class="card-title">
                📋 Image Information
            </div>
            """,
            unsafe_allow_html=True
        )


        st.write(
            f"**File:** {uploaded_file.name}"
        )


        st.write(
            f"**Format:** "
            f"{img.format if img.format else 'Image'}"
        )


        st.write(
            f"**Original Size:** "
            f"{img.size[0]} × {img.size[1]}"
        )


        st.write(
            f"**Detection Model:** "
            f"{model_choice}"
        )


        st.markdown(
            f"""
            <div class="model-info">

                🤖 {model_choice} Model

                <br>

                Input Size:
                {image_size[0]} × {image_size[1]}

            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # IMAGE PREPROCESSING
    # ========================================================

    processed_img = img.resize(
        image_size
    )


    img_array = np.array(
        processed_img
    )


    img_array = np.expand_dims(
        img_array,
        axis=0
    )


    # --------------------------------------------------------
    # CNN PREPROCESSING
    # --------------------------------------------------------

    if model_choice == "CNN":

        img_array = (
            img_array / 255.0
        )


    # --------------------------------------------------------
    # EFFICIENTNET PREPROCESSING
    # --------------------------------------------------------

    elif model_choice == "EfficientNetB0":

        img_array = efficientnet_preprocess(
            img_array
        )


    # --------------------------------------------------------
    # RESNET50 PREPROCESSING
    # --------------------------------------------------------

    else:

        img_array = resnet_preprocess(
            img_array
        )


    # ========================================================
    # MODEL PREDICTION
    # ========================================================

    prediction = model.predict(
        img_array,
        verbose=0
    )


    confidence = float(
        prediction[0][0]
    )


    # ========================================================
    # CLASSIFICATION
    # ========================================================

    if confidence > 0.5:

        result = "Real"

        conf = (
            confidence * 100
        )

    else:

        result = "Fake"

        conf = (
            (1 - confidence) * 100
        )


    # ========================================================
    # RESULT DESIGN
    # ========================================================

    if result == "Real":

        result_class = "result-real"

        result_icon = "✅"


    else:

        result_class = "result-fake"

        result_icon = "⚠️"


    # ========================================================
    # RESULT CARD
    # ========================================================

    st.markdown(
        f"""
        <div class="result-card">

            <div class="result-title">
                DETECTION RESULT
            </div>

            <div class="{result_class}">
                {result_icon} {result}
            </div>

            <div class="confidence">
                Confidence: {conf:.2f}%
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # CONFIDENCE SCORE
    # ========================================================

    st.markdown(
        "### 📊 Confidence Score"
    )


    st.progress(
        min(
            int(conf),
            100
        )
    )


    if result == "Real":

        st.success(
            f"The model classified this image as "
            f"**Real** with {conf:.2f}% confidence."
        )

    else:

        st.error(
            f"The model classified this image as "
            f"**Fake** with {conf:.2f}% confidence."
        )


    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    st.markdown(
        "### 🤖 Model Information"
    )


    if model_choice == "CNN":

        st.info(
            "CNN (Convolutional Neural Network) "
            "analyzes visual patterns and features "
            "in the input face image."
        )


    elif model_choice == "EfficientNetB0":

        st.info(
            "EfficientNetB0 is a lightweight deep "
            "learning architecture designed to provide "
            "strong image classification performance "
            "with efficient computation."
        )


    else:

        st.info(
            "ResNet50 is a deep residual neural network "
            "using residual connections to learn complex "
            "visual features from face images."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        Fake vs Real Face Image Detector

        <br>

        Powered by Deep Learning
        • CNN
        • EfficientNetB0
        • ResNet50

    </div>
    """,
    unsafe_allow_html=True
)
```
