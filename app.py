import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from PIL import Image

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Fake vs Real Face Detector",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #f8f9ff 0%, #eef2ff 100%);
    }

    /* Hide default Streamlit menu/footer */
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

    .subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 17px;
        margin-bottom: 35px;
    }

    /* Header badge */
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
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.07);
        margin-bottom: 20px;
        border: 1px solid #e5e7eb;
    }

    .card-title {
        font-size: 20px;
        font-weight: 700;
        color: #111827;
        margin-bottom: 8px;
    }

    .card-description {
        color: #6b7280;
        font-size: 14px;
        margin-bottom: 15px;
    }

    /* Result box */
    .result-card {
        background: white;
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
        margin-top: 25px;
        border: 1px solid #e5e7eb;
    }

    .result-title {
        font-size: 16px;
        color: #6b7280;
        margin-bottom: 8px;
    }

    .result-real {
        font-size: 38px;
        font-weight: 800;
        color: #16a34a;
    }

    .result-fake {
        font-size: 38px;
        font-weight: 800;
        color: #dc2626;
    }

    .confidence {
        font-size: 22px;
        font-weight: 700;
        color: #374151;
        margin-top: 8px;
    }

    /* Info boxes */
    .model-info {
        background: #eef2ff;
        padding: 15px;
        border-radius: 12px;
        color: #3730a3;
        text-align: center;
        font-weight: 600;
        margin-top: 15px;
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

    /* Upload area */
    [data-testid="stFileUploader"] {
        background: #f9fafb;
        border-radius: 15px;
        padding: 10px;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
    }

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------

@st.cache_resource
def load_cnn_model():
    return load_model("cnn_model.h5")


@st.cache_resource
def load_efficientnet_model():
    return load_model("efficientnet_model.h5")


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.markdown(
    '<div style="text-align:center;"><span class="badge">AI IMAGE ANALYSIS</span></div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-title">🔍 Fake vs Real Face Detector</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Detect whether a face image is real or AI-generated using deep learning</div>',
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# MODEL SELECTION
# ---------------------------------------------------------

st.markdown("""
<div class="card">
    <div class="card-title">🧠 Select Detection Model</div>
    <div class="card-description">
        Choose the deep learning model you want to use for image analysis.
    </div>
</div>
""", unsafe_allow_html=True)

model_choice = st.selectbox(
    "Model",
    [
        "CNN",
        "EfficientNetB0"
    ],
    label_visibility="collapsed"
)


# ---------------------------------------------------------
# LOAD SELECTED MODEL
# ---------------------------------------------------------

if model_choice == "CNN":

    model = load_cnn_model()
    image_size = (128, 128)

else:

    model = load_efficientnet_model()
    image_size = (224, 224)


# ---------------------------------------------------------
# IMAGE UPLOAD
# ---------------------------------------------------------

st.markdown("""
<div class="card">
    <div class="card-title">📤 Upload Face Image</div>
    <div class="card-description">
        Upload a JPG, JPEG, or PNG image for analysis.
    </div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------

if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")

    # Two-column layout
    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            '<div class="card-title">🖼️ Uploaded Image</div>',
            unsafe_allow_html=True
        )

        st.image(
            img,
            use_container_width=True
        )

    with col2:

        st.markdown(
            '<div class="card-title">📋 Image Information</div>',
            unsafe_allow_html=True
        )

        st.write(f"**File:** {uploaded_file.name}")
        st.write(f"**Format:** {img.format if img.format else 'Image'}")
        st.write(f"**Original Size:** {img.size[0]} × {img.size[1]}")
        st.write(f"**Detection Model:** {model_choice}")

        st.markdown(
            f"""
            <div class="model-info">
                🤖 {model_choice} Model<br>
                Input Size: {image_size[0]} × {image_size[1]}
            </div>
            """,
            unsafe_allow_html=True
        )


    # -----------------------------------------------------
    # IMAGE PREPROCESSING
    # -----------------------------------------------------

    processed_img = img.resize(image_size)

    img_array = np.array(processed_img)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )


    if model_choice == "CNN":

        img_array = img_array / 255.0

    else:

        img_array = efficientnet_preprocess(img_array)


    # -----------------------------------------------------
    # MODEL PREDICTION
    # -----------------------------------------------------

    prediction = model.predict(
        img_array,
        verbose=0
    )

    confidence = prediction[0][0]


    # -----------------------------------------------------
    # RESULT
    # -----------------------------------------------------

    if confidence > 0.5:

        result = "Real"
        conf = confidence * 100

    else:

        result = "Fake"
        conf = (1 - confidence) * 100


    # -----------------------------------------------------
    # RESULT DISPLAY
    # -----------------------------------------------------

    result_class = (
        "result-real"
        if result == "Real"
        else "result-fake"
    )

    result_icon = (
        "✅"
        if result == "Real"
        else "⚠️"
    )

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


    # -----------------------------------------------------
    # CONFIDENCE
    # -----------------------------------------------------

    st.markdown("### 📊 Confidence Score")

    st.progress(
        min(int(conf), 100)
    )

    if result == "Real":

        st.success(
            f"The model classified this image as **Real** with {conf:.2f}% confidence."
        )

    else:

        st.error(
            f"The model classified this image as **Fake** with {conf:.2f}% confidence."
        )


    # -----------------------------------------------------
    # MODEL INFORMATION
    # -----------------------------------------------------

    st.markdown("### 🤖 Model Information")

    if model_choice == "CNN":

        st.info(
            "CNN (Convolutional Neural Network) analyzes visual patterns "
            "and features in the input face image."
        )

    else:

        st.info(
            "EfficientNetB0 is a lightweight deep learning architecture "
            "designed to achieve strong image classification performance "
            "with efficient computation."
        )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.markdown(
    """
    <div class="footer">
        Fake vs Real Face Image Detector<br>
        Powered by Deep Learning • CNN • EfficientNetB0
    </div>
    """,
    unsafe_allow_html=True
)
