import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import pandas as pd


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="BloodMNIST Classifier",
    page_icon="🩸",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #666;
    margin-bottom: 30px;
}

.prediction-box {
    padding: 25px;
    border-radius: 15px;
    background-color: #f5f5f5;
    text-align: center;
    margin-top: 20px;
}

.prediction-label {
    font-size: 18px;
    color: #555;
}

.prediction-value {
    font-size: 32px;
    font-weight: 700;
    margin-top: 10px;
}

.confidence {
    font-size: 22px;
    font-weight: 600;
    margin-top: 10px;
}

.disclaimer {
    font-size: 13px;
    color: #777;
    text-align: center;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "Basophil",
    "Eosinophil",
    "Erythroblast",
    "Immature Granulocyte",
    "Lymphocyte",
    "Monocyte",
    "Neutrophil",
    "Platelet"
]


# ============================================================
# CNN MODEL
# ============================================================

class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 8)
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = CNN()

    model.load_state_dict(
        torch.load(
            "bloodmnist_cnn.pth",
            map_location=device
        )
    )

    model.to(device)
    model.eval()

    return model


model = load_model()


# ============================================================
# IMAGE TRANSFORMATION
# ============================================================

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_image(image):

    image = image.convert("RGB")

    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted_class = torch.max(
            probabilities,
            dim=1
        )

    predicted_index = predicted_class.item()

    predicted_name = CLASS_NAMES[predicted_index]

    confidence_value = confidence.item() * 100

    probabilities = probabilities[0].cpu().numpy()

    return (
        predicted_name,
        confidence_value,
        probabilities
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🩸 BloodMNIST Classifier</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'CNN-based blood cell image classification'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ About")

    st.write(
        "Upload a blood cell image and the trained "
        "CNN model will predict its class."
    )

    st.divider()

    st.subheader("Model")

    st.write("Architecture: CNN")
    st.write("Input: 28 × 28 RGB")
    st.write("Classes: 8")

    st.divider()

    st.subheader("Device")

    st.write(str(device))


# ============================================================
# FILE UPLOADER
# ============================================================

uploaded_file = st.file_uploader(
    "📤 Upload a blood cell image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# APPLICATION
# ============================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    with col1:

        st.subheader("🖼️ Uploaded Image")

        st.image(
            image,
            caption="Uploaded blood cell image",
            use_container_width=True
        )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    with col2:

        st.subheader("🤖 Prediction")

        if st.button(
            "🔍 Classify Image",
            use_container_width=True
        ):

            with st.spinner("Analyzing image..."):

                predicted_name, confidence, probabilities = (
                    predict_image(image)
                )

            st.markdown(
                f"""
                <div class="prediction-box">
                    <div class="prediction-label">
                        Predicted Class
                    </div>

                    <div class="prediction-value">
                        {predicted_name}
                    </div>

                    <div class="confidence">
                        Confidence: {confidence:.2f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # ------------------------------------------------
            # PROBABILITY TABLE
            # ------------------------------------------------

            st.subheader("📊 Class Probabilities")

            probability_data = pd.DataFrame({
                "Cell Type": CLASS_NAMES,
                "Probability (%)": [
                    round(float(p) * 100, 2)
                    for p in probabilities
                ]
            })

            probability_data = probability_data.sort_values(
                "Probability (%)",
                ascending=False
            )

            st.dataframe(
                probability_data,
                use_container_width=True,
                hide_index=True
            )

            # ------------------------------------------------
            # CHART
            # ------------------------------------------------

            st.subheader("📈 Probability Distribution")

            chart_data = probability_data.set_index(
                "Cell Type"
            )

            st.bar_chart(
                chart_data["Probability (%)"]
            )

else:

    st.info(
        "👆 Upload a JPG, JPEG, or PNG blood cell image "
        "to get a prediction."
    )


# ============================================================
# DISCLAIMER
# ============================================================

st.markdown(
    """
    <div class="disclaimer">
        ⚠️ This application is for educational and research
        purposes only. It is not intended for medical diagnosis
        or clinical decision-making.
    </div>
    """,
    unsafe_allow_html=True
)