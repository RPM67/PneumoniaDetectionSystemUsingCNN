import glob
import json
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import plotly.figure_factory as ff

# Set page configuration
st.set_page_config(
    page_title="Pneumonia Detection System",
    page_icon="🫁",
    layout="wide"
)


# Load the trained model
@st.cache_resource
def load_trained_model():
    model = load_model('pneumonia_model.h5')
    return model


# Image preprocessing function
def preprocess_image(image, img_dims):
    image = np.array(image)
    image = cv2.resize(image, (img_dims, img_dims))
    if len(image.shape) == 2:
        image = np.dstack([image, image, image])
    image = image.astype('float32') / 255
    image = np.expand_dims(image, axis=0)
    return image


# Function to load and count images in dataset
def get_dataset_info(base_path):
    datasets = ['train', 'val', 'test']
    counts = {}

    for dataset in datasets:
        normal_path = os.path.join(base_path, dataset, 'NORMAL')
        pneumonia_path = os.path.join(base_path, dataset, 'PNEUMONIA')

        normal_count = len(os.listdir(normal_path))
        pneumonia_count = len(os.listdir(pneumonia_path))

        counts[dataset] = {
            'NORMAL': normal_count,
            'PNEUMONIA': pneumonia_count
        }
    print(counts)
    return counts

def get_dataset_stats(base_dir):
    """Gather dataset statistics and sample image paths from train, test, and val folders."""
    data = {}
    sample_paths = {}
    subsets = ["train", "test", "val"]
    for subset in subsets:
        subset_dir = os.path.join(base_dir, subset)
        normal_dir = os.path.join(subset_dir, "NORMAL")
        pneumonia_dir = os.path.join(subset_dir, "PNEUMONIA")
        normal_images = glob.glob(os.path.join(normal_dir, "*.jpeg")) + glob.glob(os.path.join(normal_dir, "*.jpg"))
        pneumonia_images = glob.glob(os.path.join(pneumonia_dir, "*.jpeg")) + glob.glob(
            os.path.join(pneumonia_dir, "*.jpg"))
        data[subset] = {"Normal": len(normal_images), "Pneumonia": len(pneumonia_images)}
        sample_paths[subset] = {
            "Normal": normal_images[0] if normal_images else None,
            "Pneumonia": pneumonia_images[0] if pneumonia_images else None,
        }
    return data, sample_paths

def plot_dataset_distribution(data):
    """Create a Plotly bar chart showing image distribution by dataset subset and category."""
    records = []
    for subset, counts in data.items():
        records.append({"Subset": subset.capitalize(), "Category": "Normal", "Count": counts["Normal"]})
        records.append({"Subset": subset.capitalize(), "Category": "Pneumonia", "Count": counts["Pneumonia"]})
    df = pd.DataFrame(records)
    fig = px.bar(df, x="Subset", y="Count", color="Category", barmode="group",
                 title="Image Distribution by Dataset Subset")
    return fig


# Calculate metrics function
def calculate_metrics(model, test_data, test_labels):
    predictions = model.predict(test_data)
    predictions = (predictions > 0.5).astype(int)

    cm = confusion_matrix(test_labels, predictions)
    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions)
    recall = recall_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions)

    return cm, accuracy, precision, recall, f1

def plot_confusion_matrix():
    """Generate a Plotly heatmap for the confusion matrix using provided values."""
    # Provided confusion matrix values:
    cm_values = [[182, 52],
                 [4, 386]]
    # Define labels for axes
    x_labels = ["Predicted Normal", "Predicted Pneumonia"]
    y_labels = ["Actual Normal", "Actual Pneumonia"]

    # Create the heatmap with Plotly
    fig = px.imshow(cm_values,
                    text_auto=True,
                    color_continuous_scale="Blues",
                    x=x_labels,
                    y=y_labels)
    fig.update_layout(xaxis_title="Predicted Label", yaxis_title="Actual Label")
    return fig


def get_classification_report():
    """Generate a classification report string from the provided confusion matrix."""
    # Provided confusion matrix values:
    # [[182, 52],
    #  [4, 386]]
    # For class "Normal":
    TP_normal = 182
    FN_normal = 52
    FP_normal = 4
    support_normal = TP_normal + FN_normal  # 234
    precision_normal = TP_normal / (TP_normal + FP_normal)
    recall_normal = TP_normal / (TP_normal + FN_normal)
    f1_normal = 2 * (precision_normal * recall_normal) / (precision_normal + recall_normal)

    # For class "Pneumonia":
    TP_pneu = 386
    FN_pneu = 4
    FP_pneu = 52
    support_pneu = TP_pneu + FN_pneu  # 390
    precision_pneu = TP_pneu / (TP_pneu + FP_pneu)
    recall_pneu = TP_pneu / (TP_pneu + FN_pneu)
    f1_pneu = 2 * (precision_pneu * recall_pneu) / (precision_pneu + recall_pneu)

    total_support = support_normal + support_pneu
    accuracy = (TP_normal + TP_pneu) / total_support

    # Macro averages
    macro_precision = (precision_normal + precision_pneu) / 2
    macro_recall = (recall_normal + recall_pneu) / 2
    macro_f1 = (f1_normal + f1_pneu) / 2

    # Weighted averages
    weighted_precision = (support_normal * precision_normal + support_pneu * precision_pneu) / total_support
    weighted_recall = (support_normal * recall_normal + support_pneu * recall_pneu) / total_support
    weighted_f1 = (support_normal * f1_normal + support_pneu * f1_pneu) / total_support

    report = f"""Classification Report:
              precision    recall  f1-score   support

Normal          {precision_normal:.2f}      {recall_normal:.2f}      {f1_normal:.2f}       {support_normal}
Pneumonia       {precision_pneu:.2f}      {recall_pneu:.2f}      {f1_pneu:.2f}       {support_pneu}

Accuracy                           {accuracy:.2f}       {total_support}
Macro Avg       {macro_precision:.2f}      {macro_recall:.2f}      {macro_f1:.2f}       {total_support}
Weighted Avg    {weighted_precision:.2f}      {weighted_recall:.2f}      {weighted_f1:.2f}       {total_support}
"""
    return report


# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["About Dataset", "About Model", "Test Model"])

# Load model
model = load_trained_model()

if page == "About Dataset":
    st.title("Dataset Information")
    st.write("""
    ## Chest X-Ray Dataset for Pneumonia Detection
    
    This dataset contains 5,863 chest X-ray images for pneumonia detection, organized into three splits:
    - **Training Set**
    - **Validation Set**
    - **Test Set**
    
    Each folder contains two subfolders:
    - **Normal**
    - **Pneumonia**
    
    **Below are visual representations of the dataset** :-
    """)

    base_path = '/home/rpm/Desktop/Set Project/chest_xray'  # Update with your dataset path
    data_stats, sample_images = get_dataset_stats(base_path)

    # Display statistics and sample images for each subset
    for subset in ["train", "test", "val"]:
        st.subheader(f"{subset.capitalize()} Set")
        st.write(f"**Normal:** {data_stats[subset]['Normal']} images")
        st.write(f"**Pneumonia:** {data_stats[subset]['Pneumonia']} images")
        cols = st.columns(2)
        if sample_images[subset]["Normal"]:
            img = Image.open(sample_images[subset]["Normal"])
            cols[0].image(img, caption="Normal Sample", use_container_width=True)
        else:
            cols[0].write("No sample image found for Normal.")
        if sample_images[subset]["Pneumonia"]:
            img = Image.open(sample_images[subset]["Pneumonia"])
            cols[1].image(img, caption="Pneumonia Sample", use_container_width=True)
        else:
            cols[1].write("No sample image found for Pneumonia.")

    # Dataset statistics
    # Plot distribution using plotly
    st.subheader("Dataset Distribution")
    fig = plot_dataset_distribution(data_stats)
    st.plotly_chart(fig, use_container_width=True)

    st.write("""
    **note**:- 
    The chest X-ray images were selected from retrospective cohorts of pediatric patients 
    aged one to five years old from Guangzhou Women and Children's Medical Center.
    
    """)

elif page == "About Model":
    st.title("Model Information")

    st.markdown("---")

    # 1. Image Preprocessing
    st.write("## 1️⃣ Image Preprocessing")
    st.markdown("""
    - All images resized to **150×150 pixels**  
      ➤ Ensures uniform input size for the CNN

    - Converted to **grayscale**  
      ➤ Reduces complexity, as X-rays don't need color channels

    - **Normalized** (pixel values divided by 255)  
      ➤ Speeds up training and improves convergence

    - **Label encoded**: NORMAL → 0, PNEUMONIA → 1  
      ➤ Converts categorical labels into machine-readable format

    - **Data shuffled**  
      ➤ Prevents learning bias from class ordering

    - **Reshaped** to (150, 150, 1)  
      ➤ Matches CNN input format (height, width, channel)

    - **Augmentation** applied (rotation, zoom, flip)  
      ➤ Increases dataset diversity, helps avoid overfitting

    These preprocessing steps are essential to ensure that the model trains efficiently and accurately on clean, standardized input data.
    """)

    st.markdown("---")

    # 2. Model Architecture
    st.write("## 2️⃣ Model Architecture")
    st.subheader("Model Architecture Overview")
    st.markdown("""
    The pneumonia detection model is a **deep CNN with over 20 layers**, designed to progressively learn features from chest X-ray images — from simple edges to complex lung patterns. It uses **Separable Convolutions** for efficiency and **Dropout** for regularization, making it suitable for medical image classification with limited data.

    ### Why this Architecture?
    - Deep layers capture both low-level and high-level patterns
    - **SeparableConv2D** reduces computational cost
    - **BatchNormalization** stabilizes training
    - **Dropout** prevents overfitting
    - Designed to generalize well on unseen X-ray data

    ### Layers Summary
    - **Input Layer:** (150×150×3)  
      ➤ Accepts resized chest X-ray images

    - **Conv2D + MaxPooling:**  
      ➤ Basic feature extraction (edges, shapes)

    - **SeparableConv2D (×4 blocks):**  
      ➤ Efficient deep pattern learning

    - **BatchNormalization:**  
      ➤ Faster and more stable training

    - **Dropout Layers (0.2, 0.5, 0.7):**  
      ➤ Prevent overfitting by randomly turning off neurons

    - **Flatten:**  
      ➤ Converts feature maps to 1D vector

    - **Dense Layers (512 → 128 → 64):**  
      ➤ Learn abstract representations for classification

    - **Output Layer (Sigmoid):**  
      ➤ Predicts probability of pneumonia (binary classification)
    """)

    st.markdown("---")

    # 3. Training Configuration
    st.write("## 3️⃣ Training Configuration")

    st.subheader("Model Training Configuration")
    st.markdown("""
    - **Batch size:** 32  
      ➤ Number of samples processed before model update

    - **Epochs:** 10  
      ➤ One full pass over the entire training dataset

    - **Optimizer:** Adam  
      ➤ Adaptive learning optimizer for faster convergence

    - **Loss function:** Binary Crossentropy  
      ➤ Suitable for binary classification (NORMAL vs. PNEUMONIA)

    - **Learning rate:** 0.001 (default in Adam)  
      ➤ Controls step size during optimization

    - **Validation split:** 20%  
      ➤ Helps evaluate performance on unseen data during training
    """)

    # 4. Training History
    st.subheader("Training History")
    with open("training_history.json", "r") as file:
        data = json.load(file)

    with st.expander("📊 Training Progress (All Epochs)"):
        for epoch in range(len(data["accuracy"])):
            st.write(f"### Epoch {epoch + 1} Metrics")
            st.write(f"**Accuracy:** {data['accuracy'][epoch] * 100:.2f}%")
            st.write(f"**Loss:** {data['loss'][epoch]:.4f}")
            st.write(f"**Validation Accuracy:** {data['val_accuracy'][epoch] * 100:.2f}%")
            st.write(f"**Validation Loss:** {data['val_loss'][epoch]:.4f}")
            st.write(f"**Learning Rate:** {data['learning_rate'][epoch]:.8f}")
            st.markdown("---")

    # Plot Training Graphs
    st.markdown("#### 📈 Accuracy & Loss Over Epochs")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].plot(data['accuracy'], label="Train Accuracy", marker='o')
    ax[0].plot(data['val_accuracy'], label="Val Accuracy", marker='o')
    ax[0].set_title("Accuracy")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Accuracy")
    ax[0].legend()
    ax[0].grid(True)

    ax[1].plot(data['loss'], label="Train Loss", marker='o')
    ax[1].plot(data['val_loss'], label="Val Loss", marker='o')
    ax[1].set_title("Loss")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Loss")
    ax[1].legend()
    ax[1].grid(True)

    st.pyplot(fig)

    # Final metrics
    st.write("#### Final Training Metrics")
    train_acc = data['accuracy'][-1]
    val_acc = data['val_accuracy'][-1]
    train_loss = data['loss'][-1]
    val_loss = data['val_loss'][-1]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Train Accuracy", f"{train_acc * 100:.2f}%")
    col2.metric("Val Accuracy", f"{val_acc * 100:.2f}%")
    col3.metric("Train Loss", f"{train_loss:.4f}")
    col4.metric("Val Loss", f"{val_loss:.4f}")

    st.markdown("---")

    # 5. Test Performance
    st.write("## 4️⃣ Test Performance")

    # Displaying Test Metrics
    st.subheader("Test Metrics")
    col1, col2, col3, col4 = st.columns(4)
    cm = np.array([[182, 52],
                   [4, 386]])

    accuracy = (cm[0, 0] + cm[1, 1]) / np.sum(cm) * 100
    precision = cm[1, 1] / (cm[1, 1] + cm[0, 1]) * 100
    recall = cm[1, 1] / (cm[1, 1] + cm[1, 0]) * 100
    f1 = 2 * (precision * recall) / (precision + recall)

    col1.metric("Accuracy", f"{accuracy:.2f}%")
    col2.metric("Precision", f"{precision:.2f}%")
    col3.metric("Recall", f"{recall:.2f}%")
    col4.metric("F1-Score", f"{f1:.2f}")

    st.markdown("""
    - **Accuracy**: Overall correct predictions.
    - **Precision**: How many predicted positives were actual positives.
    - **Recall**: How well model finds actual pneumonia cases.
    - **F1-Score**: Balance between precision & recall.
    """)

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm_fig = plot_confusion_matrix()
    st.plotly_chart(cm_fig, use_container_width=True)

    # Classification Report
    st.subheader("Classification Report")
    report_path = "classification_report.txt"
    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            report = f.read()
    else:
        report = get_classification_report()
    st.text(report)




else:  # Test Model
    st.title("Test Pneumonia Detection Model")
    st.write("""
    Upload a chest X-ray image to test the model's prediction.
    The image should be in JPEG/JPG/PNG format.
    """)

    uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded X-ray image.', use_container_width=True)

        # Make prediction
        processed_image = preprocess_image(image, 150)
        prediction = model.predict(processed_image)

        # Display prediction
        st.write("## Prediction")
        probability = prediction[0][0] * 100

        if probability > 50:
            st.error(f"Prediction: PNEUMONIA (Confidence: {probability:.2f}%)")
        else:
            st.success(f"Prediction: NORMAL (Confidence: {100 - probability:.2f}%)")

        # Visualization of prediction probability
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability,
            title={'text': "Probability of Pneumonia"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgreen"},
                    {'range': [50, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        st.plotly_chart(fig)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <style>
    .sidebar-text {
        line-height: 1.2; /* Adjust line height as needed */
    }
    </style>
    <p class="sidebar-text">
    Developed by:<br>
    Anupam<br>
    Akarsh<br>
    Rahul
    </p>
    <p>
    Guided by:<br>
    Mythili N
    </p>
    """,
    unsafe_allow_html=True
)
