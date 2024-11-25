import streamlit as st
from PIL import Image, ImageDraw
import pytesseract  
import pyttsx3
import os
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAI
import torch

# Set Tesseract OCR path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

# Initialize Google Generative AI with API Key
GEMINI_API_KEY = "Your_API_KEY"  # Replace with your valid API key
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

llm = GoogleGenerativeAI(model="gemini-1.5-pro", api_key=GEMINI_API_KEY)

# Initialize Text-to-Speech engine
engine = pyttsx3.init()

# Load the YOLO model (using YOLOv5 from PyTorch Hub)
model = torch.hub.load("ultralytics/yolov5:v6.2", "yolov5s")  # YOLOv5 small model

# Streamlit App Layout
st.markdown(
    """
    <style>
     .main-title {
        font-size: 48px;
        font-weight: bold;
        text-align: center;
        color: #0662f6;
        margin-top: -20px;
     }
    .subtitle {
        font-size: 18px;
        color: #555;
        text-align: center;
        margin-bottom: 20px;
    }
    .feature-header {
        font-size: 24px;
        color: #333;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

image_path = 'Inno_logo_.png'  # Replace with your actual PNG image file path

# Specify the desired width and height
st.image(image_path, width=200)

st.markdown('<div class="main-title"> 👁️‍🗨️Visionary AI 👁️‍🗨️</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Transforming Lives with AI: Real-Time Scene Understanding, Obstacle Detection, Text Reading, and Voice Guidance. </div>', unsafe_allow_html=True)

# Sidebar Features
st.sidebar.image(r"C:\Users\gorla\streamlit\VisionAi.jpg", 
                 width=250)

# Sidebar Overview
st.sidebar.title("ℹ️ Overview")
st.sidebar.markdown(
    """
    📌 **Features**
    - 🔍 **Scene Description**: Gain AI-generated insights about the image, including identified objects and actionable suggestions.
    - 🛑 **Object and Obstacle Detection**: Identify objects or obstacles within the image and highlight them for enhanced user safety.
    - 📝 **Text Extraction**: Extract readable text from uploaded images using advanced OCR technology.
    - 🔊 **Text-to-Speech**: Listen to the extracted text as it is read aloud for seamless accessibility.

    💡 **Why It Matters**:
    VisionAssist helps bridge accessibility gaps by combining scene interpretation, text recognition, and audio feedback.
    It simplifies everyday tasks for visually impaired users.
    
    🤖 **Powered by**:
    - **Google Gemini API** For scene understanding and AI-driven insights.
    - **YOLOv5** For object and obstacle detection.
    - **Tesseract OCR** For accurate text extraction from images.
    - **pyttsx3** For converting text to natural-sounding speech.
    """
)

# How to Use Section
st.sidebar.text_area(
    "📜 How to Use", 
    """1. Upload an Image: Begin by uploading a JPG, JPEG, or PNG file.
    2. Choose a Feature:
    - 🔍 Scene Description: Understand objects and elements in the image.
    - 🛑 Object Detection: See highlighted obstacles in the image.
    - 📝 Text Extraction: Recognize and extract text visible in the image.
    - 🔊 Listen to Text: Hear the extracted text spoken aloud.""",
    height=250  # Set height as an integer for the text area
)

# Functions for functionality
def extract_text_from_image(image):
    """Extracts text from the given image using OCR."""
    return pytesseract.image_to_string(image)

def text_to_speech(text):
    """Converts the given text to speech."""
    engine.say(text)
    engine.runAndWait()

def generate_scene_description(input_prompt, image_data):
    """Generates a scene description using Google Generative AI."""
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content([input_prompt, image_data[0]])
    return response.text

def highlight_objects(image, results):
    """Draw bounding boxes around detected objects in the image."""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    
    # Loop through each detected object
    for *xyxy, conf, cls in results.xywh[0]:  # results.xywh[0] contains detections for the first image
        # Ensure that the coordinates are valid and within bounds
        x1, y1, x2, y2 = [int(i) for i in xyxy]
        
        # Ensure x1 <= x2 and y1 <= y2 (coordinates should be sorted)
        x1, x2 = sorted([x1, x2])  # Sort x-coordinates (left < right)
        y1, y2 = sorted([y1, y2])  # Sort y-coordinates (top < bottom)
        
        # Ensure coordinates are within the image size
        width, height = img.size
        x1 = max(0, min(x1, width))
        y1 = max(0, min(y1, height))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))
        
        # Draw the bounding box with a red outline
        draw.rectangle([x1, y1, x2, y2], outline="red", width=5)
    
    return img

def input_image_setup(uploaded_file):
    """Prepares the uploaded image for processing."""
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        image_parts = [
            {
                "mime_type": uploaded_file.type,
                "data": bytes_data,
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded.")

# Upload Image Section
st.markdown("<h3 class='feature-header'>📤 Upload an Image</h3>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Drag and drop or browse an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

# Buttons Section
st.markdown("<h3 class='feature-header'>🔍 Choose a Feature</h3>", unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("Scene Description"):
        if uploaded_file:
            try:
                # Prepare the input prompt and image data
                input_prompt = "Describe the objects and scene in this image."
                image_data = [
                    {
                        "mime_type": uploaded_file.type,
                        "data": uploaded_file.getvalue(),
                    }
                ]
                
                # Use the Google Gemini API for generating a scene description
                scene_description = llm(
                    input_prompt,
                    image=image_data
                )
                
                # Display the result
                st.markdown("### Scene Description:")
                st.write(scene_description)
            
            except Exception as e:
                st.error(f"Error during scene description: {e}")
        else:
            st.error("Please upload an image first.")

with col2:
    if st.button("Object Detection"):
        if uploaded_file:
            results = model(image)  # YOLO object detection
            processed_image = image.copy()  # Replace with actual bounding box logic
            st.image(processed_image, caption="Image with Object Detection", use_column_width=True)
        else:
            st.error("Please upload an image first.")

with col3:
    if st.button("Text Extraction"):
        if uploaded_file:
            extracted_text = pytesseract.image_to_string(image)
            st.text_area("Extracted Text", extracted_text, height=200)
        else:
            st.error("Please upload an image first.")

with col4:
    if st.button("Text-to-Speech"):
        if uploaded_file:
            extracted_text = pytesseract.image_to_string(image)
            if extracted_text.strip():
                engine.say(extracted_text)
                engine.runAndWait()
                st.success("Text-to-Speech conversion completed!")
            else:
                st.warning("No text found in the image.")
        else:
            st.error("Please upload an image first.")

# Footer
st.markdown(
    """
    <hr>
    <footer style="text-align:center;">
        <p>Powered by <strong>Google Gemini API</strong> Built with using Streamlit</p>
    </footer>
    """,
    unsafe_allow_html=True,
)

