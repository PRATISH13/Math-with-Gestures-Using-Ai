import cv2
import google.generativeai as genai
from PIL import Image
import streamlit as st
import numpy as np
import time
import tempfile
from cvzone.HandTrackingModule import HandDetector
from gtts import gTTS
from io import BytesIO

# Configure genai with your API key
genai.configure(api_key="AIzaSyCYIE1pn7hD0JL1psyiN6zmuX_c1B2oogQ")
model = genai.GenerativeModel('gemini-1.5-flash')

# Set page layout
st.set_page_config(layout="wide")
# Add background image
background_image = """
<style>
[data-testid="stAppViewContainer"] > .main {
    background-image: url("https://images.unsplash.com/photo-1517241034903-9a4c3ab12f00?q=80&w=2940&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}
h1, h2, h3, h4, h5, h6, p{
    color: white !important;
}


.stMarkdown div{
    color: white !important;
}


div[data-testid="stExpander"] {
    border: 2px solid white !important;
}
</style>
"""
st.markdown(background_image, unsafe_allow_html=True)

# Hero section layout
st.markdown("""
       <style>
        @keyframes gradientShine {
            0% {
                background-position: 0% 0%;
            }
            100% {
                background-position: 100% 100%;
            }
        }

        @keyframes textShine {
            0% {
                background-position: 0% 50%;
            }
            100% {
                background-position: 100% 50%;
            }
        }
        .stExpander[data-state="expanded"] {
          height: auto !important;
        }

        .shiny-button {
            padding: 0.6em 1.2em;
            border-radius: 0.375em;
            cursor: pointer;
            color: white;
            background: linear-gradient(120deg, darkmagenta, crimson, orange);
            background-size: 200% 100%;
            background-position: 100% 0;
            transition: background-position .5s;
            border: none;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            animation: gradientShine 5s ease-in-out infinite; /* Optional: keep animation if desired */
        }

        .shiny-button:hover {
            background-position: 0 0;
        }

        .section {
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 40px;
        }

        .heading-container {
            display: flex;
            justify-content: center;
            width: 100%;
            max-width: 900px; /* Adjust to fit your layout */
            margin-bottom: 20px;
        }

        .heading-container h2 {
            margin: 0 20px;
            font-size: 24px;
            text-align: center;
            background: linear-gradient(90deg, hsla(208, 67%, 81%, 1) 0%, hsla(37, 65%, 85%, 1) 50%, hsla(301, 65%, 83%, 1) 100%);
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
            text-fill-color: transparent;
            background-size: 500% auto;
            animation: textShine 5s ease-in-out infinite alternate;
            /* For older versions of IE and Edge */
            filter: progid: DXImageTransform.Microsoft.gradient(startColorstr="#AED1EF", endColorstr="#F2DFC1", GradientType=1);
        }

        .image-container {
            display: flex;
            justify-content: center;
            gap: 20px;
            width: 100%;
            max-width: 900px; /* Match the width of heading container */
        }

        .image-item {
            position: relative;
            overflow: hidden;
            width: 300px; /* Increased width */
            height: 225px; /* Increased height */
            perspective: 1000px;
        }

        .image-item img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            transition: transform 0.5s ease, box-shadow 0.5s ease;
            transform-style: preserve-3d;
        }

        .image-item:nth-child(1) img {
            transform: rotateY(15deg);
        }

        .image-item:nth-child(2) img {
            transform: rotateY(0deg);
        }

        .image-item:nth-child(3) img {
            transform: rotateY(-15deg);
        }

        .image-item:hover img {
            transform: rotateY(0deg) scale(1.1);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }
        h1 {
          background: linear-gradient(90deg, hsla(169, 76%, 48%, 1) 0%, hsla(67, 87%, 82%, 1) 100%);
          background: -moz-linear-gradient(90deg, hsla(169, 76%, 48%, 1) 0%, hsla(67, 87%, 82%, 1) 100%);
          background: -webkit-linear-gradient(90deg, hsla(169, 76%, 48%, 1) 0%, hsla(67, 87%, 82%, 1) 100%);
          background: -o-linear-gradient(90deg, hsla(169, 76%, 48%, 1) 0%, hsla(67, 87%, 82%, 1) 100%);
          background: -ms-linear-gradient(90deg, hsla(169, 76%, 48%, 1) 0%, hsla(67, 87%, 82%, 1) 100%);
          filter: progid: DXImageTransform.Microsoft.gradient(startColorstr="#1ED7B5", endColorstr="#F0F9A7", GradientType=1);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
        }


    </style>


    <div style="display: flex; align-items: center; justify-content: space-between; padding: 20px;">
        <div style="flex: 1; padding: 20px;">
            <br/><br/><br/><br/><br/>
            <h1>Welcome to the AI-powered Gesture and Math Solver App!</h1>
            <p>Interact with the app using your hand gestures, and solve math problems instantly.</p>
            <a href="#live-feed-section" style="text-decoration: none;">
                <button class="shiny-button">
                    Try Now
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="white" width="24px" height="24px" style="margin-left: 10px;">
                        <path d="M12 2v2h7.586l-9.293 9.293 1.414 1.414L21 5.414V13h2V2z"/>
                        <path d="M0 0h24v24H0z" fill="none"/>
                    </svg>
                </button>
            </a>
        </div>
        <div style="flex: 1; margin-top:160px;">
            <img src="https://i.ibb.co/dW2tNDZ/Connecting-Math-and-Machine-Learning.jpg" alt="AI-powered App" style="width: 100%; border-radius: 10px;"/>
        </div>
    </div>
    <br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/>


""", unsafe_allow_html=True)

# Live feed section anchor
st.markdown('<a name="live-feed-section"></a>', unsafe_allow_html=True)

# Initialize session state to store values for both modules

if 'output_text' not in st.session_state:
    st.session_state['output_text'] = ""
if 'canvas' not in st.session_state:
    st.session_state['canvas'] = np.zeros((720, 1280, 3), np.uint8)
if 'image_captured' not in st.session_state:
    st.session_state['image_captured'] = False
if 'image_path' not in st.session_state:
    st.session_state['image_path'] = None
if 'flipped_image' not in st.session_state:
    st.session_state['flipped_image'] = None
if 'audio_file_path' not in st.session_state:
    st.session_state['audio_file_path'] = None


# Webcam settings
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)

# Create columns for layout
col1, col2 = st.columns([5, 2])

with col1:
    st.subheader("Live Feed")
    live_feed_placeholder = st.empty()  # Placeholder for live feed
    with st.container():
        live_feed_placeholder.markdown('<div class="live-feed-container"></div>',
                                       unsafe_allow_html=True)  # Live feed section

    captured_image_placeholder = st.empty()  # Placeholder for captured image

with col2:
    st.subheader("Results")
    with st.expander("Results", expanded=True):
        result_area = st.empty()  # Placeholder for result area
        audio_placeholder = st.empty()
    processing_spinner = st.empty()  # Placeholder for processing spinner


# Function to capture and flip the image from webcam when pinky finger is raised
def capture_image(frame):
    flipped_frame = cv2.flip(frame, 1)  # Flip the image horizontally
    return flipped_frame

def text_to_audio(text, filename="output.mp3"):
    tts = gTTS(text)
    audio_fp = BytesIO()
    tts.write_to_fp(audio_fp)  # Save the audio to a BytesIO object in memory
    audio_fp.seek(0)  # Go to the beginning of the BytesIO object
    with open(filename, 'wb') as f:
        f.write(audio_fp.read())  # Write the content of the BytesIO object to the file
    return filename

def display_result_and_audio(result_text):
    if result_text:
        if "I cannot see or analyze any image" in result_text:
            result_text = "Try again"
        st.session_state['output_text'] = result_text

        # Display result text
        result_area.markdown(
            f"<div style='font-size: 20px;'>{result_text}</div>",
            unsafe_allow_html=True
        )

        # Generate and display the audio
        audio_file_path = text_to_audio(result_text)
        if audio_file_path:
            # Use the dedicated placeholder to render the audio bar
            audio_placeholder.audio(audio_file_path, format="audio/mp3", start_time=0)
        else:
            st.error("Audio file not found or could not be created.")
    else:
        st.error("No result was returned from the API.")




# Function to process image using Gemini API
def process_image_with_gemini(image, retry_attempts=3, delay_between_attempts=5):
    try:
        for attempt in range(retry_attempts):
            try:
                # Save the PIL image to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                    pil_image = Image.fromarray(image)
                    pil_image.save(temp_file, format='JPEG')
                    temp_file_path = temp_file.name

                myfile = genai.upload_file(temp_file_path)
                result = model.generate_content(["Solve the math problem in this image.", myfile])

                if result.text:
                    # Check if the result text contains an irrelevant message
                    if "I cannot see or analyze any image. I am only a text-based chat assistant and thus I cannot process any image." in result.text:
                        return "Try again"
                    return result.text

                else:
                    raise Exception("No text returned from Gemini API")
            except Exception as e:
                st.error(f"Error processing image with Gemini (attempt {attempt + 1}): {e}")
                time.sleep(delay_between_attempts)
        return None
    except Exception as e:
        st.error(f"Error communicating with Gemini API: {e}")
        return None


# Function to get hand information
def getHandInfo(img):
    hands, img = detector.findHands(img, draw=False, flipType=True)

    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        fingers = detector.fingersUp(hand)
        return fingers, lmList
    else:
        return None


# Function to draw on canvas
def draw(info, prev_pos, canvas):
    fingers, lmList = info
    current_pos = None
    if fingers == [0, 1, 0, 0, 0]:  # Index finger up (draw mode)
        current_pos = lmList[8][0:2]
        if prev_pos is None:
            prev_pos = current_pos
        cv2.line(canvas, current_pos, prev_pos, (255, 0, 255), 10)
    elif fingers == [1, 0, 0, 0, 0]:  # Thumb up (clear canvas)
        canvas = np.zeros_like(canvas)
        st.session_state['flipped_image'] = None  # Remove captured image
        captured_image_placeholder.empty()  # Clear the captured image display
        # Additional logic to remove the captured image from the streamlit display
        if 'captured_image' in st.session_state:
            st.session_state.pop('captured_image')  # Clear the captured image display

    return current_pos, canvas


# Function to handle image capture and processing when pinky is raised
def check_and_capture_image(fingers, frame):
    if fingers == [0, 0, 0, 0, 1]:  # Pinky finger up (capture mode)
        flipped_image = capture_image(frame)
        st.session_state['flipped_image'] = flipped_image
        st.session_state['image_captured'] = True
        return flipped_image
    return None


# Function to send the drawn canvas to the AI model when four fingers are up
def sendToAI(model, canvas, fingers):
    if fingers == [1, 1, 1, 1, 0]:  # Four fingers up (trigger processing)
        pil_image = Image.fromarray(canvas)
        response = model.generate_content(["Solve this math problem", pil_image])
        return response.text
    return None


# Function to reset the state after displaying the result
def reset_state():
    st.session_state['output_text'] = ""
    st.session_state['canvas'] = np.zeros((720, 1280, 3), np.uint8)
    st.session_state['image_captured'] = False
    st.session_state['image_path'] = None
    st.session_state['flipped_image'] = None

    st.experimental_rerun()


# Main loop for webcam feed and hand gesture drawing
prev_pos = None
canvas = st.session_state['canvas']
output_text = ""

while cap.isOpened():
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # Get hand gesture info
    info = getHandInfo(img)
    if info:
        fingers, lmList = info
        prev_pos, canvas = draw(info, prev_pos, canvas)

        # Capture image when pinky finger is raised
        if not st.session_state['image_captured']:
            flipped_image = check_and_capture_image(fingers, img)
            if flipped_image is not None:
                # Display the captured image when pinky is raised

                # Show spinner while processing
                if st.session_state['image_captured'] and st.session_state['flipped_image'] is not None:
                    with st.spinner('Processing the image...'):
                        result_text = process_image_with_gemini(st.session_state['flipped_image'])
                        st.session_state['image_captured'] = False  # Remove captured image after processing

                        if result_text:
                            # Check if the result contains the unwanted message
                            if "I am sorry, I cannot see or analyze any image" in result_text:
                                result_text = "Try again"

                            st.session_state['output_text'] = result_text
                            display_result_and_audio(result_text)


                            result_area.markdown(
                                f"<div style='font-size: 20px;'>{st.session_state['output_text']}</div>",
                                unsafe_allow_html=True)
                        else:
                            st.error("No result was returned from the API.")

                # captured_image_placeholder.image(cv2.resize(flipped_image, (300, 200)), caption='Captured Image')

        # Send canvas to AI when four fingers are up
        if fingers == [1, 1, 1, 1, 0]:  # Check for four fingers gesture
            if not st.session_state['image_captured']:
                with st.spinner("Processing your drawing..."):
                    output_text = sendToAI(model, canvas, fingers)
                    if output_text:
                        display_result_and_audio(output_text)
                        st.session_state['output_text'] = output_text
                        result_area.markdown(f"<div style='font-size: 20px;'>{st.session_state['output_text']}</div>",
                                             unsafe_allow_html=True)

    # Increase the size of the live feed
    image_combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
    live_feed_placeholder.image(image_combined, channels="BGR", use_column_width=True)

    st.session_state['canvas'] = canvas

    cv2.waitKey(1)

cap.release()
