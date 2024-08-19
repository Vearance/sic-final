import streamlit as st
import pymongo
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from PIL import Image
import tensorflow as tf
import pandas as pd
import numpy as np
import time
from joblib import load
from datetime import datetime, timedelta
import cv2
import os

# Get directory
current_dir = os.path.dirname(os.path.abspath(__file__))

tflite_path = os.path.join(current_dir, "..", "model", "image_model.tflite")
iforest_path = os.path.join(current_dir, "..", "model", "iforest_pipeline.pkl")

# Load sensor model
try:
    model = load(iforest_path)
    model_status = "Model loaded successfully!"
except Exception as e:
    model_status = f"Error loading model: {e}"
    st.stop()

password = ""
uri = ""

# Load model
interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# MongoDB connection
try:
    client = MongoClient(uri, server_api=ServerApi('1'))
    db = client["temp_humid"]
    collection = db["sensor_collection"]
    client.admin.command('ping')
    db_status = "Connected to MongoDB!"
except Exception as e:
    db_status = f"Error connecting to MongoDB: {e}"
    st.stop()

st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Fire Detection System - Real-time Model Testing")

# Session state
if 'run_loop' not in st.session_state:
    st.session_state.run_loop = False
if 'data_list' not in st.session_state:
    st.session_state.data_list = []
if 'alert_time' not in st.session_state:
    st.session_state.alert_time = datetime.min
if 'alert_active' not in st.session_state:
    st.session_state.alert_active = False

st.session_state.run_loop = True

alert_duration = timedelta(seconds=5)


def fetch_latest_data():
    latest_data = collection.find_one(sort=[('time', pymongo.DESCENDING)])
    return latest_data


def fetch_initial_data(limit=50):
    initial_data = list(collection.find().sort('time', pymongo.DESCENDING).limit(limit))
    return initial_data[::-1]


def make_prediction(data):
    try:
        df = pd.DataFrame({'temperature': [float(data['temperature'])],
                           'humidity': [float(data['humidity'])]})
        prediction = model.predict(df)
        return prediction[0]
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None


def update_chart():
    df = pd.DataFrame(st.session_state.data_list)
    if not df.empty and 'temperature' in df.columns and 'humidity' in df.columns:
        chart.line_chart(df.set_index('time')[['temperature', 'humidity']])
    else:
        st.warning("Not enough data to display chart yet.")


def send_alert(message):
    if not st.session_state.alert_active:
        st.session_state.alert_active = True
        st.session_state.alert_time = datetime.now()
        st.session_state.alert_placeholder = st.empty()
    st.session_state.alert_placeholder.warning(f"ALERT: {message}")


def clear_alert():
    if st.session_state.alert_active:
        if datetime.now() - st.session_state.alert_time > alert_duration:
            st.session_state.alert_placeholder.empty()
            st.session_state.alert_active = False


def fetch_historical_data():
    historical_data = list(collection.find().sort('time', pymongo.DESCENDING).limit(1000))
    return pd.DataFrame(historical_data)


def transform_image(image):
    image = image.convert('L')
    image = image.resize((96, 96))
    image = np.array(image).astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=(0, -1))

    if input_details[0]['dtype'] == np.float32:
        return image.astype(np.float32)
    elif input_details[0]['dtype'] == np.int8:
        return (image * 255).astype(np.int8)
    else:
        raise ValueError(f"Unsupported input dtype: {input_details[0]['dtype']}")


def process_frame(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image_tensor = transform_image(image)

    interpreter.set_tensor(input_details[0]['index'], image_tensor)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    cam_prediction = output_data[0][0]
    result = "Fire detected" if cam_prediction > 0.5 else "No fire detected"

    return result


# Fetch initial data
if not st.session_state.data_list:
    initial_data = fetch_initial_data()
    for data in initial_data:
        st.session_state.data_list.append({
            'time': datetime.strptime(data['time'], "%d %B %Y, %H:%M:%S"),
            'temperature': float(data['temperature']),
            'humidity': float(data['humidity']),
        })

st.markdown(
    """
    <style>
    .success-box {
        background-color: #B4E380;
        border-color: #c3e6cb;
        color: #155724;
        padding: 5px;
        border-radius: 5px;
        font-size: 10px;
        width: 100%;
        margin: 3px 0;
    }
    .error-box {
        background-color: #f8d7da;
        border-color: #f5c6cb;
        color: #721c24;
        padding: 5px;
        border-radius: 5px;
        font-size: 10px;
        width: 100%;
        margin: 3px 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

cam = cv2.VideoCapture(1)


def get_frame():
    ret, frame = cam.read()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None


# Layout
col1, col2 = st.columns([10, 4])


with col1:
    st.subheader("Camera Feed")
    camera_feed = st.empty()
    camera_prediction = st.empty()

    with st.expander("Real-time Data and Predictions", expanded=True):
        chart = st.empty()
        data_display = st.empty()
        prediction_display = st.empty()

with col2:
    start_button = st.button("Start real-time testing")
    stop_button = st.button("Stop")

    if st.button("View historical data"):
        historical_data = fetch_historical_data()
        st.line_chart(historical_data.set_index('time')[['temperature', 'humidity']])

with st.sidebar:
    st.subheader("Settings")
    with st.expander("Model and Database Status", expanded=True):
        if model_status == "Model loaded successfully!":
            st.markdown(f"<div class='success-box'>{model_status}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='error-box'>{model_status}</div>", unsafe_allow_html=True)

        if db_status == "Connected to MongoDB!":
            st.markdown(f"<div class='success-box'>{db_status}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='error-box'>{db_status}</div>", unsafe_allow_html=True)
        st.write("")


# Initial chart
update_chart()

# Display camera
initial_frame = get_frame()
if initial_frame is not None:
    camera_feed.image(initial_frame, channels="RGB", use_column_width=True)
else:
    camera_feed.error("Failed to initialize camera feed")

error_placeholder = st.empty()
last_prediction_time = time.time()

if start_button:
    st.session_state.run_loop = True

if stop_button:
    st.session_state.run_loop = False

while True:
    # Update camera frame
    frame = get_frame()
    if frame is not None:
        camera_feed.image(frame, channels="RGB", use_column_width=True)

    if st.session_state.get('run_loop', False):
        current_time = time.time()
        camera_result = "No prediction yet"

        if current_time - last_prediction_time >= 3:
            camera_result = process_frame(frame)
            camera_prediction.write(f"Camera Prediction: {camera_result}")
            last_prediction_time = current_time

        # Sensor data processing
        latest_data = fetch_latest_data()
        if latest_data:
            try:
                features = {
                    'time': datetime.strptime(latest_data['time'], "%d %B %Y, %H:%M:%S"),
                    'temperature': float(latest_data['temperature']),
                    'humidity': float(latest_data['humidity']),
                }
                prediction = make_prediction(features)

                st.session_state.data_list.append(features)
                if len(st.session_state.data_list) > 50:
                    st.session_state.data_list.pop(0)

                update_chart()

                data_display.write(f"Latest data: Time: {features['time']}, Temperature: {features['temperature']:.2f}, Humidity: {features['humidity']:.2f}")
                if prediction is not None:
                    prediction_result = 'Fire detected' if prediction else 'No fire detected'
                    prediction_display.write(f"Sensor Prediction: {prediction_result}")

                    if prediction_result == 'Fire detected' and camera_result == 'Fire detected':
                        send_alert("Fire detected in both sensor and camera!")
                    elif st.session_state.alert_active:
                        clear_alert()

                # Clear any previous error messages
                error_placeholder.empty()

            except Exception as e:
                error_placeholder.error(f"Error processing data: {str(e)}")

    if not st.session_state.run_loop:
        st.write("Testing stopped.")

    time.sleep(0.1)
