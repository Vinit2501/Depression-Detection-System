import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
import pyaudio
import wave
import time

emotion_model = load_model("C:\\Users\\win10\\OneDrive\\Desktop\\emotion_detection_model.h5")
stress_model = load_model("C:\\Users\\win10\\OneDrive\\Desktop\\stress_detection_model.h5")

# Function to record audio
def record_audio():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    RECORD_SECONDS = 5
    WAVE_OUTPUT_FILENAME = "recorded_audio.wav"

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* recording")

    frames = []

    progress_bar = st.progress(0)
    start_time = time.time()
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
        elapsed_time = time.time() - start_time
        progress_bar.progress(min(elapsed_time / RECORD_SECONDS, 1.0))

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

       



def verify_audio(audio_data):
    # Load the audio data
    audio, sr = librosa.load(audio_data)
    # Check if the audio is blank
    if np.max(np.abs(audio)) < 0.015:
        st.write("Audio is blank. Please record again.")
    else:
        st.write("Audio recorded successfully!")


def add_noise(y, noise_factor=0.17):
    noise = np.random.normal(0, noise_factor, size=y.shape)
    noisy_y = y + noise
    return noisy_y


# Function to extract features from recorded audio
def extract_features(audio_file):
    y, sr = librosa.load(audio_file)
    y_noisy = add_noise(y)
    mfcc = librosa.feature.mfcc(y=y_noisy, sr=sr)
    mfcc_mean = np.mean(mfcc, axis=1)
    return mfcc_mean

# Function to evaluate stress level
def evaluate_stress_level(audio_file):
    features = extract_features(audio_file)
    scaled_features = StandardScaler().fit_transform(features.reshape(-1, 1)).flatten()
    scaled_features = scaled_features.reshape(1, -1)
    emotion_output = emotion_model.predict(scaled_features)
    emotion_class = np.argmax(emotion_output)
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']
    emotion = emotions[emotion_class]
    if emotion not in ['happy', 'surprised', 'neutral']:
        # Evaluate the input on the stress detection model
        features = extract_features(audio_file)
        scaled_features_stress = StandardScaler().fit_transform(features.reshape(-1, 1)).flatten()
        scaled_features_stress = scaled_features.reshape(1, -1)
        stress_output = stress_model.predict(scaled_features_stress)
        stress_class = np.argmax(stress_output)
        stresses = ['High', 'Low', 'Normal']
        stress = stresses[stress_class]
        
        if stress == 'Low':
            st.write("Person is Calm.")
        elif stress == 'Normal':
            st.write("Person is Stressed.")
        else:
            st.write("Person is Highly stressed/Depressed. Must consult a Doctor.")
    else:
        st.write("You are feeling " + emotion)



# Create GUI
st.title("Stress Detection")

# Record audio button
record_button = st.button("Record Audio")
if record_button:
    record_audio()
    verify_audio('C:\\Users\\win10\\OneDrive\\Desktop\\Final year\\recorded_audio.wav') 

# Evaluate stress level button
evaluate_button = st.button("Evaluate Audio")
if evaluate_button:
    audio_file = 'C:\\Users\\win10\\OneDrive\\Desktop\\Final year\\recorded_audio.wav'
    stress_level = evaluate_stress_level(audio_file)

# Display audio waveform
audio_file = 'C:\\Users\\win10\\OneDrive\\Desktop\\Final year\\recorded_audio.wav'
y, sr = librosa.load(audio_file)
plt.figure(figsize=(10, 6))
librosa.display.waveshow(y, sr=sr)
st.pyplot(plt)