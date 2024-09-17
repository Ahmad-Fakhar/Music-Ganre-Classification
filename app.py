import streamlit as st
import time
from transformers import pipeline
import librosa
import numpy as np
import plotly.graph_objects as go
import tempfile
import os
import soundfile as sf

# Set page config
st.set_page_config(page_title="🎵 Music Genre Classification", layout="wide")
# Custom CSS for UI
st.markdown("""
<style>
    .main-title {
        font-size: 3rem;
        color: #1DB954;
        text-align: center;
        padding: 2rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-title {
        font-size: 1.5rem;
        color: #191414;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stAudio {
        margin: 2rem auto;
        display: block;
    }
    .genre-result {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        color: #1DB954;
        margin: 1rem 0;
    }
    .prediction-time {
        font-size: 1.2rem;
        color: #191414;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return pipeline("audio-classification", model="juangtzi/wav2vec2-base-finetuned-gtzan")

pipe = load_model()

def convert_to_wav(audio_file):
    """Converts uploaded audio file to WAV format."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
        # Use soundfile to load and save the audio file as WAV
        audio_data, samplerate = sf.read(audio_file)
        sf.write(tmp_wav.name, audio_data, samplerate)
        return tmp_wav.name

def classify_audio(audio_file):
    """Classifies the audio file using the loaded model."""
    start_time = time.time()

    # Convert to WAV format before passing to the model
    wav_file = convert_to_wav(audio_file)
    
    try:
        # Use the wav file with the model
        preds = pipe(wav_file)
        outputs = {p["label"]: p["score"] for p in preds}
        end_time = time.time()
        prediction_time = end_time - start_time
        return outputs, prediction_time
    finally:
        os.unlink(wav_file)  # Remove the temp file

# Page title and subtitle
st.markdown("<h1 class='main-title'>🎵 Music Genre Classification</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>CNN By Deep Learning</p>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Upload a music file and let AI detect its genre!</p>", unsafe_allow_html=True)

# Sidebar with model and dataset information
st.sidebar.title("About")
st.sidebar.info("""
This app uses to classify music genres.
Model: juangtzi/wav2vec2-base-finetuned-gtzan
Dataset: GTZAN
""")

# Upload file section
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg"])

if uploaded_file is not None:
    # Display the uploaded audio file
    st.audio(uploaded_file)
    
    # Classify the uploaded audio
    if st.button("Classify Genre"):
        with st.spinner("Analyzing the music... 🎧"):
            try:
                results, pred_time = classify_audio(uploaded_file)
                
                # Get the top predicted genre
                top_genre = max(results, key=results.get)
                
                # Display the top predicted genre
                st.markdown(f"<h2 class='genre-result'>Detected Genre: {top_genre.capitalize()}</h2>", unsafe_allow_html=True)
                st.markdown(f"<p class='prediction-time'>Prediction Time: {pred_time:.2f} seconds</p>", unsafe_allow_html=True)
                
                # Plot the genre probabilities as a bar chart
                fig = go.Figure(data=[go.Bar(
                    x=list(results.keys()),
                    y=list(results.values()),
                    marker_color='#1DB954'
                )])
                fig.update_layout(
                    title="Genre Probabilities",
                    xaxis_title="Genre",
                    yaxis_title="Probability",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)

                # # Load the audio for displaying waveform
                # y, sr = librosa.load(uploaded_file, sr=None)
                
                # # Plot the audio waveform
                # st.subheader("Audio Waveform")
                # fig_waveform = go.Figure(data=[go.Scatter(y=y, mode='lines', line=dict(color='#1DB954'))])
                # fig_waveform.update_layout(
                #     title="Audio Waveform",
                #     xaxis_title="Time",
                #     yaxis_title="Amplitude",
                #     paper_bgcolor='rgba(0,0,0,0)',
                #     plot_bgcolor='rgba(0,0,0,0)'
                # )
                # st.plotly_chart(fig_waveform, use_container_width=True)

                # 🎈 Show balloons after successfully displaying the results
                st.balloons()
            
            except Exception as e:
                st.error(f"An error occurred while processing the audio: {str(e)}")
                st.info("Please try uploading the file again or use a different audio file.")

# Footer
st.markdown("""
<div style='text-align: center; margin-top: 2rem;'>
    <p>Created by Muhammad Jawad And Ahmad Fakhar</p>
</div>
""", unsafe_allow_html=True)
