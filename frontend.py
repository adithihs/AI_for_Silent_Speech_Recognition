import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import google.generativeai as genai
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv3D, MaxPool3D, TimeDistributed, Flatten,
                                     Bidirectional, LSTM, Dense, Dropout, StringLookup)

# ---------------------------------------------------------
# 1. ROBUST API SETUP (EXAM MODE)
# ---------------------------------------------------------
# Use your NEW working key here
GENAI_API_KEY = "AIzaSyCytxsVFfhIfiiEx_6Klq21_-4m8D61KC8" 

gemini_model = None

try:
    genai.configure(api_key=GENAI_API_KEY)
    # FORCE connection to 'gemini-1.5-flash' (Fastest & most reliable)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    print("✅ Connected to Gemini 1.5 Flash")
except Exception as e:
    print(f"⚠️ Connection Failed: {e}")
    # We leave gemini_model as None so the code knows to skip it later

def refine_with_gemini(raw_text):
    """
    Refines raw output. 
    CRITICAL: If API fails, returns raw_text silently to prevent exam crashes.
    """
    # 1. Validate input
    if not raw_text or not raw_text.strip():
        return "No speech detected."
        
    # 2. Check if model exists; if not, return raw text immediately
    if gemini_model is None:
        return raw_text 

    # 3. Strict prompt
    prompt = f"""
    You are a grammar corrector for the GRID Lip Reading dataset.
    User Input: "{raw_text}"
    Task: Convert to strict pattern: [Command] [Color] [Preposition] [Letter] [Digit] [Adverb]
    Example: "bin blue at f five soon"
    Output (Text ONLY):
    """
    
    try:
        # 4. Generate Content with 0 temperature (strict)
        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.0)
        )
        return response.text.strip()
        
    except Exception as e:
        # PRINT THE ERROR to console (for you), but SHOW RAW TEXT to user
        print(f"\n🔴 GEMINI ERROR: {e}\n")
        print("➡️ Returning raw model output instead.")
        return raw_text

# ---------------------------------------------------------
# 2. CONSTANTS & VIDEO PROCESSING
# ---------------------------------------------------------
FRAME_H, FRAME_W = 46, 140
TARGET_FRAMES = 75
vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = StringLookup(vocabulary=vocab, oov_token="")
num_to_char = StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)

def load_video_frames(file_path):
    cap = cv2.VideoCapture(file_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        crop = gray[190:236, 80:220]
        crop = crop.astype("float32")
        frames.append(crop)
    cap.release()
    
    if len(frames) == 0: 
        return np.zeros((TARGET_FRAMES, FRAME_H, FRAME_W, 1), dtype="float32")
    
    frames = np.array(frames)
    mean, std = frames.mean(), frames.std() + 1e-7
    frames = (frames - mean) / std
    
    if len(frames) >= TARGET_FRAMES: 
        frames = frames[:TARGET_FRAMES]
    else:
        pad = np.repeat(frames[-1][np.newaxis, ...], TARGET_FRAMES - len(frames), axis=0)
        frames = np.concatenate([frames, pad], axis=0)
        
    return frames[..., None].astype("float32")

# ---------------------------------------------------------
# 3. MODEL ARCHITECTURE
# ---------------------------------------------------------
def build_model():
    model = Sequential([
        Conv3D(64, 3, padding="same", activation="relu", input_shape=(TARGET_FRAMES, FRAME_H, FRAME_W, 1)),
        MaxPool3D((1, 2, 2)),
        Conv3D(128, 3, padding="same", activation="relu"),
        MaxPool3D((1, 2, 2)),
        Conv3D(128, 3, padding="same", activation="relu"),
        TimeDistributed(Flatten()),
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.4),
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.4),
        Dense(char_to_num.vocabulary_size() + 1, activation="softmax")
    ])
    return model

def decode_prediction(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy=True for speed/stability if beam search is causing issues, 
    # but kept beam search here as per your request.
    decode, _ = tf.nn.ctc_beam_search_decoder(
        inputs=np.transpose(pred, (1, 0, 2)), 
        sequence_length=input_len.astype(np.int32),
        beam_width=100, top_paths=1
    )
    decoded_indices = tf.sparse.to_dense(decode[0]).numpy()[0]
    return "".join([num_to_char(idx).numpy().decode('utf-8') for idx in decoded_indices])

# ---------------------------------------------------------
# 4. STREAMLIT UI
# ---------------------------------------------------------
st.set_page_config(page_title="Silent Speech Recognition", page_icon="🔇")
st.title("Silent Speech Recognition")

@st.cache_resource
def load_model_with_weights():
    # Ensure the weights file exists before trying to load
    if not os.path.exists("best_weights.weights.h5"):
        st.error("❌ Error: 'best_weights.weights.h5' not found in directory!")
        return None
        
    model = build_model()
    model.load_weights("best_weights.weights.h5")
    return model

# Load model once at startup
model = load_model_with_weights()

uploaded_video = st.file_uploader("Upload video (.mp4 / .mpg)", type=["mp4", "mpg"])

if uploaded_video and model is not None:
    # Save temp file
    tfile = "temp_input_video.mp4"
    with open(tfile, "wb") as f:
        f.write(uploaded_video.read())
        
    st.video(tfile)
    
    status = st.empty()
    status.write("⏳ Processing Video...")
    
    try:
        # 1. Preprocess
        frames = load_video_frames(tfile)
        
        # 2. Predict (Lip Reading)
        prediction = model.predict(frames[np.newaxis, ...])
        raw_text = decode_prediction(prediction)
        
        status.write(f"⏳ Refining Text: '{raw_text}' ...")
        
        # 3. Refine (Grammar Check)
        # This function is now SAFE - it will return raw_text if API fails
        final_text = refine_with_gemini(raw_text)
        
        status.empty()
        
        # 4. Display Result
        st.subheader("Predicted Text:")
        st.success(final_text)
        
    except Exception as e:
        st.error(f"System Error: {e}")