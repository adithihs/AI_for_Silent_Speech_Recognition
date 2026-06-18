import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv3D, MaxPool3D, TimeDistributed, Flatten,
                                     Bidirectional, LSTM, Dense, Dropout, StringLookup)

# ---------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------
FRAME_H, FRAME_W = 46, 140
TARGET_FRAMES = 75

# ---------------------------------------------------------
# VOCAB
# ---------------------------------------------------------
vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = StringLookup(vocabulary=vocab, oov_token="")
num_to_char = StringLookup(vocabulary=char_to_num.get_vocabulary(),
                           oov_token="", invert=True)

# ---------------------------------------------------------
# VIDEO PREPROCESSING
# ---------------------------------------------------------
def load_video_frames(file_path):
    cap = cv2.VideoCapture(file_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        crop = gray[190:236, 80:220]      # Same ROI as training
        crop = crop.astype("float32")
        frames.append(crop)

    cap.release()

    if len(frames) == 0:
        frames = [np.zeros((FRAME_H, FRAME_W), "float32")] * TARGET_FRAMES

    frames = np.array(frames)
    mean, std = frames.mean(), frames.std() + 1e-7
    frames = (frames - mean) / std

    if len(frames) >= TARGET_FRAMES:
        frames = frames[:TARGET_FRAMES]
    else:
        last = frames[-1]
        pad = np.repeat(last[np.newaxis, ...], TARGET_FRAMES - len(frames), axis=0)
        frames = np.concatenate([frames, pad], axis=0)

    return frames[..., None].astype("float32")


# ---------------------------------------------------------
# MODEL ARCHITECTURE
# (must match EXACTLY with training)
# ---------------------------------------------------------
def build_model():
    model = Sequential()

    model.add(Conv3D(64, 3, padding="same", activation="relu",
                     input_shape=(TARGET_FRAMES, FRAME_H, FRAME_W, 1)))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(Conv3D(128, 3, padding="same", activation="relu"))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(Conv3D(128, 3, padding="same", activation="relu"))
    model.add(TimeDistributed(Flatten()))

    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Dropout(0.4))

    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Dropout(0.4))

    model.add(Dense(char_to_num.vocabulary_size() + 1, activation="softmax"))

    return model


# ---------------------------------------------------------
# CTC DECODE
# ---------------------------------------------------------
def decode_prediction(pred):
    pred_ids = tf.argmax(pred, axis=-1)
    chars = num_to_char(pred_ids)
    text = tf.strings.reduce_join(chars, axis=-1)
    return text.numpy()[0].decode("utf-8").replace("[UNK]", "").strip()


# ---------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------
st.title("🔵 Lip Reading AI — Prediction (Using best_weights)")
st.write("Upload a video and the model will predict the spoken text.")

uploaded_video = st.file_uploader("Upload video (.mp4 / .mpg)", type=["mp4", "mpg"])


# Load model + weights only once
@st.cache_resource
def load_model_with_weights():
    model = build_model()
    model.load_weights("best_weights.weights.h5")
    return model


if uploaded_video:
    # Save temp file
    with open("temp_input_video.mp4", "wb") as f:
        f.write(uploaded_video.read())

    st.video("temp_input_video.mp4")

    st.write("⏳ Processing video...")
    frames = load_video_frames("temp_input_video.mp4")

    model = load_model_with_weights()

    st.write("⏳ Predicting...")
    prediction = model.predict(frames[np.newaxis, ...])

    text = decode_prediction(prediction)

    st.subheader("📝 Predicted Text:")
    st.success(text)
