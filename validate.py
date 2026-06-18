import os
import random
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv3D, MaxPool3D, TimeDistributed, Flatten,
                                     Bidirectional, LSTM, Dense, Dropout)
from tensorflow.keras.optimizers import Adam
from jiwer import wer, cer   # pip install jiwer

# ---------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------
FRAME_H, FRAME_W = 46, 140
TARGET_FRAMES = 75
ALIGNMENT_MAX_LEN = 50

# ---------------------------------------------------------
# LOADING FUNCTIONS (same as training)
# ---------------------------------------------------------
def load_video(path: str):
    cap = cv2.VideoCapture(path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        crop = gray[190:236, 80:220]  
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

    frames = frames[..., None]
    return frames.astype("float32")


vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

def load_alignments(path):
    if isinstance(path, bytes):
        path = path.decode()
    with open(path, "r") as f:
        lines = f.readlines()

    tokens = []
    for line in lines:
        p = line.split()
        if p[2] != "sil":
            tokens.append(" ")
            tokens.append(p[2])

    text = "".join(tokens)
    chars = tf.strings.unicode_split(text, "UTF-8")
    ids = char_to_num(chars).numpy()

    if len(ids) > ALIGNMENT_MAX_LEN:
        ids = ids[:ALIGNMENT_MAX_LEN]
    else:
        ids = np.pad(ids, (0, ALIGNMENT_MAX_LEN - len(ids)), constant_values=0)

    return ids.astype("int64")

# ---------------------------------------------------------
# BUILD MODEL (same as training)
# ---------------------------------------------------------
model = Sequential()
model.add(Conv3D(64, 3, padding="same", activation="relu",
                 input_shape=(75, FRAME_H, FRAME_W, 1)))
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

model.compile(optimizer=Adam(1e-4))

print("\n🔄 Loading best weights...")
model.load_weights("best_weights.weights.h5")
print("Loaded!\n")

# ---------------------------------------------------------
# CTC DECODER
# ---------------------------------------------------------
def ctc_decode(pred):
    decoded, _ = tf.keras.backend.ctc_decode(pred,
                    input_length=np.ones(pred.shape[0]) * pred.shape[1],
                    greedy=True)
    text = num_to_char(decoded[0]).numpy()
    return ["".join([c.decode() for c in t if c not in [b'']]) for t in text]

# ---------------------------------------------------------
# RUN VALIDATION ON 10 RANDOM FILES
# ---------------------------------------------------------
align_dir = "data/s1_processed/align"
all_align_files = os.listdir(align_dir)
sample_files = random.sample(all_align_files, 10)

total_wer = 0
total_cer = 0

print("\n========== VALIDATION RESULTS ==========\n")

for fname in sample_files:
    align_path = os.path.join(align_dir, fname)
    base = fname.replace(".align", "")
    video_path = os.path.join("data/s1_processed", base + ".mpg")

    frames = load_video(video_path)
    frames = np.expand_dims(frames, axis=0)

    true_ids = load_alignments(align_path)
    true_chars = num_to_char(true_ids).numpy()
    true_text = "".join([c.decode() for c in true_chars if c != b''])

    pred = model.predict(frames)
    pred_text = ctc_decode(pred)[0]

    case_wer = wer(true_text, pred_text)
    case_cer = cer(true_text, pred_text)

    total_wer += case_wer
    total_cer += case_cer

    print(f"File: {fname}")
    print(f"GT   : {true_text}")
    print(f"PRED : {pred_text}")
    print(f"WER  : {case_wer:.3f} | CER : {case_cer:.3f}")
    print("----------------------------------------")

print("\n========== FINAL ACCURACY ==========\n")
char_acc = (1 - (total_cer/10)) * 100
word_acc = (1 - (total_wer/10)) * 100

print(f"\nCharacter Accuracy : {char_acc:.2f}%")
print(f"Word Accuracy      : {word_acc:.2f}%")


