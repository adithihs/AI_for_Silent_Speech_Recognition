import numpy as np
import matplotlib.pyplot as plt
import cv2
import imageio
import os
import glob
import string
import tensorflow as tf
from typing import List
import json
from datetime import datetime

import urllib.request

url = "https://github.com/italojs/facial-landmarks-recognition/raw/master/shape_predictor_68_face_landmarks.dat"
filename = "shape_predictor_68_face_landmarks.dat"

# print("Downloading… Please wait (100MB)…")
# urllib.request.urlretrieve(url, filename)
# print("Download complete:", filename)
import cv2
import mediapipe as mp

# Initialize mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
FRAME_DIM = 64

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True
)

MOUTH_LANDMARKS = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 
    308, 415, 310, 311, 312, 13, 82, 81, 80, 191,
    78, 95, 88
]
def clip_mouth_mediapipe(frame, last_crop=None):
    """Extract mouth using MediaPipe mesh; fallback to last crop if needed."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return last_crop  # fallback

    landmarks = results.multi_face_landmarks[0].landmark

    # Mouth indices (MediaPipe mesh)
    mouth_ids = [
        61, 146, 91, 181, 84, 17, 314, 405, 321,
        375, 291, 308, 324, 318, 402, 317, 14,
        87, 178, 88, 95
    ]

    h, w, _ = frame.shape
    pts = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in mouth_ids])

    x1, y1 = np.min(pts, axis=0)
    x2, y2 = np.max(pts, axis=0)

    # padding
    pad = 20
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return last_crop

    # square resize
    crop = cv2.resize(crop, (96, 96))

    return crop


def adjust_frame_count(frames, target_frames):
    n = len(frames)
    if n == target_frames:
        return tf.stack(frames)
    frames_tf = tf.stack(frames)  # [N,H,W,1], float32 0..1 later
    if n > target_frames:
        start = tf.random.uniform([], 0, n - target_frames, dtype=tf.int32)
        return frames_tf[start:start + target_frames]
    # n < target_frames: repeat last frame
    pad = target_frames - n
    last = frames_tf[-1:]
    pad_block = tf.repeat(last, repeats=pad, axis=0)
    return tf.concat([frames_tf, pad_block], axis=0)


def load_video(path: str, target_frames=75, target_size=(FRAME_DIM, FRAME_DIM)) -> tf.Tensor:
    """Video loader using MediaPipe mouth detection with fallback."""
    
    cap = cv2.VideoCapture(path)
    frames = []
    last_crop = None  # fallback when mouth not detected
    
    frame_count = 0
    extracted = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Get mouth using mediapipe
        mouth_region = clip_mouth_mediapipe(frame, last_crop=last_crop)

        # If still None → skip frame
        if mouth_region is None:
            print(f"⚠ No mouth in frame {frame_count}: {path}")
            continue
        
        # Store fallback
        last_crop = mouth_region
        extracted += 1

        # Resize + grayscale + channel expand
        mouth_region = cv2.resize(mouth_region, target_size)
        mouth_region = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2GRAY)
        mouth_region = tf.expand_dims(mouth_region, axis=-1)
        frames.append(mouth_region)
    
    cap.release()

    # ------------------------------------------------------------------
    # SAFETY: If video still empty, generate blank frames (so training continues)
    # ------------------------------------------------------------------
    if len(frames) == 0:
        print(f"❌ Warning: NO FRAMES extracted for: {path}")
        blank = tf.zeros((target_frames, *target_size, 1), dtype=tf.float32)
        return blank

    # ------------------------------------------------------------------
    # Fix number of frames
    # ------------------------------------------------------------------
    frames = adjust_frame_count(frames, target_frames)

    frames_tensor = tf.stack(frames)
    frames_tensor = tf.cast(frames_tensor, tf.float32)

    # Normalize safely
    return tf.clip_by_value(frames_tensor / 255.0, 0.0, 1.0)

vocab = string.ascii_lowercase + "'?! "

vocab = list(vocab)
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)
char2num_dict = {c: char_to_num(c).numpy() for c in char_to_num.get_vocabulary()}
num2char_dict = {char_to_num(c).numpy():c  for c in num_to_char.get_vocabulary()}
def load_alignment(path : str):
    with open(path, "r") as f:
        lines = f.readlines()
    tokens = []
    for line in lines:
        start, end, text = line.split()
        if text!='sil':
            tokens.append(text)

    chars = list(" ".join(tokens))
    return char_to_num(chars)
num = load_alignment("data/s10_processed/align/bbab8n.align")

print(num)

print(num_to_char(num))
print(tf.cast(tf.shape(num)[0], tf.int32))


def load_data(video_path : str):
    print(video_path)
    video_id = video_path.numpy().decode('UTF-8').replace("\\", "/").split("/")[-1].split(".")[0]
    folder_id = video_path.numpy().decode('UTF-8').replace("\\", "/").split("/")[-2].split("_")[0]
    align_path = f"data/{folder_id}_processed/align/{video_id}.align"
    video_path = video_path.numpy().decode('UTF-8')
    video_data = load_video(video_path)
    char_num = load_alignment(align_path)
    return video_data, char_num
def mappable_function(path: str):
    # Do not swallow errors; let ignore_errors() handle them.
    print(path)
    return tf.py_function(load_data, [path], (tf.float32, tf.int64))

videos = glob.glob("data/s2_processed/*.mpg")
from sklearn.model_selection import train_test_split
train,test = train_test_split(videos,test_size = 0.2,random_state = 7)
BATCH_SIZE = 10
def create_dataset(cache_filepath, video_paths_list):
    data = tf.data.Dataset.from_tensor_slices(video_paths_list)
    data = data.map(mappable_function, num_parallel_calls=tf.data.AUTOTUNE)
    data = data.cache(cache_filepath)
    data = data.shuffle(len(video_paths_list))
    data = data.apply(tf.data.experimental.ignore_errors())
    data = data.padded_batch(
        BATCH_SIZE,
        padded_shapes=([75, FRAME_DIM, FRAME_DIM, 1], [None]),
        padding_values=(tf.constant(0., tf.float32), tf.constant(-1, tf.int64)),
        drop_remainder=True,  # important for multi-GPU
    )
    data = data.prefetch(tf.data.AUTOTUNE)
    data = data.repeat()
    return data

os.makedirs("cache", exist_ok=True)

data = create_dataset(f"cache/preprocessed_data_cache",train)
val = create_dataset(f"cache/preprocessed_val_cache",test)
#data = create_dataset(f"preprocessed_data_cache",train)
#val = create_dataset(f"preprocessed_val_cache",test)

def CTCLoss(y_true, y_pred):
    # y_true: int64 with -1 padding (shape [B, max_L])
    # y_pred: float32 logits/probs (shape [B, T, C])
    batch_size = tf.shape(y_true)[0]
    T = tf.shape(y_pred)[1]

    # true lengths = number of labels != -1 per sample
    label_len = tf.math.count_nonzero(tf.not_equal(y_true, -1), axis=1, dtype=tf.int64)
    label_len = tf.reshape(label_len, [batch_size, 1])

    # all time steps are valid (no temporal downsample in your 3D path)
    input_len = tf.cast(T, tf.int64) * tf.ones([batch_size, 1], dtype=tf.int64)

    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_len, label_len)
from tensorflow.keras.layers import (
    Input, Conv3D, BatchNormalization, Activation, MaxPool3D,
    Reshape, TimeDistributed, Dense, Dropout,
    Bidirectional, GRU, Add, LayerNormalization,SpatialDropout3D
)
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.regularizers import l2


reg_strength = 0.0001
    
model = Sequential()

model.add(Input(shape=(75, FRAME_DIM, FRAME_DIM, 1)))  # (frames, height, width, channels)

model.add(Conv3D(64, (3,3,3), padding='same', kernel_regularizer=l2(reg_strength)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv3D(64, (3,3,3), padding='same', kernel_regularizer=l2(reg_strength)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool3D((1,2,2))) 
model.add(SpatialDropout3D(0.3))

model.add(Conv3D(128, (3,3,3), padding='same', kernel_regularizer=l2(reg_strength)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv3D(128, (3,3,3), padding='same', kernel_regularizer=l2(reg_strength)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool3D((1,2,2))) 
model.add(SpatialDropout3D(0.3))

model.add(Conv3D(256, (3,3,3), padding='same', kernel_regularizer=l2(reg_strength)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv3D(256, (3,3,3), padding='same', kernel_regularizer=l2(reg_strength)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool3D((1,2,2))) # spatial reduction
model.add(SpatialDropout3D(0.3))

# Reshape for GRU 
final_dim = FRAME_DIM // 8
model.add(Reshape((75, 256 * final_dim * final_dim)))
model.add(TimeDistributed(Dense(512, kernel_regularizer=l2(reg_strength))))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.3))

# GRU Layers (Bidirectional)
model.add(Bidirectional(GRU(256, return_sequences=True, kernel_regularizer=l2(reg_strength), recurrent_regularizer=l2(reg_strength/10))))
model.add(Dropout(0.5))
model.add(Bidirectional(GRU(256, return_sequences=True, kernel_regularizer=l2(reg_strength), recurrent_regularizer=l2(reg_strength/10))))
model.add(Dropout(0.5))

model.add(Dense(char_to_num.vocabulary_size()+1, activation='softmax', kernel_regularizer=l2(reg_strength)))
model.summary()

# You might want to try a lower learning rate with regularization
model.compile(optimizer=Adam(0.0001), loss=CTCLoss)


class ProduceExample(tf.keras.callbacks.Callback):
    def __init__(self, dataset, log_filepath='example_predictions.json', name='') -> None:
        self.dataset = dataset
        self.it = self.dataset.as_numpy_iterator()
        self.log_filepath = log_filepath
        self.name = name



    def on_epoch_end(self, epoch, logs=None) -> None:
        print(f'epoch:{epoch+1} has ended')
        data = self.it.next()
        if data[0].shape[0] < 5:
            self.it = self.dataset.as_numpy_iterator()
            data = self.it.next()
        yhat = model.predict(data[0], verbose=0)
        decoded = tf.keras.backend.ctc_decode(yhat, [75]*data[0].shape[0], greedy=True)[0][0].numpy()  # Adjust length to match
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = {
            'timestamp': timestamp,
            'epoch': epoch + 1,
            'source': self.name,
            'examples': []
        }

        print("="*20 + f" Source: {self.name} " + "="*20)
        for x in range(min(5, len(decoded))):  # Limit to available predictions
            original = tf.strings.reduce_join(num_to_char(data[1][x])).numpy().decode('utf-8')
            prediction = tf.strings.reduce_join(num_to_char(decoded[x])).numpy().decode('utf-8')
            if x == 0:
                print(f'Original   : {original}')
                print(f'Prediction : {prediction}')
                print('~' * 80)

            log_entry['examples'].append({
                'original': original,
                'prediction': prediction
            })

        # Save to JSON
        if os.path.exists(self.log_filepath):
            with open(self.log_filepath, 'r') as f:
                logs = json.load(f)
        else:
            logs = []

        logs.append(log_entry)

        with open(self.log_filepath, 'w') as f:
            json.dump(logs, f, indent=4)
import time
from IPython.display import display

class LossPlotter(tf.keras.callbacks.Callback):
    def __init__(self, interval_sec=5, save_path=f'loss_plot.png'):
        super().__init__()
        self.interval_sec = interval_sec
        self.train_losses = []
        self.val_losses = []
        self.save_path = save_path
        self.tic = None  # Will initialize on first epoch end
        self.fig = None
        self.ax = None
        self.display_handle = None

    def plot(self):
        if self.ax is None:
            self.fig, self.ax = plt.subplots()
            self.display_handle = display(self.fig, display_id=True)

        self.ax.cla()
        self.ax.plot(self.train_losses, label='Train Loss')
        self.ax.plot(self.val_losses, label='Val Loss')
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('Loss')
        self.ax.set_title('Training and Validation Loss')
        self.ax.legend()
        self.display_handle.update(self.fig)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.train_losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

        # Initialize timer after first epoch
        if self.tic is None:
            self.tic = time.time()
            self.plot()
        elif time.time() - self.tic > self.interval_sec:
            self.plot()
            self.tic = time.time()
        # save plot for every 5 iterations    
        if len(self.train_losses)%5 == 0:
            self.on_train_end()

    def on_train_end(self, logs=None):
        if self.train_losses:  # Plot final only if there's data
            self.plot()
            self.fig.savefig(self.save_path)
            print(f"\nLoss plot saved to {os.path.abspath(self.save_path)}")
HISTORY_FILE_PATH = 'training_history.json'
class SaveEpochHistoryCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.epoch_history = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        epoch_entry = {
            'timestamp': timestamp,
            'epoch': epoch,
        }
        for key, value in logs.items():
            epoch_entry[key] = float(value)
        self.epoch_history.append(epoch_entry)
        self._save_history_to_file()
    
    def _save_history_to_file(self):
        # Load existing data
        if os.path.exists(HISTORY_FILE_PATH):
            with open(HISTORY_FILE_PATH, 'r') as f:
                all_runs = json.load(f)
        else:
            all_runs = []

        # Append the latest entries
        all_runs.extend(self.epoch_history)
        self.epoch_history = []  # Clear the buffer after saving

        with open(HISTORY_FILE_PATH, 'w') as f:
            json.dump(all_runs, f, indent=4)
callbacks = [
     LossPlotter(interval_sec = 10),
    
    # Save the model in .keras format
    ModelCheckpoint("mmodel.keras", monitor="val_loss", save_best_only=True, verbose=1),

    # Save only the weights
    ModelCheckpoint("mdel_weights.weights.h5", monitor="val_loss", save_best_only=True, verbose=1, save_weights_only=True),

    # Custom callback 
    ProduceExample(val,log_filepath ='val_examples.json',name ="validation"),
    ProduceExample(data,log_filepath = 'train_examples.json',name = "train"),
    SaveEpochHistoryCallback()

]
print("\n🔍 DEBUG: Checking dataset...")
batch_count = 0
for batch in data.take(5):
    print("Batch:", batch)
    batch_count += 1

print("Total batches available:", batch_count)
exit()


train_steps = len(train)// BATCH_SIZE
val_steps = len(test) // BATCH_SIZE
history=model.fit(data,
                  epochs=50, 
                  validation_data= val,
                  steps_per_epoch = train_steps,
                  validation_steps = val_steps,
                  callbacks=callbacks,
                  verbose=2)