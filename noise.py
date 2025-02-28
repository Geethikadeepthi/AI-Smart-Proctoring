import os
import warnings
import numpy as np
import pyaudio
import winsound
import ctypes
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
import librosa
from scipy.ndimage import zoom
import noisereduce as nr
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping
from joblib import Parallel, delayed
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import threading

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Constants for beep
frequency = 2500  # Beep frequency
duration = 500    # Beep duration in milliseconds

# Function to display a message box
def show_message_box(title, message):
    ctypes.windll.user32.MessageBoxW(0, message, title, 0x40 | 0x1000)  # 0x40 is the icon for information, 0x1000 makes it top-most

# Function to resize the spectrogram
def resize_spectrogram(spectrogram, target_shape=(64, 64)):
    """
    Resize the spectrogram to the target shape.
    """
    # Calculate zoom factors
    zoom_factors = (
        target_shape[0] / spectrogram.shape[0],
        target_shape[1] / spectrogram.shape[1],
    )
    
    # Resize the spectrogram
    resized_spectrogram = zoom(spectrogram, zoom_factors)
    
    # Add channel dimension
    resized_spectrogram = np.expand_dims(resized_spectrogram, axis=-1)
    
    return resized_spectrogram

# Function to convert audio to spectrogram
def audio_to_spectrogram(audio_data, target_shape=(64, 64)):
    """
    Convert audio data to a spectrogram and resize it to the target shape.
    """
    # Resample audio to 16 kHz (matching training preprocessing)
    audio_data = librosa.resample(audio_data.astype(np.float32), orig_sr=44100, target_sr=16000)
    
    # Normalize audio to [-1.0, 1.0]
    audio_data = audio_data / np.max(np.abs(audio_data))
    
    # Reduce noise
    audio_data = nr.reduce_noise(y=audio_data, sr=16000)
    
    # Generate Mel-spectrogram (matching training preprocessing)
    mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=16000)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    # Resize spectrogram to target shape
    resized_spectrogram = zoom(mel_spectrogram_db, 
                              (target_shape[0] / mel_spectrogram_db.shape[0], 
                               target_shape[1] / mel_spectrogram_db.shape[1]))
    
    # Add channel dimension
    resized_spectrogram = np.expand_dims(resized_spectrogram, axis=-1)
    
    return resized_spectrogram

# Split Audio into Chunks
def split_audio(audio_path, window_size=3, sr=16000):
    """Split audio into fixed-length windows (e.g., 3 seconds)."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"The file {audio_path} does not exist. Please check the path.")
    
    audio, sr = librosa.load(audio_path, sr=sr)
    samples_per_window = window_size * sr
    chunks = []
    
    for start in range(0, len(audio), samples_per_window):
        end = start + samples_per_window
        chunk = audio[start:end]
        
        # Pad the last chunk if it's shorter than window_size
        if len(chunk) < samples_per_window:
            chunk = np.pad(chunk, (0, samples_per_window - len(chunk)))
            
        chunks.append(chunk)
    
    return chunks

#Preprocess Audio Chunks
def preprocess_chunk(chunk, target_shape=(64, 64)):
    """Preprocess a single audio chunk into a spectrogram."""
    try:
        # Reduce noise (optional)
        chunk = nr.reduce_noise(y=chunk, sr=16000)
        
        # Generate Mel-spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=16000, n_mels=64)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        # Resize spectrogram to target shape
        resized_spectrogram = zoom(mel_spectrogram_db, 
                                  (target_shape[0] / mel_spectrogram_db.shape[0], 
                                   target_shape[1] / mel_spectrogram_db.shape[1]))
        
        # Add channel dimension
        resized_spectrogram = np.expand_dims(resized_spectrogram, axis=-1)
        
        # Ensure the shape is correct
        if resized_spectrogram.shape != target_shape + (1,):
            raise ValueError(f"Invalid spectrogram shape: {resized_spectrogram.shape}")
        
        return resized_spectrogram
    except Exception as e:
        print(f"Error processing chunk: {e}")
        return None

#Load Dataset and Preprocess
def load_dataset(dataset_path):
    """Load audio files from 'noise' and 'speech' folders and preprocess."""
    noise_folder = os.path.join(dataset_path, "noise")
    speech_folder = os.path.join(dataset_path, "speech")
    
    # Check if folders exist
    if not os.path.exists(noise_folder):
        raise FileNotFoundError(f"The folder {noise_folder} does not exist.")
    if not os.path.exists(speech_folder):
        raise FileNotFoundError(f"The folder {speech_folder} does not exist.")
    
    # Get list of files
    noise_files = [os.path.join(noise_folder, f) for f in os.listdir(noise_folder) if f.endswith(".wav")]
    speech_files = [os.path.join(speech_folder, f) for f in os.listdir(speech_folder) if f.endswith(".flac")]
    
    # Check if folders contain valid files
    if not noise_files:
        raise ValueError(f"No .wav files found in the noise folder: {noise_folder}")
    if not speech_files:
        raise ValueError(f"No .flac files found in the speech folder: {speech_folder}")
    
    X = []
    y = []
    
    # Load noise files (label = 0)
    for audio_path in noise_files:
        chunks = split_audio(audio_path)
        labels = [0] * len(chunks)  # Assign label 0 for noise
        spectrograms = Parallel(n_jobs=-1)(delayed(preprocess_chunk)(chunk) for chunk in chunks)
        spectrograms = [spec for spec in spectrograms if spec is not None]  # Filter out None values
        X.extend(spectrograms)
        y.extend(labels[:len(spectrograms)])  # Ensure labels match the number of valid spectrograms
    
    # Load speech files (label = 1)
    for audio_path in speech_files:
        chunks = split_audio(audio_path)
        labels = [1] * len(chunks)  # Assign label 1 for speech
        spectrograms = Parallel(n_jobs=-1)(delayed(preprocess_chunk)(chunk) for chunk in chunks)
        spectrograms = [spec for spec in spectrograms if spec is not None]  # Filter out None values
        X.extend(spectrograms)
        y.extend(labels[:len(spectrograms)])  # Ensure labels match the number of valid spectrograms
    
    # Convert lists to NumPy arrays
    X = np.array(X)
    y = np.array(y)
    
    print(f"Loaded {len(X)} samples.")  # Debug statement
    print(f"Shape of X: {X.shape}")  # Debug statement
    print(f"Shape of y: {y.shape}")  # Debug statement
    
    return X, y

#Build CNN Model
def build_cnn_model(input_shape=(64, 64, 1)):
    """Build a CNN model for speech detection."""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Binary output (0 = noise, 1 = speech)
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

#Train the Model
def train_model(X, y):
    """Train the CNN model."""
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Calculate class weights
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(class_weights))
    
    # Build the model
    model = build_cnn_model()
    model.summary()
    
    # Define early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=32,  # Increased batch size
        class_weight=class_weights,
        callbacks=[early_stopping]
    )
    
    return model

#Detect Speech in New Audio
def detect_speech(audio_path, model, threshold=0.7):
    """Detect speech segments in an audio file."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"The file {audio_path} does not exist. Please check the path.")
    
    chunks = split_audio(audio_path)
    spectrograms = []

    # Preprocess all chunks
    for chunk in chunks:
        spectrogram = preprocess_chunk(chunk)
        if spectrogram is not None:
            spectrograms.append(spectrogram)

    # Batch all spectrograms together
    if spectrograms:
        spectrograms = np.array(spectrograms)
        predictions = model.predict(spectrograms, verbose=0)  # Suppress output
        return predictions > threshold
    else:
        return [False] * len(chunks)  # Assume noise if no valid spectrograms

# NoiseDetector Class
class NoiseDetector:
    def __init__(self, threshold=0.7):
    
        self.threshold = threshold  # Threshold for speech detection
        self.running = False  # Flag to control the detection loop
        self.latest_alert = False  # Track recent alerts
        self.model = load_model('speech_detection_cnn.h5')  # Load the trained model


    def preprocess_chunk(self, chunk, target_shape=(64, 64)):
        
        try:
            # Reduce noise (optional)
            chunk = nr.reduce_noise(y=chunk, sr=16000)
            
            # Generate Mel-spectrogram
            mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=16000, n_mels=64)
            mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
            
            # Resize spectrogram to target shape
            resized_spectrogram = zoom(mel_spectrogram_db, 
                                      (target_shape[0] / mel_spectrogram_db.shape[0], 
                                       target_shape[1] / mel_spectrogram_db.shape[1]))
            
            # Add channel dimension
            resized_spectrogram = np.expand_dims(resized_spectrogram, axis=-1)
            
            # Ensure the shape is correct
            if resized_spectrogram.shape != target_shape + (1,):
                raise ValueError(f"Invalid spectrogram shape: {resized_spectrogram.shape}")
            
            return resized_spectrogram
        except Exception as e:
            print(f"Error processing chunk: {e}")
            return None

    def detect(self):
        """
        Real-time speech detection using the trained model.
        
        Returns:
            bool: True if speech is detected, False otherwise.
        """
        # Audio Configuration
        CHUNK = 1024  # Number of audio samples per chunk
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100  # Sampling rate

        # Initialize PyAudio
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        self.running = True
        print("Listening for speech...")

        while self.running:
            try:
                # Read audio data
                data = stream.read(CHUNK)
                audio_data = np.frombuffer(data, dtype=np.int16)

                # Convert audio data to floating-point format
                audio_data = audio_data.astype(np.float32) / 32768.0  # Normalize to [-1.0, 1.0]
                
                # Convert audio data to spectrogram
                spectrogram = self.preprocess_chunk(audio_data, target_shape=(64, 64))
                if spectrogram is None:
                    continue

                spectrogram = np.expand_dims(spectrogram, axis=0)  # Add batch dimension

                # Predict using CNN
                prediction = self.model.predict(spectrogram)
                is_speech = prediction[0][0] > self.threshold  # Threshold for speech detection

                # Trigger alert if speech is detected
                if is_speech:
                    print("Speech detected!")
                    self.latest_alert = True  # Set flag instead of immediate
                    return True  # Return True if speech is detected

            except Exception as e:
                print(f"Error during detection: {e}")
                break

        print("Stopping noise detection...")
        stream.stop_stream()
        stream.close()
        p.terminate()
        return False  # Return False if no speech is detected

    def stop(self):
        """Stop the noise detection loop."""
        self.running = False

# Main Workflow
if __name__ == "__main__":
    # Load and preprocess dataset
    dataset_path = r"C:\Users\GEETHIKA\Downloads\MSU\noise_train"
    X, y = load_dataset(dataset_path)
    
    # Train the model
    model = train_model(X, y)
    
    # Save the trained model
    model.save('speech_detection_cnn.h5')
    
    # Start real-time speech detection
    noise_detector = NoiseDetector(threshold=0.7)
    noise_detector.detect()