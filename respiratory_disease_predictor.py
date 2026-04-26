"""
=============================================================================
RESPIRATORY DISEASE PREDICTION FROM VOICE/AUDIO SAMPLES
Using Coswara Dataset (COVID-19, Cough, Breathing Sounds)
=============================================================================

Dataset Sources:
1. Coswara Dataset: https://www.kaggle.com/datasets/sarabhian/coswara
2. COUGHVID: https://kaggle.com/datasets/nasrulhakim86/coughvid
3. Respiratory Sound Database: https://kaggle.com/datasets/vbookshelf/respiratory-sound-database

Author: Auto-generated ML Pipeline
Description: Audio-based respiratory disease classifier using MFCC + CNN/LSTM
=============================================================================
"""

# ============================================================
# SECTION 1: IMPORTS & DEPENDENCIES
# ============================================================
import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
import shutil
import tempfile
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, roc_auc_score)
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import (Dense, Conv2D, MaxPooling2D, Flatten,
                                      LSTM, Dropout, BatchNormalization,
                                      GlobalAveragePooling2D, Input,
                                      Bidirectional, TimeDistributed,
                                      Reshape, Conv1D, MaxPooling1D)
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                         ReduceLROnPlateau)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import joblib

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

print("✅ All libraries loaded successfully!")
print(f"TensorFlow version: {tf.__version__}")


# ============================================================
# SECTION 2: CONFIGURATION
# ============================================================
class Config:
    """Central configuration for the entire pipeline"""

    # Audio parameters
    SAMPLE_RATE       = 22050
    DURATION          = 5           # seconds per clip
    N_MFCC            = 40          # MFCC coefficients
    N_MELS            = 128         # Mel spectrogram bands
    HOP_LENGTH        = 512
    N_FFT             = 2048
    FIXED_LENGTH      = 216         # Fixed time-frames

    # Model parameters
    BATCH_SIZE        = 32
    EPOCHS            = 100
    LEARNING_RATE     = 1e-3
    DROPOUT_RATE      = 0.4
    TEST_SIZE         = 0.2
    VAL_SIZE          = 0.1
    RANDOM_STATE      = 42

    # Paths
    DATA_DIR          = "./coswara_data"
    MODEL_DIR         = "./models"
    OUTPUT_DIR        = "./outputs"
    MODEL_PATH        = "./models/respiratory_cnn_lstm.h5"
    ENCODER_PATH      = "./models/label_encoder.pkl"
    SCALER_PATH       = "./models/feature_scaler.pkl"

    # Disease classes (from Coswara + respiratory datasets)
    DISEASE_CLASSES = [
        "healthy",
        "covid19_positive",
        "asthma",
        "bronchitis",
        "pneumonia",
        "upper_respiratory_infection",
        "chronic_obstructive_pulmonary_disease"
    ]

    # Symptoms associated (for multi-label output)
    SYMPTOMS = ["fever", "cough", "breathing_difficulty",
                "fatigue", "sore_throat", "chest_pain"]

    # Create directories
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


cfg = Config()


# ============================================================
# SECTION 3: DATA DOWNLOAD & PREPARATION (Kaggle API)
# ============================================================
class DatasetDownloader:
    """
    Downloads datasets from Kaggle using the Kaggle API.
    Requires: kaggle.json credentials placed in ~/.kaggle/
    """

    @staticmethod
    def setup_kaggle():
        """Setup Kaggle API credentials"""
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_dir.mkdir(exist_ok=True)
        print("📌 To use Kaggle datasets:")
        print("   1. Go to https://www.kaggle.com/account")
        print("   2. Click 'Create New API Token' → downloads kaggle.json")
        print("   3. Place kaggle.json in ~/.kaggle/")
        print("   4. Run: pip install kaggle")

    @staticmethod
    def download_coswara():
        """Download Coswara dataset"""
        try:
            import kaggle
            print("📥 Downloading Coswara dataset...")
            kaggle.api.dataset_download_files(
                'sarabhian/coswara',
                path=cfg.DATA_DIR,
                unzip=True
            )
            print("✅ Coswara dataset downloaded!")
        except Exception as e:
            print(f"⚠️  Could not download automatically: {e}")
            print("📌 Manual download: https://www.kaggle.com/datasets/sarabhian/coswara")

    @staticmethod
    def download_respiratory_sounds():
        """Download ICBHI Respiratory Sound Database"""
        try:
            import kaggle
            print("📥 Downloading Respiratory Sound Database...")
            kaggle.api.dataset_download_files(
                'vbookshelf/respiratory-sound-database',
                path=cfg.DATA_DIR + "_respiratory",
                unzip=True
            )
            print("✅ Respiratory Sound Database downloaded!")
        except Exception as e:
            print(f"⚠️  Manual download: https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database")

    @staticmethod
    def download_coughvid():
        """Download COUGHVID dataset"""
        try:
            import kaggle
            print("📥 Downloading COUGHVID dataset...")
            kaggle.api.dataset_download_files(
                'nasrulhakim86/coughvid',
                path=cfg.DATA_DIR + "_coughvid",
                unzip=True
            )
            print("✅ COUGHVID dataset downloaded!")
        except Exception as e:
            print(f"⚠️  Manual download: https://www.kaggle.com/datasets/nasrulhakim86/coughvid")


# ============================================================
# SECTION 4: AUDIO FEATURE EXTRACTION
# ============================================================
class AudioFeatureExtractor:
    """
    Extracts rich audio features from voice/cough/breathing samples.
    Features: MFCC, Mel Spectrogram, Chroma, Spectral features, ZCR
    """

    def __init__(self, sr=cfg.SAMPLE_RATE, duration=cfg.DURATION):
        self.sr = sr
        self.duration = duration
        self.max_samples = sr * duration

    def load_audio(self, file_path):
        """Load and normalize audio file"""
        try:
            audio, sr = librosa.load(file_path, sr=self.sr, duration=self.duration)
            # Pad or trim to fixed length
            if len(audio) < self.max_samples:
                audio = np.pad(audio, (0, self.max_samples - len(audio)))
            else:
                audio = audio[:self.max_samples]
            return audio
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

    def extract_mfcc(self, audio):
        """Extract MFCC features - primary feature for speech/cough analysis"""
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sr,
                                     n_mfcc=cfg.N_MFCC,
                                     hop_length=cfg.HOP_LENGTH,
                                     n_fft=cfg.N_FFT)
        # Include delta and delta-delta for temporal dynamics
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        return mfcc, mfcc_delta, mfcc_delta2

    def extract_mel_spectrogram(self, audio):
        """Extract Mel Spectrogram - captures spectral energy distribution"""
        mel = librosa.feature.melspectrogram(y=audio, sr=self.sr,
                                              n_mels=cfg.N_MELS,
                                              hop_length=cfg.HOP_LENGTH,
                                              n_fft=cfg.N_FFT)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        return mel_db

    def extract_spectral_features(self, audio):
        """Extract spectral features for respiratory pattern analysis"""
        # Spectral centroid - brightness of sound
        centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sr)
        # Spectral bandwidth - spread around centroid
        bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sr)
        # Spectral rolloff - frequency below which 85% energy is contained
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sr)
        # Zero crossing rate - noisiness indicator
        zcr = librosa.feature.zero_crossing_rate(audio)
        # RMS energy - loudness
        rms = librosa.feature.rms(y=audio)
        # Chroma features - pitch class profiles
        chroma = librosa.feature.chroma_stft(y=audio, sr=self.sr)

        spectral = np.vstack([centroid, bandwidth, rolloff, zcr, rms, chroma])
        return spectral

    def extract_all_features_flat(self, audio):
        """Extract flattened feature vector for classical ML models"""
        features = []

        # MFCC statistics (mean + std for each coefficient)
        mfcc, mfcc_d, mfcc_d2 = self.extract_mfcc(audio)
        features.extend(np.mean(mfcc, axis=1))
        features.extend(np.std(mfcc, axis=1))
        features.extend(np.mean(mfcc_d, axis=1))
        features.extend(np.mean(mfcc_d2, axis=1))

        # Spectral statistics
        spec = self.extract_spectral_features(audio)
        features.extend(np.mean(spec, axis=1))
        features.extend(np.std(spec, axis=1))

        # Pitch features
        pitches, magnitudes = librosa.piptrack(y=audio, sr=self.sr)
        pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
        features.append(pitch_mean)

        return np.array(features)

    def extract_cnn_features(self, audio):
        """
        Extract 2D feature maps for CNN model input.
        Returns: Combined MFCC + Delta + Delta2 (3-channel image-like input)
        """
        mfcc, mfcc_d, mfcc_d2 = self.extract_mfcc(audio)

        # Pad/trim to fixed time length
        def pad_or_trim(arr, length=cfg.FIXED_LENGTH):
            if arr.shape[1] < length:
                arr = np.pad(arr, ((0, 0), (0, length - arr.shape[1])))
            else:
                arr = arr[:, :length]
            return arr

        mfcc    = pad_or_trim(mfcc)
        mfcc_d  = pad_or_trim(mfcc_d)
        mfcc_d2 = pad_or_trim(mfcc_d2)

        # Stack as 3-channel input (like RGB image)
        features_3ch = np.stack([mfcc, mfcc_d, mfcc_d2], axis=-1)
        return features_3ch  # Shape: (40, 216, 3)

    def augment_audio(self, audio):
        """Data augmentation to increase dataset size & robustness"""
        augmented = []

        # Original
        augmented.append(audio)

        # Add white noise
        noise = audio + 0.005 * np.random.randn(len(audio))
        augmented.append(noise)

        # Time stretch (slow down / speed up)
        stretched = librosa.effects.time_stretch(audio, rate=0.9)
        stretched = np.pad(stretched, (0, max(0, len(audio) - len(stretched))))[:len(audio)]
        augmented.append(stretched)

        # Pitch shift
        pitched = librosa.effects.pitch_shift(audio, sr=self.sr, n_steps=2)
        augmented.append(pitched)

        # Random gain
        gain = audio * np.random.uniform(0.7, 1.3)
        augmented.append(gain)

        return augmented


# ============================================================
# SECTION 5: DATASET LOADER
# ============================================================
class CoswaraDatasetLoader:
    """
    Loads and preprocesses the Coswara dataset.
    Coswara structure:
      - metadata.csv (patient info, health status, symptoms)
      - audio folders: breathing-deep, breathing-shallow,
                       cough-heavy, cough-shallow, vowel-a/e/o
    """

    def __init__(self, data_dir=cfg.DATA_DIR):
        self.data_dir = data_dir
        self.extractor = AudioFeatureExtractor()

    def load_coswara_metadata(self):
        """Load and parse Coswara metadata CSV"""
        meta_path = os.path.join(self.data_dir, "combined_data.csv")
        if not os.path.exists(meta_path):
            # Try alternate paths
            for fname in ["metadata.csv", "data.csv", "combined_data.csv"]:
                alt_path = os.path.join(self.data_dir, fname)
                if os.path.exists(alt_path):
                    meta_path = alt_path
                    break

        if not os.path.exists(meta_path):
            print(f"⚠️  Metadata CSV not found at {self.data_dir}")
            print("    Generating synthetic demo data...")
            return self._generate_synthetic_data()

        df = pd.read_csv(meta_path)
        print(f"📊 Loaded metadata: {df.shape[0]} samples, {df.shape[1]} columns")
        print(f"   Columns: {list(df.columns[:10])}...")
        return df

    def _generate_synthetic_data(self, n_samples=500):
        """
        Generate synthetic dataset for demonstration/testing.
        Replace with real data download for production use.
        """
        np.random.seed(cfg.RANDOM_STATE)
        classes = cfg.DISEASE_CLASSES
        data = {
            'patient_id': [f"P{i:04d}" for i in range(n_samples)],
            'label': np.random.choice(classes, n_samples),
            'age': np.random.randint(18, 80, n_samples),
            'gender': np.random.choice(['M', 'F'], n_samples),
            'fever': np.random.randint(0, 2, n_samples),
            'cough': np.random.randint(0, 2, n_samples),
            'breathing_difficulty': np.random.randint(0, 2, n_samples),
        }
        df = pd.DataFrame(data)
        print(f"✅ Generated {n_samples} synthetic samples for demo")
        print(f"   Class distribution:\n{df['label'].value_counts()}")
        return df

    def prepare_dataset(self, use_augmentation=True):
        """
        Full pipeline: load audio → extract features → build dataset
        Returns: X (features), y (labels)
        """
        df = self.load_coswara_metadata()

        X_flat = []   # For classical ML
        X_cnn  = []   # For CNN/LSTM
        y      = []

        print("\n🔄 Extracting audio features...")
        audio_col = 'audio_path' if 'audio_path' in df.columns else None
        label_col = 'label' if 'label' in df.columns else 'health_status'

        for idx, row in df.iterrows():
            label = row[label_col]

            if audio_col and os.path.exists(str(row[audio_col])):
                audio = self.extractor.load_audio(row[audio_col])
            else:
                # Generate synthetic audio signal for demo
                t = np.linspace(0, cfg.DURATION, cfg.SAMPLE_RATE * cfg.DURATION)
                freq = 200 + np.random.randint(0, 300)
                audio = 0.5 * np.sin(2 * np.pi * freq * t)
                audio += 0.1 * np.random.randn(len(audio))

            if audio is None:
                continue

            audios = [audio]
            if use_augmentation:
                audios = self.extractor.augment_audio(audio)

            for aug_audio in audios:
                flat_feat = self.extractor.extract_all_features_flat(aug_audio)
                cnn_feat  = self.extractor.extract_cnn_features(aug_audio)
                X_flat.append(flat_feat)
                X_cnn.append(cnn_feat)
                y.append(label)

            if idx % 50 == 0:
                print(f"   Processed {idx}/{len(df)} samples...")

        X_flat = np.array(X_flat)
        X_cnn  = np.array(X_cnn)
        y      = np.array(y)

        print(f"\n✅ Feature extraction complete!")
        print(f"   Flat features shape : {X_flat.shape}")
        print(f"   CNN features shape  : {X_cnn.shape}")
        print(f"   Labels shape        : {y.shape}")
        print(f"   Class distribution  : {pd.Series(y).value_counts().to_dict()}")

        return X_flat, X_cnn, y


# ============================================================
# SECTION 6: MODEL ARCHITECTURES
# ============================================================
class ModelBuilder:
    """
    Multiple model architectures for respiratory disease classification.
    1. CNN Model (image-like MFCC input)
    2. CNN + Bidirectional LSTM (temporal sequence modeling)
    3. 1D CNN (raw feature sequence)
    """

    @staticmethod
    def build_cnn_model(input_shape, num_classes):
        """
        2D CNN Model treating MFCC as image.
        Best for: capturing frequency patterns in spectrograms
        """
        inputs = Input(shape=input_shape, name="audio_input")

        # Block 1
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)

        # Block 2
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)

        # Block 3
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.4)(x)

        # Classifier
        x = Dense(256, activation='relu')(x)
        x = Dropout(cfg.DROPOUT_RATE)(x)
        x = Dense(128, activation='relu')(x)
        outputs = Dense(num_classes, activation='softmax', name="disease_output")(x)

        model = Model(inputs, outputs, name="CNN_Respiratory")
        return model

    @staticmethod
    def build_cnn_lstm_model(input_shape, num_classes):
        """
        CNN + Bidirectional LSTM Hybrid Model.
        Best for: capturing both local spectral and global temporal patterns.
        Ideal for cough/breathing sequences.
        """
        inputs = Input(shape=input_shape, name="audio_input")

        # CNN feature extractor
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)

        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 4))(x)
        x = Dropout(0.25)(x)

        # Reshape for LSTM: (batch, time_steps, features)
        shape = x.shape
        x = Reshape((shape[1], shape[2] * shape[3]))(x)

        # Bidirectional LSTM for temporal modeling
        x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.3))(x)
        x = Bidirectional(LSTM(64, dropout=0.3))(x)

        # Dense layers
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(cfg.DROPOUT_RATE)(x)
        x = Dense(128, activation='relu')(x)

        # Disease classification output
        disease_out = Dense(num_classes, activation='softmax',
                            name="disease_output")(x)

        model = Model(inputs, disease_out, name="CNN_BiLSTM_Respiratory")
        return model

    @staticmethod
    def build_1d_cnn_model(input_shape, num_classes):
        """
        1D CNN for flat feature vectors.
        Best for: quick training, lower compute requirements
        """
        inputs = Input(shape=input_shape, name="feature_input")
        x = Reshape((input_shape[0], 1))(inputs)

        x = Conv1D(64, 3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(2)(x)

        x = Conv1D(128, 3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(2)(x)

        x = Conv1D(256, 3, activation='relu', padding='same')(x)
        x = GlobalAveragePooling1D()(x)

        x = Dense(256, activation='relu')(x)
        x = Dropout(0.4)(x)
        x = Dense(128, activation='relu')(x)
        outputs = Dense(num_classes, activation='softmax')(x)

        model = Model(inputs, outputs, name="1DCNN_Respiratory")
        return model


# ============================================================
# SECTION 7: TRAINING PIPELINE
# ============================================================
class RespiratoryDiseaseTrainer:
    """Full training pipeline with cross-validation and evaluation"""

    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.model = None
        self.history = None

    def prepare_labels(self, y):
        """Encode string labels to integers"""
        y_encoded = self.label_encoder.fit_transform(y)
        num_classes = len(self.label_encoder.classes_)
        y_cat = to_categorical(y_encoded, num_classes)
        print(f"📌 Classes: {list(self.label_encoder.classes_)}")
        print(f"📌 Number of classes: {num_classes}")
        return y_cat, num_classes

    def compute_class_weights(self, y_encoded):
        """Handle imbalanced classes"""
        classes = np.unique(y_encoded)
        weights = compute_class_weight('balanced', classes=classes, y=y_encoded)
        return dict(zip(classes, weights))

    def build_and_compile(self, input_shape, num_classes, model_type='cnn_lstm'):
        """Build and compile the selected model"""
        if model_type == 'cnn':
            self.model = ModelBuilder.build_cnn_model(input_shape, num_classes)
        elif model_type == 'cnn_lstm':
            self.model = ModelBuilder.build_cnn_lstm_model(input_shape, num_classes)
        elif model_type == '1d_cnn':
            self.model = ModelBuilder.build_1d_cnn_model(input_shape, num_classes)

        self.model.compile(
            optimizer=Adam(learning_rate=cfg.LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy',
                     tf.keras.metrics.AUC(name='auc'),
                     tf.keras.metrics.Precision(name='precision'),
                     tf.keras.metrics.Recall(name='recall')]
        )
        self.model.summary()
        return self.model

    def get_callbacks(self):
        """Training callbacks for monitoring and optimization"""
        return [
            EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=cfg.MODEL_PATH,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-6,
                verbose=1
            )
        ]

    def train(self, X, y_raw, model_type='cnn_lstm'):
        """Full training loop"""
        print("\n" + "="*60)
        print("🚀 STARTING TRAINING PIPELINE")
        print("="*60)

        # Encode labels
        y_cat, num_classes = self.prepare_labels(y_raw)
        y_encoded = self.label_encoder.transform(y_raw)

        # Train/test split
        X_train, X_test, y_train, y_test, ye_train, ye_test = train_test_split(
            X, y_cat, y_encoded,
            test_size=cfg.TEST_SIZE,
            random_state=cfg.RANDOM_STATE,
            stratify=y_encoded
        )

        # Further split validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=cfg.VAL_SIZE,
            random_state=cfg.RANDOM_STATE
        )

        print(f"\n📊 Dataset split:")
        print(f"   Train: {X_train.shape[0]} samples")
        print(f"   Val  : {X_val.shape[0]} samples")
        print(f"   Test : {X_test.shape[0]} samples")

        # Build model
        input_shape = X.shape[1:]
        self.build_and_compile(input_shape, num_classes, model_type)

        # Class weights for imbalanced data
        class_weights = self.compute_class_weights(ye_train)
        print(f"\n⚖️  Class weights: {class_weights}")

        # Train
        print("\n🏋️  Training model...")
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=cfg.BATCH_SIZE,
            epochs=cfg.EPOCHS,
            validation_data=(X_val, y_val),
            class_weight=class_weights,
            callbacks=self.get_callbacks(),
            verbose=1
        )

        # Evaluate
        print("\n📈 Evaluating on test set...")
        self.evaluate(X_test, y_test, ye_test)

        # Save artifacts
        self.save_artifacts()

        return self.history

    def evaluate(self, X_test, y_test_cat, y_test_encoded):
        """Comprehensive model evaluation"""
        loss, acc, auc, prec, rec = self.model.evaluate(X_test, y_test_cat, verbose=0)
        print(f"\n{'='*50}")
        print(f"TEST RESULTS:")
        print(f"  Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
        print(f"  AUC       : {auc:.4f}")
        print(f"  Precision : {prec:.4f}")
        print(f"  Recall    : {rec:.4f}")
        print(f"{'='*50}")

        # Per-class report
        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        class_names = list(self.label_encoder.classes_)

        print("\nClassification Report:")
        print(classification_report(y_test_encoded, y_pred,
                                    target_names=class_names))

        # Confusion matrix
        self.plot_confusion_matrix(y_test_encoded, y_pred, class_names)
        self.plot_training_history()

        return acc, auc

    def plot_confusion_matrix(self, y_true, y_pred, class_names):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.title('Confusion Matrix - Respiratory Disease Prediction', fontsize=14)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.OUTPUT_DIR, 'confusion_matrix.png'), dpi=150)
        print(f"✅ Confusion matrix saved to {cfg.OUTPUT_DIR}/confusion_matrix.png")

    def plot_training_history(self):
        """Plot training/validation curves"""
        if self.history is None:
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Accuracy
        axes[0].plot(self.history.history['accuracy'], label='Train Accuracy')
        axes[0].plot(self.history.history['val_accuracy'], label='Val Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Loss
        axes[1].plot(self.history.history['loss'], label='Train Loss')
        axes[1].plot(self.history.history['val_loss'], label='Val Loss')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.suptitle('CNN-BiLSTM Training History', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.OUTPUT_DIR, 'training_history.png'), dpi=150)
        print(f"✅ Training history saved to {cfg.OUTPUT_DIR}/training_history.png")

    def save_artifacts(self):
        """Save model, encoder, and scaler"""
        self.model.save(cfg.MODEL_PATH)
        joblib.dump(self.label_encoder, cfg.ENCODER_PATH)
        joblib.dump(self.scaler, cfg.SCALER_PATH)
        print(f"\n💾 Model saved    : {cfg.MODEL_PATH}")
        print(f"💾 Encoder saved  : {cfg.ENCODER_PATH}")
        print(f"💾 Scaler saved   : {cfg.SCALER_PATH}")


# ============================================================
# SECTION 8: INFERENCE / PREDICTION ENGINE
# ============================================================
class RespiratoryDiseasePredictor:
    """
    Real-time prediction from uploaded voice/cough audio samples.
    Supports: .wav, .mp3, .ogg, .flac, .m4a
    """

    def __init__(self, model_path=cfg.MODEL_PATH,
                 encoder_path=cfg.ENCODER_PATH):
        self.extractor = AudioFeatureExtractor()
        self.model = None
        self.label_encoder = None
        self._load_artifacts(model_path, encoder_path)

    def _load_artifacts(self, model_path, encoder_path):
        """Load saved model and encoder"""
        # Ensure absolute paths
        model_path = os.path.abspath(model_path)
        encoder_path = os.path.abspath(encoder_path)
        
        try:
            print(f"Loading model from: {model_path}")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
                
            self.model = load_model(model_path)
            
            print(f"Loading encoder from: {encoder_path}")
            if not os.path.exists(encoder_path):
                raise FileNotFoundError(f"Encoder file not found at {encoder_path}")
                
            self.label_encoder = joblib.load(encoder_path)
            print(f"Model and encoder loaded successfully.")
        except Exception as e:
            print(f"Error loading artifacts: {e}")
            self.model = None
            self.label_encoder = None

    def predict(self, audio_path):
        """
        Predict respiratory disease from audio file.

        Args:
            audio_path (str): Path to audio file (.wav, .mp3, .ogg, etc.)

        Returns:
            dict: Prediction results with disease, confidence, and symptoms
        """
        if self.model is None or self.label_encoder is None:
            return {"error": "Model files not loaded. Please ensure models/respiratory_cnn_lstm.h5 and models/label_encoder.pkl exist."}

        if not os.path.exists(audio_path):
            return {"error": f"File not found: {audio_path}"}

        print(f"Analyzing audio: {os.path.basename(audio_path)}")
        print("Processing...")

        # Load audio
        audio = self.extractor.load_audio(audio_path)
        if audio is None:
            return {"error": "Failed to load audio file"}

        # Extract features
        features = self.extractor.extract_cnn_features(audio)
        features = np.expand_dims(features, axis=0)  # Add batch dim

        # Predict
        probabilities = self.model.predict(features, verbose=0)[0]
        class_names = list(self.label_encoder.classes_)

        # Sort by confidence
        sorted_idx = np.argsort(probabilities)[::-1]
        top_predictions = [
            {
                "disease": class_names[i],
                "confidence": float(probabilities[i]),
                "percentage": f"{probabilities[i]*100:.1f}%"
            }
            for i in sorted_idx
        ]

        # Primary prediction
        primary = top_predictions[0]
        symptoms = self._infer_symptoms(primary['disease'], probabilities)

        result = {
            "primary_diagnosis": primary['disease'].replace('_', ' ').title(),
            "confidence": primary['confidence'],
            "confidence_pct": primary['percentage'],
            "top_3_predictions": top_predictions[:3],
            "inferred_symptoms": symptoms,
            "recommendation": self._get_recommendation(primary['disease'],
                                                        primary['confidence']),
            "audio_file": os.path.basename(audio_path)
        }

        self._print_result(result)
        return result

    def _infer_symptoms(self, disease, probabilities):
        """Infer likely symptoms based on predicted disease"""
        symptom_map = {
            "covid19_positive":    ["fever", "cough", "breathing_difficulty", "fatigue"],
            "asthma":              ["wheezing", "breathing_difficulty", "chest_tightness"],
            "bronchitis":          ["persistent_cough", "mucus", "fatigue", "fever"],
            "pneumonia":           ["fever", "cough", "chest_pain", "breathing_difficulty"],
            "upper_respiratory_infection": ["sore_throat", "runny_nose", "cough", "mild_fever"],
            "chronic_obstructive_pulmonary_disease": ["chronic_cough", "breathing_difficulty", "wheezing"],
            "healthy":             []
        }
        return symptom_map.get(disease, ["Please consult a doctor"])

    def _get_recommendation(self, disease, confidence):
        """Generate medical recommendation based on prediction"""
        if confidence < 0.5:
            return "⚠️  Low confidence. Please consult a healthcare provider for proper diagnosis."

        recommendations = {
            "covid19_positive": "🔴 HIGH PRIORITY: Isolate immediately and get a PCR/Antigen test. Contact your doctor.",
            "asthma":           "🟡 Use prescribed inhaler. Avoid triggers. See a pulmonologist if symptoms worsen.",
            "bronchitis":       "🟡 Rest, stay hydrated. See a doctor if fever persists > 3 days.",
            "pneumonia":        "🔴 HIGH PRIORITY: Seek immediate medical attention. May require antibiotics.",
            "upper_respiratory_infection": "🟢 Rest and hydrate. OTC medications may help. See doctor if no improvement in 7 days.",
            "chronic_obstructive_pulmonary_disease": "🔴 See a pulmonologist immediately for spirometry and treatment plan.",
            "healthy":          "🟢 No respiratory issues detected. Maintain good health practices."
        }
        return recommendations.get(disease, "Consult a healthcare professional.")

    def _print_result(self, result):
        """Pretty print prediction results"""
        print("\n" + "="*55)
        print("  🏥 RESPIRATORY DISEASE PREDICTION RESULTS")
        print("="*55)
        print(f"  📁 File      : {result['audio_file']}")
        print(f"  🔬 Diagnosis : {result['primary_diagnosis']}")
        print(f"  📊 Confidence: {result['confidence_pct']}")
        print(f"\n  📋 Top Predictions:")
        for i, pred in enumerate(result['top_3_predictions'], 1):
            bar = "█" * int(pred['confidence'] * 20)
            print(f"     {i}. {pred['disease']:<40} {pred['percentage']:>6}  {bar}")
        print(f"\n  🩺 Likely Symptoms: {', '.join(result['inferred_symptoms'])}")
        print(f"\n  💊 Recommendation:")
        print(f"     {result['recommendation']}")
        print("="*55)

    def predict_from_microphone(self, duration=5):
        """Record from microphone and predict (requires pyaudio)"""
        try:
            import sounddevice as sd
            import soundfile as sf

            print(f"🎙️  Recording for {duration} seconds... Speak/Cough now!")
            audio_data = sd.rec(int(duration * cfg.SAMPLE_RATE),
                                samplerate=cfg.SAMPLE_RATE,
                                channels=1, dtype='float32')
            sd.wait()
            temp_path = "/tmp/temp_recording.wav"
            sf.write(temp_path, audio_data, cfg.SAMPLE_RATE)
            print("✅ Recording complete!")
            return self.predict(temp_path)
        except ImportError:
            print("⚠️  Install sounddevice: pip install sounddevice soundfile")


# ============================================================
# SECTION 9: WEB API (FastAPI)
# ============================================================
def create_fastapi_app():
    """
    Create FastAPI REST endpoint for the prediction service.
    Usage: uvicorn respiratory_disease_predictor:app --reload
    """
    try:
        from fastapi import FastAPI, UploadFile, File, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        import uvicorn
        import shutil
        import tempfile

        app = FastAPI(
            title="Respiratory Disease Predictor API",
            description="AI-powered respiratory disease prediction from voice samples",
            version="1.0.0"
        )

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"]
        )

        predictor = RespiratoryDiseasePredictor()

        @app.get("/")
        async def root():
            return {"message": "Respiratory Disease Predictor API",
                    "endpoints": ["/predict", "/health"]}

        @app.post("/predict")
        async def predict_disease(file: UploadFile = File(...)):
            """
            Upload an audio file (.wav, .mp3, .ogg) and get disease prediction.
            Returns: diagnosis, confidence, symptoms, and recommendation.
            """
            allowed_types = ['audio/wav', 'audio/mp3', 'audio/mpeg',
                             'audio/ogg', 'audio/flac']
            if file.content_type not in allowed_types:
                raise HTTPException(400, f"Unsupported file type: {file.content_type}")

            # Save temp file
            suffix = os.path.splitext(file.filename)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                shutil.copyfileobj(file.file, tmp)
                tmp_path = tmp.name

            try:
                result = predictor.predict(tmp_path)
                return result
            finally:
                os.unlink(tmp_path)

        @app.get("/health")
        async def health_check():
            return {"status": "healthy", "model_loaded": predictor.model is not None}

        return app

    except ImportError:
        print("⚠️  FastAPI not installed. Run: pip install fastapi uvicorn python-multipart")
        return None


# ============================================================
# SECTION 10: STREAMLIT WEB UI
# ============================================================
STREAMLIT_APP = '''
# streamlit_app.py - Run: streamlit run streamlit_app.py

import streamlit as st
import numpy as np
import os
import tempfile
from respiratory_disease_predictor import RespiratoryDiseasePredictor

st.set_page_config(
    page_title="Respiratory Disease Predictor",
    page_icon="🫁",
    layout="wide"
)

# Header
st.title("🫁 Respiratory Disease Predictor")
st.markdown("""
**AI-powered analysis of voice/cough/breathing audio samples**
*Based on Coswara Dataset | CNN + BiLSTM Model*
""")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.info("""
    This tool analyzes audio recordings of:
    - Cough sounds
    - Breathing patterns
    - Voice samples

    to predict respiratory conditions including:
    COVID-19, Asthma, Bronchitis,
    Pneumonia, URTI, and COPD.
    """)
    st.warning("⚠️ Not a substitute for medical diagnosis!")

# Load predictor
@st.cache_resource
def load_predictor():
    return RespiratoryDiseasePredictor()

predictor = load_predictor()

# File uploader
col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("📁 Upload Audio Sample")
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'ogg', 'flac', 'm4a'],
        help="Upload a voice/cough/breathing recording"
    )

    if uploaded_file:
        st.audio(uploaded_file, format='audio/wav')
        st.success(f"✅ Uploaded: {uploaded_file.name}")

        if st.button("🔬 Analyze", type="primary", use_container_width=True):
            with st.spinner("Analyzing audio..."):
                # Save temp
                suffix = os.path.splitext(uploaded_file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                result = predictor.predict(tmp_path)
                os.unlink(tmp_path)

                # Store in session state
                st.session_state['result'] = result

with col2:
    if 'result' in st.session_state:
        result = st.session_state['result']
        st.subheader("🏥 Prediction Results")

        # Primary diagnosis
        disease = result['primary_diagnosis']
        confidence = result['confidence']
        color = "red" if confidence > 0.8 else "orange" if confidence > 0.5 else "gray"
        st.markdown(f"### Diagnosis: :{color}[{disease}]")
        st.metric("Confidence", result['confidence_pct'])

        # Top predictions bar chart
        import pandas as pd
        top3 = result['top_3_predictions']
        df = pd.DataFrame(top3)
        df['confidence_pct_val'] = df['confidence'] * 100
        st.bar_chart(df.set_index('disease')['confidence_pct_val'])

        # Symptoms
        st.subheader("🩺 Likely Symptoms")
        for sym in result['inferred_symptoms']:
            st.markdown(f"- {sym.replace('_', ' ').title()}")

        # Recommendation
        st.subheader("💊 Recommendation")
        st.info(result['recommendation'])

        st.caption("*Disclaimer: This is an AI tool for screening only. Always consult a qualified medical professional.*")
'''

# Save Streamlit app
with open("streamlit_app.py", "w", encoding="utf-8") as f:
    f.write(STREAMLIT_APP)
print("✅ Streamlit app saved to streamlit_app.py")


# ============================================================
# SECTION 12: FASTAPI BACKEND
# ============================================================
def create_fastapi_app():
    print("🚀 Initializing FastAPI App...")
    app = FastAPI(title="Respiratory Disease Prediction API")

    # Enable CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    print("⏳ Loading AI Model (this may take 15-20 seconds)...")
    try:
        predictor = RespiratoryDiseasePredictor()
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        predictor = None

    @app.post("/predict")
    async def predict_audio(file: UploadFile = File(...)):
        if not predictor:
            raise HTTPException(status_code=500, detail="Model not initialized")
            
        print(f"📥 Received prediction request for: {file.filename}")
        # Save uploaded file temporarily
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        
        try:
            print(f"🔬 Running inference on: {tmp_path}")
            result = predictor.predict(tmp_path)
            os.unlink(tmp_path)
            if "error" in result:
                print(f"⚠️ Prediction error: {result['error']}")
                raise HTTPException(status_code=400, detail=result["error"])
            print("📤 Prediction complete!")
            return result
        except Exception as e:
            print(f"🔥 Server error during prediction: {e}")
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/")
    async def read_index():
        print("🏠 Serving homepage")
        return FileResponse("respiratory_predictor_website.html")

    # Serve static assets (css, js, etc.) from current directory
    app.mount("/", StaticFiles(directory="."), name="static")
    
    return app

if __name__ == "__main__":
    # If run directly as a script, start the API
    app = create_fastapi_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
