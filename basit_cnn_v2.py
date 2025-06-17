#!/usr/bin/env python3
"""
BASIT GPU CNN v2 – 8 ÖĞRENILEBILIR KATMAN
Data aug + BatchNorm + EarlyStopping
Hedef: ≥90-92 % test accuracy, overfitting kontrolü
"""

import os
import json
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Sonuçlar için klasör yapısını oluştur
def create_directories():
    """Gerekli klasör yapısını oluşturur"""
    directories = [
        'models/saved_models',
        'models/checkpoints',
        'results/plots',
        'results/metrics',
        'results/confusion_matrices',
        'logs/tensorboard'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    return directories

# GPU ayarları
def setup_gpu():
    """GPU ayarlarını yapılandırır"""
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"✅ GPU hazır: {gpus[0]}")
    return gpus

# Modeli oluştur
def build_model():
    """CNN modelini oluşturur"""
    model = models.Sequential([
        layers.Input((224, 224, 3)),
        # Block-1
        layers.Conv2D(32, 3, padding="same"), layers.BatchNormalization(),
        layers.Activation("relu"), layers.MaxPooling2D(2),

        # Block-2
        layers.Conv2D(64, 3, padding="same"), layers.BatchNormalization(),
        layers.Activation("relu"), layers.MaxPooling2D(2),

        # Block-3
        layers.Conv2D(128, 3, padding="same"), layers.BatchNormalization(),
        layers.Activation("relu"), layers.MaxPooling2D(2),

        # Block-4
        layers.Conv2D(256, 3, padding="same"), layers.BatchNormalization(),
        layers.Activation("relu"), layers.MaxPooling2D(2),

        # Block-5
        layers.Conv2D(256, 3, padding="same"), layers.BatchNormalization(),
        layers.Activation("relu"), layers.MaxPooling2D(2),

        # Block-6
        layers.Conv2D(512, 3, padding="same"), layers.BatchNormalization(),
        layers.Activation("relu"), layers.MaxPooling2D(2),

        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(3, activation="softmax")
    ])
    
    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy", 
                          tf.keras.metrics.Precision(name='precision'),
                          tf.keras.metrics.Recall(name='recall')])
    return model

def plot_training_history(history, timestamp):
    """Eğitim geçmişini görselleştirir ve kaydeder"""
    # Doğruluk grafiği
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
    plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
    plt.title('Model Doğruluğu')
    plt.ylabel('Doğruluk')
    plt.xlabel('Epok')
    plt.legend()
    
    # Kayıp grafiği
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Eğitim Kaybı')
    plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
    plt.title('Model Kaybı')
    plt.ylabel('Kayıp')
    plt.xlabel('Epok')
    plt.legend()
    
    plt.tight_layout()
    plot_path = f'results/plots/training_history_{timestamp}.png'
    plt.savefig(plot_path)
    plt.close()
    return plot_path

def save_confusion_matrix(y_true, y_pred, class_names, timestamp):
    """Karışıklık matrisini oluşturur ve kaydeder"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.title('Karışıklık Matrisi')
    plt.ylabel('Gerçek Etiketler')
    plt.xlabel('Tahmin Edilen Etiketler')
    
    cm_path = f'results/confusion_matrices/cm_{timestamp}.png'
    plt.savefig(cm_path)
    plt.close()
    return cm_path

def save_metrics_report(y_true, y_pred, class_names, timestamp):
    """Metrik raporunu kaydeder"""
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # Metrikleri kaydet
    metrics_path = f'results/metrics/metrics_{timestamp}.json'
    with open(metrics_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    # Okunabilir metrik raporu oluştur
    txt_report = classification_report(y_true, y_pred, target_names=class_names)
    with open(f'results/metrics/classification_report_{timestamp}.txt', 'w') as f:
        f.write(txt_report)
    
    return metrics_path, txt_report

def save_model_summary(model, timestamp):
    """Model özetini kaydeder"""
    summary = []
    model.summary(print_fn=lambda x: summary.append(x))
    summary = '\n'.join(summary)
    
    with open(f'results/model_summary_{timestamp}.txt', 'w') as f:
        f.write(summary)
    
    return summary

def basit_cnn_v2():
    print("🚀 BASIT CNN v2 EĞİTİMİ BAŞLIYOR (GPU)")
    
    # Zaman damgası oluştur
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Klasörleri oluştur
    create_directories()
    
    # GPU'yu ayarla
    setup_gpu()
    
    with tf.device("/GPU:0"):
        # Veri yükleme
        print("📁 Veri yükleniyor...")
        train_raw = tf.keras.preprocessing.image_dataset_from_directory(
            "data/splits/train", 
            image_size=(224, 224),
            batch_size=16, 
            label_mode="categorical", 
            seed=42
        )

        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            "data/splits/val", 
            image_size=(224, 224),
            batch_size=16, 
            label_mode="categorical", 
            seed=42
        )

        test_ds = tf.keras.preprocessing.image_dataset_from_directory(
            "data/splits/test", 
            image_size=(224, 224),
            batch_size=16, 
            label_mode="categorical", 
            shuffle=False, 
            seed=42
        )
        
        # Sınıf isimlerini al
        class_names = train_raw.class_names
        
        # Veri artırma
        data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ])

        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = (train_raw
                   .map(lambda x, y: (data_augmentation(x, training=True), y))
                   .cache()
                   .prefetch(AUTOTUNE))
        
        val_ds = val_ds.cache().prefetch(AUTOTUNE)
        
        # Modeli oluştur
        print("🏗️ Model oluşturuluyor...")
        model = build_model()
        
        # Callback'ler
        early_stop = EarlyStopping(
            monitor="val_accuracy",
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        checkpoint = ModelCheckpoint(
            filepath=f'models/checkpoints/best_model_{timestamp}.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
        
        tensorboard_callback = TensorBoard(
            log_dir=f'logs/tensorboard/{timestamp}',
            histogram_freq=1
        )
        
        # Eğitim
        print("🔥 EĞİTİM BAŞLADI")
        history = model.fit(
            train_ds, 
            validation_data=val_ds,
            epochs=60, 
            callbacks=[early_stop, checkpoint, tensorboard_callback],
            verbose=1
        )
        
        # Modeli kaydet
        model_path = f'models/saved_models/model_{timestamp}.h5'
        model.save(model_path)
        
        # Model özetini kaydet
        summary = save_model_summary(model, timestamp)
        
        # Test değerlendirmesi
        print("📊 Test değerlendirmesi...")
        test_loss, test_accuracy, test_precision, test_recall = model.evaluate(test_ds, verbose=0)
        
        # Tahminler
        y_true = np.concatenate([y for x, y in test_ds], axis=0)
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(model.predict(test_ds), axis=1)
        
        # Metrikleri hesapla ve kaydet
        metrics_path, txt_report = save_metrics_report(y_true, y_pred, class_names, timestamp)
        
        # Karışıklık matrisini oluştur ve kaydet
        cm_path = save_confusion_matrix(y_true, y_pred, class_names, timestamp)
        
        # Eğitim grafiklerini oluştur ve kaydet
        plot_path = plot_training_history(history, timestamp)
        
        # Sonuçları topla
        result = {
            "model": "CNN_GPU_with_Aug_BN",
            "timestamp": timestamp,
            "test_accuracy": round(test_accuracy * 100, 2),
            "test_precision": round(test_precision * 100, 2),
            "test_recall": round(test_recall * 100, 2),
            "test_loss": round(test_loss, 4),
            "epochs_trained": len(history.history["loss"]),
            "model_path": os.path.abspath(model_path),
            "metrics_path": os.path.abspath(metrics_path),
            "confusion_matrix_path": os.path.abspath(cm_path),
            "training_plot_path": os.path.abspath(plot_path),
            "class_names": class_names,
            "best_epoch": len(history.history['val_accuracy']) - 10,  # early stopping'den sonraki en iyi epoch
            "training_time": str(datetime.datetime.now() - datetime.datetime.strptime(timestamp, "%Y%m%d_%H%M%S"))
        }
        
        # Sonuçları JSON olarak kaydet
        results_path = f'results/training_results_{timestamp}.json'
        with open(results_path, 'w') as f:
            json.dump(result, f, indent=4)
        
        # Konsola özeti yazdır
        print("\n" + "="*50)
        print("� EĞİTİM TAMAMLANDI - ÖZET")
        print("="*50)
        print(f"Model kaydedildi: {model_path}")
        print(f"Test Doğruluğu: {result['test_accuracy']}%")
        print(f"Test Hassasiyeti: {result['test_precision']}%")
        print(f"Test Duyarlılık: {result['test_recall']}%")
        print(f"Eğitim Süresi: {result['training_time']}")
        print("\nSınıflandırma Raporu:")
        print(txt_report)
        print("="*50 + "\n")
        
        return result

if __name__ == "__main__":
    basit_cnn_v2()