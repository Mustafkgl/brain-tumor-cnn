#!/usr/bin/env python3
"""
BASIT GPU CNN - TEK DOSYA
Direkt çalışacak, karmaşık işler yok
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
from tensorflow.keras import layers, models
import json, datetime

# GPU zorla
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print(f"✅ GPU hazır: {gpus[0]}")

def basit_cnn():
    print("🚀 BASIT CNN EĞİTİMİ BAŞLIYOR (GPU)")
    
    with tf.device('/GPU:0'):
        # Veri yükle
        print("📁 Veri yükleniyor...")
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            'data/splits/train',
            image_size=(224, 224),
            batch_size=16,
            label_mode='categorical',
            seed=42
        )
        
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            'data/splits/val',
            image_size=(224, 224),
            batch_size=16,
            label_mode='categorical',
            seed=42
        )
        
        test_ds = tf.keras.preprocessing.image_dataset_from_directory(
            'data/splits/test',
            image_size=(224, 224),
            batch_size=16,
            label_mode='categorical',
            shuffle=False,
            seed=42
        )
        
        # Cache ve prefetch
        train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)
        
        # Model oluştur
        print("🏗️ Model oluşturuluyor...")
        model = models.Sequential([
            layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=(224, 224, 3)),
            layers.MaxPooling2D(2),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(2),
            layers.Conv2D(128, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(2),
            layers.Conv2D(128, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(2),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(3, activation='softmax')  # 3 sınıf
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("🔥 GPU EĞİTİMİ BAŞLADI!")
        print("=" * 60)
        
        # Eğit
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=10,  # 10 epoch
            verbose=1
        )
        
        print("=" * 60)
        print("📊 Test değerlendirmesi...")
        
        # Test
        test_loss, test_acc = model.evaluate(test_ds, verbose=0)
        
        # Score hesapla (accuracy - katman sayısı 7'den sonraki)
        katman_sayisi = len([l for l in model.layers if 'conv2d' in l.name or 'dense' in l.name])
        extra_katman = max(0, katman_sayisi - 7)
        score = round(test_acc * 100 - extra_katman, 2)
        
        # Sonuç
        result = {
            "model": "CNN_GPU",
            "test_accuracy": round(test_acc * 100, 2),
            "katman_sayisi": katman_sayisi,
            "extra_katman": extra_katman,
            "final_score": score,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Kaydet
        os.makedirs('models', exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f'models/cnn_gpu_{ts}.h5'
        model.save(model_path)
        
        # JSON kaydet
        with open(f'models/cnn_gpu_{ts}.json', 'w') as f:
            json.dump(result, f, indent=2)
        
        print("🎉 EĞİTİM TAMAMLANDI!")
        print("=" * 60)
        print(json.dumps(result, indent=2))
        print("=" * 60)
        print(f"Model kaydedildi: {model_path}")

if __name__ == "__main__":
    basit_cnn()