#!/bin/bash

echo "========================================================"
echo "BRAIN TUMOR CNN PROJESİ - YAPISAL ANALİZ"
echo "========================================================"

# Genel dizin yapısı
echo ""
echo "🗂️  GENEL DİZİN YAPISI:"
echo "--------"
if command -v tree &> /dev/null; then
    tree -L 3 -a
else
    find . -type d | head -20 | sort
fi

echo ""
echo "========================================================"
echo "📊 DOSYA İSTATİSTİKLERİ:"
echo "--------"

# Python dosyaları
echo "🐍 Python dosyaları:"
find . -name "*.py" -type f | wc -l
find . -name "*.py" -type f | head -10

# Veri dosyaları
echo ""
echo "📸 Görüntü dosyaları:"
echo "JPG: $(find . -name "*.jpg" -type f | wc -l)"
echo "PNG: $(find . -name "*.png" -type f | wc -l)"

# Model dosyaları
echo ""
echo "🤖 Model dosyaları:"
echo "H5: $(find . -name "*.h5" -type f | wc -l)"
echo "JSON: $(find . -name "*.json" -type f | wc -l)"

echo ""
echo "========================================================"
echo "📁 VERİ YAPISI DETAYI:"
echo "--------"

# Raw veri sayımı
if [ -d "data/raw" ]; then
    echo "Raw veri:"
    for dir in data/raw/*/; do
        if [ -d "$dir" ]; then
            count=$(ls "$dir" 2>/dev/null | wc -l)
            echo "  $(basename "$dir"): $count dosya"
        fi
    done
fi

# Splits veri sayımı
if [ -d "data/splits" ]; then
    echo ""
    echo "Splits veri:"
    for split in train val test; do
        echo "  $split:"
        if [ -d "data/splits/$split" ]; then
            for cls in data/splits/$split/*/; do
                if [ -d "$cls" ]; then
                    count=$(ls "$cls" 2>/dev/null | wc -l)
                    echo "    $(basename "$cls"): $count dosya"
                fi
            done
        fi
    done
fi

# Embeddings
if [ -d "data/embeddings" ]; then
    echo ""
    echo "Embeddings:"
    embedding_count=$(find data/embeddings -name "*.npy" 2>/dev/null | wc -l)
    echo "  NPY dosyaları: $embedding_count"
fi

echo ""
echo "========================================================"
echo "📋 PYTHON DOSYALARI İÇERİKLERİ:"
echo "--------"

# Ana Python dosyalarının ilk birkaç satırı
for file in *.py; do
    if [ -f "$file" ]; then
        echo ""
        echo "📄 $file (ilk 10 satır):"
        echo "---"
        head -10 "$file"
        echo "... (toplam $(wc -l < "$file") satır)"
    fi
done

echo ""
echo "========================================================"
echo "📊 MODELS KLASÖRÜndeki SONUÇLAR:"
echo "--------"

if [ -d "models" ]; then
    echo "Model dosyaları:"
    ls -la models/ 2>/dev/null
    
    echo ""
    echo "Son JSON sonuçları:"
    for json_file in models/*.json; do
        if [ -f "$json_file" ]; then
            echo ""
            echo "📄 $(basename "$json_file"):"
            cat "$json_file"
        fi
    done
fi

echo ""
echo "========================================================"
echo "💾 DISK KULLANIMI:"
echo "--------"
du -sh data/ models/ *.py 2>/dev/null | sort -hr

echo ""
echo "========================================================"
echo "🔧 SİSTEM BİLGİLERİ:"
echo "--------"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null || echo 'GPU info bulunamadı')"
echo "Python: $(python3 --version)"
echo "TensorFlow: $(python3 -c 'import tensorflow as tf; print(tf.__version__)' 2>/dev/null || echo 'TF yok')"
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'PyTorch yok')"

echo ""
echo "========================================================"
echo "✅ PROJE ANALİZİ TAMAMLANDI"
echo "========================================================"