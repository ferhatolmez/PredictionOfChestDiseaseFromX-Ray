import os
import shutil

# Kaggle kütüphanesi yüklü olmalı: pip install kaggle
def download_data():
    if not os.path.exists('data/NORMAL') or not os.path.exists('data/PNEUMONIA'):
        print("Veri seti indiriliyor...")
        os.system('kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p data/ --unzip')
        print("Veri seti başarıyla indirildi. Klasör yapısı düzenleniyor...")
        
        # Kaggle veri seti train, val, test olarak ayrılmış. Notebook hepsini tek yerden okuduğu için birleştiriyoruz.
        base_dir = "data/chest_xray"
        splits = ["train", "val", "test"]
        labels = ["NORMAL", "PNEUMONIA"]
        
        for label in labels:
            os.makedirs(f"data/{label}", exist_ok=True)
            for split in splits:
                split_dir = os.path.join(base_dir, split, label)
                if os.path.exists(split_dir):
                    for file in os.listdir(split_dir):
                        src = os.path.join(split_dir, file)
                        dst = os.path.join(f"data/{label}", file)
                        if not os.path.exists(dst):
                            shutil.move(src, dst)
                            
        # chest_xray ve içindeki gereksiz dosyaları temizle
        if os.path.exists(base_dir):
            shutil.rmtree(base_dir)
        # Orijinal klasör yapısındaki macos klasörleri varsa sil
        if os.path.exists("data/__MACOSX"):
            shutil.rmtree("data/__MACOSX")
            
        print("Klasör yapısı başarıyla düzenlendi. Veriler 'data/' klasörü altında kullanıma hazır.")
    else:
        print("Veri seti zaten data/ klasöründe mevcut.")

if __name__ == "__main__":
    download_data()
