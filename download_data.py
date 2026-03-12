import os
import shutil

# Kaggle kütüphanesi yüklü olmalı: pip install kaggle
def download_data():
    # Check if all 4 folders exist
    if not (os.path.exists('data/NORMAL') and os.path.exists('data/PNEUMONIA') and os.path.exists('data/COVID19') and os.path.exists('data/TUBERCULOSIS')):
        print("Veri seti indiriliyor...")
        os.system('kaggle datasets download -d jtiptj/chest-xray-pneumoniacovid19tuberculosis -p data/ --unzip')
        print("Veri seti başarıyla indirildi. Klasör yapısı düzenleniyor...")
        
        # Kaggle veri seti train, val, test olarak ayrılmış. Notebook hepsini tek yerden okuduğu için birleştiriyoruz.
        # However, the exact structure of this dataset might differ, we gracefully handle 'train', 'test', 'val' if present.
        
        # Orijinal klasör yapısında genellikle Data/train vs. veya train/NORMAL gibi bir yapı vardır.
        # Bu yüzden tüm indirilen dosyaları tarayıp ilgili sınıfların kök dizinine ('data/SINIF_ADI') taşıyacağız.
        
        labels_mapping = {
            "NORMAL": "NORMAL",
            "PNEUMONIA": "PNEUMONIA",
            "COVID19": "COVID19",
            "COVID-19": "COVID19",
            "TUBERCULOSIS": "TUBERCULOSIS",
            "TURBERCULOSIS": "TUBERCULOSIS" # Handling potential spelling error
        }
        
        allowed_labels = ["NORMAL", "PNEUMONIA", "COVID19", "TUBERCULOSIS"]
        
        for root_label in allowed_labels:
            os.makedirs(f"data/{root_label}", exist_ok=True)
            
        for root, dirs, files in os.walk('data/'):
            # Skip the destination folders themselves to avoid infinite loops or moving already moved files
            if root in [f'data/{l}' for l in allowed_labels]:
                continue
                
            for folder_name in dirs:
                upper_folder = folder_name.upper()
                if upper_folder in labels_mapping:
                    dest_label = labels_mapping[upper_folder]
                    src_dir = os.path.join(root, folder_name)
                    
                    for file in os.listdir(src_dir):
                        src_file = os.path.join(src_dir, file)
                        dst_file = os.path.join(f"data/{dest_label}", file)
                        if os.path.isfile(src_file) and not os.path.exists(dst_file):
                            shutil.move(src_file, dst_file)
                            
        # Temizlik: data/ içindeki ana 4 klasör harici her şeyi sil
        for item in os.listdir('data/'):
            item_path = os.path.join('data/', item)
            if os.path.isdir(item_path) and item not in allowed_labels:
                shutil.rmtree(item_path)
            elif os.path.isfile(item_path):
                os.remove(item_path)
            
        print("Klasör yapısı başarıyla düzenlendi. Veriler 'data/' klasörü altında kullanıma hazır.")
    else:
        print("Veri seti zaten data/ klasöründe mevcut.")

if __name__ == "__main__":
    download_data()
