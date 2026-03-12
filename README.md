<div align="center">
  
  <br>
  
  # Prediction of Chest Disease from X-Ray
  
  <p>
    <b>Deep Learning dayalı Görüntü Sınıflandırma ile Göğüs Röntgeninden Viral Enfeksiyon (Zatürre / Pneumonia) Tespiti.</b>
  </p>
</div>

---

## 📊 Özet (Overview)

Bu proje, Makine Öğrenimi ve Evrişimli Sinir Ağları (CNN) kullanarak göğüs X-Ray görüntülerini analiz etmeyi ve hastalıkların (Covid-19, Normal, Zatürre ve Verem/Tuberculosis) tespitini yapmayı amaçlamaktadır.

## 💾 Veri Seti (Dataset)

This project uses the **Chest X-Ray (Pneumonia, Covid-19, Tuberculosis)** dataset. Due to GitHub's file size limits, the raw images are not included in this repository.

To run the project, please follow these steps to obtain the dataset:

### Seçenek 1: Otomatik İndirme (Önerilen)
Repo içerisinde bulunan `download_data.py` betiğini çalıştırarak veri setini Kaggle üzerinden otomatik indirebilir ve klasör yapısını (train, val, test yerine düz bir yapıya) notebook'un beklediği formata getirebilirsiniz.

```bash
pip install kaggle
python download_data.py
```
> *Not: Kaggle API'sini kullanabilmeniz için `kaggle.json` kimlik bilginizin sisteminizde (genellikle `~/.kaggle/kaggle.json`) yapılandırılmış olması gerekmektedir.*

### Seçenek 2: Manuel İndirme
1. Veri setini şuradan indirin: [Kaggle - Chest X-Ray (Pneumonia, Covid-19, Tuberculosis)](https://www.kaggle.com/datasets/jtiptj/chest-xray-pneumoniacovid19tuberculosis)
2. İndirdiğiniz arşiv dosyasını `data/` klasörü altına çıkartıp resimleri `data/NORMAL/`, `data/PNEUMONIA/`, `data/COVID19/` ve `data/TUBERCULOSIS/` klasörlerine yerleştirin.

---

## 🚀 Kurulum & Çalıştırma (Getting Started)

Projeyi bilgisayarınıza klonlayın ve gerekli bağımlılıkları indirin:

```bash
git clone https://github.com/KULLANICI_ADINIZ/PredictionOfChestDiseaseFromX-Ray.git
cd PredictionOfChestDiseaseFromX-Ray
```

Veri setini indirdikten sonra Jupyter Notebook üzerinden projeyi çalıştırabilirsiniz:
```bash
jupyter notebook "göğüs röntgeninden viral enfeksiyon tespiti.ipynb"
```

## 🧠 Model

Projede `MobileNetV2` ve ek `Dense`, `Dropout`, ve `BatchNormalization` katmanları ile optimize edilmiş önceden eğitilmiş (Pre-Trained) Transfer Learning yaklaşımı kullanılmıştır.
