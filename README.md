# Prediction of Chest Disease from X-Ray

[![tests](https://github.com/ferhatolmez/PredictionOfChestDiseaseFromX-Ray/actions/workflows/tests.yml/badge.svg)](https://github.com/ferhatolmez/PredictionOfChestDiseaseFromX-Ray/actions/workflows/tests.yml)

Classification of chest X-rays into **COVID-19, Normal, Pneumonia and Tuberculosis** with a pretrained CNN (MobileNetV2, transfer learning and fine-tuning), a **dataset audit** that looks for patient overlap, duplicates and shortcuts, **Grad-CAM** explanations, and two classic image-processing algorithms implemented **from scratch in NumPy**.

## Project structure

```
├── chestxray/                     # the pipeline, as a Python package
│   ├── config.py                  #   all paths and hyperparameters
│   ├── data.py                    #   loading, patient IDs, patient-grouped split
│   ├── preprocessing.py           #   white balance
│   ├── duplicates.py              #   perceptual hash, near-identical images
│   ├── audit.py                   #   dataset audit (python -m chestxray.audit)
│   ├── pipeline.py                #   tf.data input pipeline + augmentation
│   ├── model.py                   #   MobileNetV2 model
│   ├── train.py                   #   two-phase training (python -m chestxray.train)
│   ├── evaluate.py                #   metrics and plots
│   ├── gradcam.py                 #   Grad-CAM heatmaps
│   └── predict.py                 #   predictions for new images (used by app.py)
├── from_scratch/                  # NumPy only: histogram equalization, Sobel, a neural network
├── app.py                         # demo web app (Gradio): drop an X-ray, get class + Grad-CAM
├── tests/                         # pytest unit tests
├── chest_xray_classification.ipynb  # results and explanations (uses the package)
├── results/                       # metrics, training history, split of every run
├── docs/
│   ├── dataset_audit.md           #   what the audit found and how the split handles it
│   └── math_in_this_project.md    #   the mathematics behind the code
├── download_data.py               # downloads and flattens the Kaggle dataset
├── requirements.txt / environment.yml
└── .github/workflows/tests.yml    # style check + tests on Linux and Windows
```

## Getting started

```bash
git clone https://github.com/ferhatolmez/PredictionOfChestDiseaseFromX-Ray.git
cd PredictionOfChestDiseaseFromX-Ray
git lfs pull                       # the trained model is stored with Git LFS
```

Create an environment with **either** Anaconda **or** pip:

```bash
# Anaconda / Miniconda
conda env create -f environment.yml
conda activate chest-xray

# or: venv + pip
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Get the data ([Kaggle dataset](https://www.kaggle.com/datasets/jtiptj/chest-xray-pneumoniacovid19tuberculosis), 7,135 images, needs a Kaggle API token in `~/.kaggle/kaggle.json`), then run:

```bash
python download_data.py            # -> data/COVID19, data/NORMAL, data/PNEUMONIA, data/TUBERCULOSIS
python -m chestxray.audit          # -> results/audit.json
python -m chestxray.train --seed 0 # -> results/seed_0/ (about 30 minutes on a CPU)
jupyter notebook chest_xray_classification.ipynb
```

## Method

1. **Preprocessing:** resize to 224×224, BGR→RGB, white balance per channel. Images are cached in `data/cache_224.npz`.
2. **Split:** stratified by class **and grouped by patient**: `StratifiedGroupKFold` puts ≈71% of the patients in training, ≈14% in validation and ≈14% in the test set. Near-identical images are found with a perceptual hash and kept in the same set as well. Validation is used for early stopping, learning-rate reduction and checkpoint selection; the test set is used **once**, at the end.
3. **Model:** MobileNetV2 (ImageNet) + `GlobalAveragePooling2D` → `Dense(256)` → `Dropout(0.5)` → `Dense(4, softmax)`. A `Rescaling` layer maps inputs to [-1, 1], the range MobileNetV2 was trained on.
4. **Training:** phase 1 trains only the head on the frozen backbone (lr 1e-3); phase 2 fine-tunes the last 30 backbone layers (lr 1e-5). Light augmentation in a `tf.data` pipeline (no horizontal flip: the heart is on the left), balanced class weights.

## Results

All numbers are on the **test set**, which is used once after training. The split is grouped by patient and by duplicate image (see the audit below).

**Fine-tuned MobileNetV2, three runs with different seeds** (the seed changes the split, the initial weights, the augmentation and the batch order):

| Seed | Accuracy | Macro F1 | Macro ROC-AUC | F1 COVID-19 | F1 Normal | F1 Pneumonia | F1 Tuberculosis |
|---|---|---|---|---|---|---|---|
| 0 | 0.938 | 0.929 | 0.994 | 0.914 | 0.903 | 0.955 | 0.943 |
| 1 | 0.919 | 0.912 | 0.992 | 0.913 | 0.873 | 0.939 | 0.921 |
| 2 | 0.937 | 0.933 | 0.994 | 0.940 | 0.894 | 0.954 | 0.944 |
| **mean ± std** | **0.931 ± 0.011** | **0.924 ± 0.011** | **0.993 ± 0.001** | 0.922 | 0.890 | 0.949 | 0.936 |

The weakest class is **Normal** (precision 0.82): most errors are Pneumonia X-rays predicted as Normal, the costly kind of error in medicine.

**Masking experiment** (frozen backbone, seed 0): what if the model may only see part of the image?

| The model sees | Accuracy | Macro F1 |
|---|---|---|
| the whole image | 0.928 | 0.914 |
| only the lungs (central ellipse) | 0.934 | 0.919 |
| everything **except** the lungs | **0.889** | **0.854** |
| *(always guess the largest class)* | *0.599* | — |

Even with the lungs blacked out the classes are easy to tell apart — strong evidence of **shortcut learning** caused by the different sources (details in the audit below).

**NumPy MLP written from scratch** (`from_scratch/neural_net.py`, 32×32 gray input, same split): accuracy **0.946**, macro F1 **0.928** — as good as the CNN, which again shows how easy this dataset is.

**Grad-CAM** heatmaps (notebook, section 7) mostly fall on the chest, not on text markers; they are too coarse to rule out global cues such as brightness.

## Dataset audit — read this before trusting the numbers

Details: [`docs/dataset_audit.md`](docs/dataset_audit.md).

- **Each class comes from a different source**, with its own file type and image size. A random forest that sees only the image size and brightness — nothing medical — predicts the class with **90%** accuracy (60% for always guessing the largest class). Part of a high score may therefore come from recognising the source, not the disease.
- **Several images per patient** (Pneumonia: 4,273 images of 1,674 patients) — handled by the patient-grouped split.
- **52 sets of duplicate images**, one X-ray labelled both COVID-19 and Pneumonia — handled by keeping duplicates in the same set.

## Demo app

```bash
python app.py        # opens http://127.0.0.1:7860
```

A small web interface written in Python with [Gradio](https://www.gradio.app/): drag and drop a chest X-ray to see the probability of each class and a **Grad-CAM** heatmap of the regions that drove the prediction, or drop several files to get a table. It uses `chestxray/predict.py`, which applies exactly the training preprocessing.

> **Not a medical device.** The model was trained on one public dataset whose classes come from different sources (see the audit above), so it says little about X-rays from other hospitals.

## Algorithms from scratch (NumPy)

| Module | What it does |
|---|---|
| `histogram_equalization.py` | Builds the histogram and CDF by hand and maps every gray level through the CDF to stretch contrast. |
| `sobel.py` | 2-D correlation with reflect-101 padding, Sobel Gx/Gy kernels, gradient magnitude and direction. |

![From-scratch demo: original, histogram equalized, Sobel edges](assets/from_scratch_demo.png)

Both are verified against OpenCV: histogram equalization matches `cv2.equalizeHist` exactly, and Gx/Gy match `cv2.Sobel(ksize=3)`.

## Tests and style

```bash
pytest                                                             # 36 unit tests
ruff check . && ruff format --check chestxray from_scratch tests   # PEP 8 / PEP 257
```

The style check and the tests run automatically on **Ubuntu and Windows** with GitHub Actions on every push (the TensorFlow tests run locally only).

## Mathematics

[`docs/math_in_this_project.md`](docs/math_in_this_project.md) explains the linear algebra, probability, statistics and calculus behind the code (convolution as an inner product, gradient norm, CDF, softmax, chain rule / backpropagation) and links every concept to where it is used.
