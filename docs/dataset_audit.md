# Dataset audit

Run with `python -m chestxray.audit`; the numbers below are in `results/audit.json`.

The Kaggle dataset [Chest X-Ray (Pneumonia, Covid-19, Tuberculosis)](https://www.kaggle.com/datasets/jtiptj/chest-xray-pneumoniacovid19tuberculosis) is itself a merge of several public datasets. Before trusting a high test score, it is worth asking what else, besides the disease, separates the classes.

## 1. Every class comes from a different source

| Class | Images | Patients | Max images of one patient | File type | Median size (h × w) |
|---|---|---|---|---|---|
| COVID-19 | 576 | 576* | 1* | `.jpg` | 1028 × 1165 |
| Normal | 1,583 | 1,444 | 4 | `.jpeg` | 1323 × 1654 |
| Pneumonia | 4,273 | 1,674 | 31 | `.jpeg` | 776 × 1160 |
| Tuberculosis | 703 | 703* | 1* | `.png` | 512 × 512 |

\* COVID-19 and Tuberculosis file names contain no patient ID, so each image is treated as its own patient.

Normal and Pneumonia come from one pediatric dataset (children aged 1–5, one hospital in Guangzhou). Tuberculosis images are all 512 × 512 PNGs. COVID-19 images are collected from many places.

### Shortcut probe

If something that says nothing about disease can predict the class, a CNN can learn that instead of the disease ("shortcut learning"). A random forest was trained on such features only (5-fold cross-validation):

| Features (nothing medical) | Accuracy |
|---|---|
| none: always predict the largest class (Pneumonia) | 0.599 |
| brightness, contrast, border brightness of the 224 × 224 image | 0.687 |
| original height, width, aspect ratio | 0.835 |
| both | **0.900** |

The CNN never sees the original size (every image is resized to 224 × 224), but it does see brightness, contrast, the aspect-ratio distortion of resizing and compression artefacts. A test accuracy in the 90s therefore does **not** prove that the model recognises disease. Grad-CAM (notebook, section 7) is one way to check where it looks; a test set from a different hospital would be the real test.

## 2. Several images of one patient

Pneumonia has 4,273 images of 1,674 patients, up to 31 images of one child. With a random split, images of the same patient land in both training and test sets, and the model is partly tested on patients it has already seen.

**Fix:** `chestxray.data.patient_id` reads the patient ID from the file name (`person123_bacteria_45.jpeg` → `person123`, `IM-0605-0001.jpeg` → `IM-0605`), and `split_by_patient` uses `StratifiedGroupKFold`, so every patient is in exactly one set.

## 3. Duplicate images

`chestxray.duplicates` computes a 64-bit perceptual hash of every image (8 × 8 block means, each bit = brighter than average) and then confirms each candidate pair by pixel correlation (> 0.95), because different images can share a hash.

- 52 sets of near-identical images (109 images in total);
- 23 of these sets contain images with **different patient IDs**, so grouping by patient alone would not keep them together;
- 1 set crosses classes: `COVID19(277).jpg` and `person1004_virus_1686.jpeg` are the same pediatric X-ray, labelled once as COVID-19 and once as Pneumonia — a **labelling error** in the dataset.

**Fix:** `merge_groups` joins the patient groups of every duplicate set (union-find), so duplicates are also always in the same set.

## What this means for the results

- The split is now stricter (by patient and by duplicate) than a plain random split, so scores may be slightly lower — but they are more honest.
- Even so, the classes differ in source, so the scores measure "this dataset" and say little about other hospitals. That limitation is stated in the README.
