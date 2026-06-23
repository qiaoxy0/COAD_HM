# Detecting Hypermutation Status in Colon Adenocarcinoma from Whole Slide Images

![Project Summary](https://github.com/qiaoxy0/COAD_HM/blob/main/summary.png)

This repository contains the computational pipeline and deep learning models for predicting **Hypermutation (HM)** status in Colon Adenocarcinoma (COAD) directly from Hematoxylin and Eosin (H&E) stained Whole Slide Images (WSIs). 

Using weakly supervised **Attention-based Multiple Instance Learning (AMIL)** adapted from CLAM, this project classifies slide-level hypermutation status and identifies the specific histological regions driving the predictions.

---

## Project Overview

Somatic hypermutation status in colorectal cancer is a critical biomarker for immunotherapy response. While traditionally determined via genomic sequencing (e.g., DNA sequencing), this project demonstrates that deep learning models can detect hypermutation status directly from the phenotypic patterns visible in standard H&E stained pathology slides. 

The pipeline extracts millions of tiles from whole slide images, encodes them into low-dimensional features, and trains a weakly supervised multiple-instance classifier using slide-level labels.

---

## Dataset & Patient Cohort

### Cohort Matching & Downsampling
The dataset is derived from the Cancer Genome Atlas (**TCGA-COAD** cohort):

| Resolution Level | Hypermutated (HM) | Non-Hypermutated (Non-HM) | Total |
| :--- | :--- | :--- | :--- |
| **Patient Level** | 76 | 79 | 155 |
| **Slide Level** | 140 | 142 | 282 |

### Quality Control & Filtering
During preprocessing, slides were filtered based on diagnostic utility, slide-level blurring, and tile yields. 

## Image Preprocessing Pipeline

Pathology slides are massive (gigapixel scale) and must be preprocessed prior to feature extraction:

### 1. Tiling
Whole slide images are cropped into small, non-overlapping tiles. Magnifications are normalized during extraction:
*   **x20 magnification slides**: Extracted at a resolution of 0.5 $\mu m$/pixel, resized to $512 \times 512$ pixels.
*   **x40 magnification slides**: Extracted at a resolution of 0.25 $\mu m$/pixel, resized to $512 \times 512$ pixels.

### 2. Blurriness and Background Filtering
To prevent low-informative background or blurred tissue from entering the training loop, Canny edge detection is applied. Tiles are only kept if their edge-density score exceeds a threshold:
$$\text{Edge Score} > 2.0$$

### 3. Stain Normalization
Histology slides exhibit significant color variations due to different scanners, staining protocols, and reagents. To improve model generalization, stain normalization is performed using **Macenko's method** (Macenko et al., 2009). All tiles are normalized against a standard reference image (`preprocessing/Ref.png`) from KatherLab.

---

## Deep Learning Algorithm

### 1. Feature Extraction (ResNet50)
A ResNet50 model pre-trained on ImageNet is used to extract low-dimensional embeddings from the tissue tiles. This reduces each $512 \times 512 \times 3$ tile into a 1024-dimensional feature vector, speeding up subsequent model training.

### 2. Attention-based Multiple Instance Learning (AMIL)
Since only slide-level labels (HM vs. Non-HM) are available, we frame the task as **Multiple Instance Learning (MIL)**:
*   Each WSI is treated as a **bag**.
*   Each tile embedding is an **instance**.
*   An attention network assigns a weight $a_i$ to each instance, indicating its relative importance to the slide-level classification.

### 3. Hybrid Loss Function
The model is optimized using a hybrid loss function that balances slide-level and tile-level objectives:
$$\mathcal{L}_{\text{total}} = w_{\text{bag}} \mathcal{L}_{\text{bag}} + (1 - w_{\text{bag}}) \mathcal{L}_{\text{instance}}$$

*   **Bag-level Loss ($\mathcal{L}_{\text{bag}}$)**: Cross-Entropy loss computed using slide-level predictions against the true hypermutation label.
*   **Instance-level Loss ($\mathcal{L}_{\text{instance}}$)**: Computed using a smoothed Top-1 Support Vector Machine (SVM) loss at the tile level to guide cluster separation of positive and negative instances.

### 4. Multi-Class Learning (Tumor vs. Normal)
A three-class classification task (**Normal** vs. **Hypermutated** vs. **Non-Hypermutated**) is also implemented. Although the normal class is non-informative for mutation status, its inclusion forces the network to learn to distinguish normal histology from tumor tissue, acting as an implicit focus mechanism.

---

## Repository Structure

```text
COAD_HM-main/
├── README.md                              # This file
├── summary.png                            # Project summary image
├── requirements.txt                       # Python packages required
├── Extract_features.py                    # Feature extraction script using ResNet50
├── AMIL_train.py                          # Training script for Attention-based MIL (3-fold CV)
├── Classic_train.py                       # Training script for fully-supervised tile-level baseline
├── Topk_tiles.py                          # Extract tiles with the highest attention weights
├── heatmap_gen.py                         # Generate attention heatmaps from WSIs
├── preprocessing/                         # Slide tiling and normalization scripts
│   ├── Extract_tiles.py                   # Slide tiler using OpenSlide
│   ├── Normalize.py                       # Batch stain normalization coordinator
│   ├── stainNorm_Macenko.py               # Macenko normalizer class
│   ├── stain_utils.py                     # Stain analysis utility functions
│   └── Ref.png                            # KatherLab reference image for normalization
├── models/                                # Model definitions
│   ├── model_att_mil.py                   # AMIL/CLAM architecture
│   └── resnet_custom.py                   # Customized ResNet feature extractor
├── dataset/                               # Dataset loaders
│   └── dataset_generator.py               # Bag loaders for HDF5 and .pt data
├── eval/                                  # Evaluation metrics
│   └── eval.py                            # AUC, ROC, and result collation utilities
└── utils/                                 # General utilities
    ├── core_utils.py                      # Training loop implementations
    ├── data_utils.py                      # Classical supervised loading utilities
    └── utils.py                           # Logging and networking helpers
```

---

## Usage Instructions

### 1. Slide Tiling
Extract tiles from diagnostic whole slide files (e.g. `.svs` or `.ndpi` format):
```bash
python preprocessing/Extract_tiles.py -s /path/to/slides/ -o /path/to/output_tiles/ --px 512
```

### 2. Stain Normalization
Normalize tiles to mitigate domain shifts:
```bash
python preprocessing/Normalize.py
```
*Note*: Update the `inputPath` and `outputPath` inside [preprocessing/Normalize.py](file:///Users/qiaoxy/Downloads/COAD_HM-main/preprocessing/Normalize.py) to point to your tiled directories.

### 3. Feature Extraction
Embed the normalized tiles into low-dimensional representations using the pre-trained ResNet50:
```bash
python Extract_features.py --datadir_train /path/to/normalized_tiles/ --feat_dir /path/to/extracted_features/ --batch_size 256
```

### 4. Attention-MIL Model Training
Train the multiple-instance model using stratified 3-fold cross-validation:
```bash
python AMIL_train.py --feat_dir /path/to/extracted_features/ --csvFile /path/to/annotations.csv --k 3 --output_dir ./results
```

### 5. Heatmap Generation
Generate attention heatmaps to visualize the regions of high classification importance:
```bash
python heatmap_gen.py --ckpt_path /path/to/model_checkpoint.pt --slide_name <slide_id> --h5_path /path/to/slide_coordinates.h5
```

---

## Dependencies
Major Python libraries required for this project include:
*   PyTorch
*   torchvision
*   timm
*   OpenSlide Python
*   OpenCV Python
*   scikit-learn
*   h5py
*   pandas
*   numpy
*   matplotlib
*   seaborn

Install all packages using:
```bash
pip install -r requirements.txt
```

---

## References

1. The Cancer Genome Atlas (TCGA) Research Network: https://www.cancer.gov/tcga.
2. Macenko M, Niethammer M, Marron JS, et al. *"A method for normalizing histology slides for quantitative analysis."* IEEE ISBI, 2009.
3. Lu MY, Williamson DFK, Kong DH, et al. *"Data-efficient and weakly supervised computational pathology on whole-slide images."* Nature Biomedical Engineering, 2021 (CLAM reference).
