# 🔊 Urban Sound Classification using Convolutional Neural Networks

> Classifying 10 categories of urban environmental sounds by transforming raw audio into Mel-Spectrograms and training CNN architectures — including a custom-built model and Transfer Learning variants of InceptionV3 — with adversarial robustness evaluation via DeepFool.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Pipeline Architecture](#pipeline-architecture)
- [Feature Extraction](#feature-extraction)
- [Data Preprocessing](#data-preprocessing)
- [Models](#models)
- [Training Strategy](#training-strategy)
- [Results Summary](#results-summary)
- [Adversarial Robustness — DeepFool](#adversarial-robustness--deepfool)
- [Conclusions](#conclusions)
- [Tech Stack](#tech-stack)
- [References & Resources](#references--resources)
- [Team](#team)

---

## Overview

This project explores **urban sound classification** using deep learning on the **UrbanSound8K** dataset — a benchmark corpus of 8,732 labeled audio clips across 10 environmental sound categories. Raw `.wav` files are converted into **Mel-Spectrogram** images and fed into Convolutional Neural Networks that learn to identify spectral-temporal patterns unique to each sound class.

Four CNN configurations were trained and compared:

| Model | Augmentation | Transfer Learning |
|---|---|---|
| Custom CNN | ✗ | N/A |
| Custom CNN | ✓ | N/A |
| InceptionV3 | ✗ | ✗ (random init) |
| InceptionV3 | ✓ | ✓ (ImageNet weights) |

Beyond accuracy, the project evaluates each model's **adversarial robustness** using the DeepFool attack algorithm, providing insight into how stable each model's decision boundaries are under minimal input perturbations.

---

## Dataset

- **Name:** UrbanSound8K
- **Size:** 8,732 audio clips across 10 folds (fold-based cross-validation)
- **Download:** [UrbanSound Dataset Website](https://urbansounddataset.weebly.com/download-urbansound8k.html)
- **Classes (10):**

| ID | Class |
|---|---|
| 0 | air_conditioner |
| 1 | car_horn |
| 2 | children_playing |
| 3 | dog_bark |
| 4 | drilling |
| 5 | engine_idling |
| 6 | gun_shot |
| 7 | jackhammer |
| 8 | siren |
| 9 | street_music |

> **Class Imbalance Note:** `car_horn` (429 samples) and `gun_shot` (374 samples) are significantly underrepresented compared to the average of ~873 samples per class. Data augmentation via SpecAugment-style input masking was applied to address this.

---

## Project Structure

```
Sound_Classification_CNN/
│
├── data/
│   ├── sound_datasets/
│   │   └── urbansound8k/
│   │       ├── audio/                  # Raw .wav files organized by fold
│   │       └── metadata/
│   │           └── UrbanSound8K.csv
│   └── processed/
│       ├── melspecs/                   # Extracted .npy mel spectrograms by fold
│       │   └── fold{1-10}/
│       ├── melspec_metadata_fold{N}.csv
│       └── training_set/
│           ├── X_train.npy             # Augmented training data
│           └── y_train.npy
│
├── Models/
│   ├── custom_cnn_no_aug.pth
│   ├── custom_cnn.pth
│   ├── inception_v3_noTF_noAug.pth
│   ├── inception_v3_noTF.pth
│   └── inception_v3.pth
│
├── Projeto_DF_CNN.ipynb               # Full pipeline notebook
└── README.md
```

---

## Pipeline Architecture

```
Raw Audio (.wav)
      │
      ▼
Feature Extraction (Librosa)
  → Mel-Spectrogram (128 × 173)
  → Saved as .npy arrays
      │
      ▼
Data Preprocessing
  → Tiling (pad short clips by looping)
  → Input Masking Augmentation
  → Train / Validation / Test split (Folds 1–8 / 10 / 9)
      │
      ▼
Model Training (PyTorch)
  → Custom CNN  |  InceptionV3 (no TL)  |  InceptionV3 (TL)
  → Multi-seed training (seeds: 42, 0, 1)
  → Early Stopping + ReduceLROnPlateau
      │
      ▼
Evaluation
  → Accuracy, Confusion Matrix, Per-Class Analysis
  → DeepFool Adversarial Robustness (ρ_adv)
```

---

## Feature Extraction

Audio files are converted to **Mel-Spectrograms** using [Librosa](https://librosa.org/), a Python library for audio analysis. Mel-Spectrograms represent frequency content over time using a perceptually-motivated frequency scale, making them well-suited as visual input for CNNs.

### Extraction Parameters

| Parameter | Value | Rationale |
|---|---|---|
| `DURATION` | 4 seconds | Majority of clips are 4s; standardizes input length |
| `SAMPLE_RATE` | 22,050 Hz | Captures frequencies up to ~11 kHz; sufficient for environmental sounds |
| `N_MELS` | 128 | Standard choice for audio classification; balances resolution and cost |
| `N_FFT` | 2,048 | Window size ≈ 93ms; ~10.8 Hz/bin at the given sample rate |
| `HOP_LENGTH` | 512 | ~23ms between frames; standard temporal resolution |

### Output Shape

Each spectrogram is saved as a **128 × 97** (or tiled to **128 × 173**) NumPy array in decibel scale (`librosa.power_to_db`), normalized relative to the peak energy.

---

## Data Preprocessing

### Handling Variable-Length Clips

Audio clips shorter than 4 seconds produce spectrograms narrower than the target 173 frames. Three strategies were evaluated:

| Strategy | Approach | Issue |
|---|---|---|
| Zero Padding | Fill remaining frames with silence | Model may associate silence with a class |
| Stretching | Interpolate to target width | Changes pitch; distorts spectral patterns |
| **Tiling (chosen)** | **Repeat the spectrogram to fill target width** | **Preserves natural sound patterns; improves robustness** |

Tiling was selected because it avoids both artificial silence and spectral distortion, instead repeating the actual sound signal — consistent with how repeating environmental sounds occur in real environments.

### Data Augmentation (SpecAugment-style Input Masking)

To address class imbalance and improve generalization, augmentation was applied using frequency and time masking:

- **Frequency Masking:** Randomly zeroes out horizontal frequency bands (`freq_mask_param=15`, `n_masks=2`)
- **Time Masking:** Randomly zeroes out vertical time segments (`time_mask_param=25`, `n_masks=2`)

This forces the model to learn robust frequency-time representations rather than memorizing specific patterns — effectively a form of SpecAugment tailored to Mel-Spectrograms.

### Dataset Split

| Split | Folds | Samples |
|---|---|---|
| Training (with aug) | 1–8 | 8,220 |
| Training (no aug) | 1–8 | 7,079 |
| Validation | 10 | 837 |
| Test | 9 | 816 |

---

## Models

### Custom CNN

A purpose-built 4-block convolutional network with two fully-connected layers, designed specifically for Mel-Spectrogram classification.

```
Input: (1, 128, 173)
  ↓
Block 1: Conv2d(1→64) + BN + MaxPool2d + Dropout(0.4)
Block 2: Conv2d(64→128) + BN + MaxPool2d + Dropout(0.5)
Block 3: Conv2d(128→256) + BN + MaxPool2d + Dropout(0.5)
Block 4: Conv2d(256→512) + BN + MaxPool2d + Dropout(0.5)
  ↓
Flatten → FC(512) → Dropout → FC(10)
  ↓
Output: 10 class logits
```

### InceptionV3 — No Transfer Learning

Google's Inception architecture (V3) initialized with random weights and retrained from scratch on the UrbanSound8K data. The input layer was modified to accept 1-channel grayscale spectrograms (128×173 → upscaled to 299×299 with 3-channel replication). The output layer was adapted to 10 classes.

### InceptionV3 — Transfer Learning (ImageNet)

The same InceptionV3 architecture loaded with **pre-trained ImageNet weights** (`Inception_V3_Weights.IMAGENET1K_V1`). The input channel was modified for grayscale compatibility, and the final classification head was replaced with a 10-class linear layer. Pre-trained convolutional features serve as a strong initialization point for audio pattern learning via mel-spectrograms.

---

## Training Strategy

All models were trained using a **multi-seed approach** (seeds: 42, 0, 1) to evaluate variance in results and select the most stable checkpoint.

### Regularization Techniques

| Technique | Purpose |
|---|---|
| **L2 Regularization** (`weight_decay=1e-4`) | Penalizes large weights to prevent overfitting |
| **Batch Normalization** | Normalizes layer outputs (mean=0, std=1) for stable training |
| **Dropout** (0.4–0.5) | Stochastically disables neurons to improve generalization |
| **Early Stopping** | Halts training when validation loss stops improving (patience: 20 for CNN, 10 for Inception) |
| **ReduceLROnPlateau** | Reduces learning rate when validation accuracy plateaus |

### Hyperparameters

| Parameter | Custom CNN | InceptionV3 |
|---|---|---|
| Optimizer | Adam | Adam |
| Initial LR | 0.001 | 0.001 |
| Max Epochs | 200 | 100 |
| Batch Size | 32 | 32 |
| Loss Function | CrossEntropyLoss | CrossEntropyLoss |

---

## Results Summary

### Model Accuracy Comparison

| Model | Val Accuracy | Test Accuracy |
|---|---|---|
| Custom CNN (no aug) | 80.88% | 76.84% |
| Custom CNN (with aug) | 77.06% | 78.55% |
| InceptionV3 (no TL, no aug) | 82.56% | 80.76% |
| InceptionV3 (no TL, with aug) | 84.35% | 76.59% |
| InceptionV3 (TL, no aug) | 84.23% | 79.78% |
| **InceptionV3 (TL, with aug)** | **84.35%** | **81.99%** |

InceptionV3 with Transfer Learning and augmented data achieved the best generalization on the held-out test set. The custom CNN showed competitive performance given its significantly simpler architecture.

### Key Observations

**Data augmentation effects:** For the Custom CNN, augmentation improved test accuracy (+1.71%) at the cost of slightly lower validation accuracy, suggesting better generalization. For InceptionV3 without transfer learning, augmentation improved validation but hurt test performance — indicative of train-validation overfitting.

**Transfer learning advantage:** InceptionV3 initialized with ImageNet weights consistently outperformed its randomly initialized counterpart on the test set, particularly with augmented training data — confirming that visual feature representations learned on natural images transfer well to Mel-Spectrogram classification.

**Common confusion pairs:** The most frequently confused class pairs across models were engine_idling ↔ air_conditioner and drilling ↔ jackhammer — acoustically similar sounds that share continuous, low-frequency harmonic content. Siren and gun_shot were consistently among the most challenging classes.

---

## Adversarial Robustness — DeepFool

The [DeepFool algorithm](https://arxiv.org/abs/1511.04599) (Moosavi-Dezfooli et al., 2016) finds the **minimum perturbation** needed to cross a classifier's decision boundary — effectively measuring how robust the model is to imperceptibly small input changes.

### Metric

$$\rho_{\text{adv}} = \mathbb{E}\left[ \frac{\lVert r \rVert_2}{\lVert x \rVert_2} \right]$$

A **lower ρ_adv** means smaller perturbations suffice to fool the model → **less robust**.

### Implementation

The multi-class DeepFool algorithm (Algorithm 2 from the original paper) was implemented in PyTorch. For each sample, the algorithm:
1. Computes gradients of the predicted class logit w.r.t. the input
2. Computes gradients for all other classes
3. Approximates the nearest decision boundary hyperplane
4. Accumulates minimal perturbations until the predicted label changes

### Robustness Results

| Model | ρ_adv | Relative Robustness |
|---|---|---|
| Custom CNN (no aug) | ~0.0040–0.0050 | Low |
| Custom CNN (with aug) | ~0.0045–0.0055 | Slightly better |
| InceptionV3 (no TL) | ~0.0425 | Very low (fragile) |
| **InceptionV3 (TL)** | **~0.00037** | **Highest robustness** |

**Key insight:** The InceptionV3 with Transfer Learning was by far the most adversarially robust model, with ρ_adv nearly two orders of magnitude lower than the non-transfer version. This suggests that pre-trained weights lead to substantially more stable and well-defined decision boundaries. The InceptionV3 trained from scratch was the most vulnerable — consistent with literature showing that randomly initialized networks form highly irregular decision boundaries.

### Example Adversarial Attacks

DeepFool consistently converged in 1–4 iterations across all models, illustrating how efficiently minimal perturbations can fool even high-accuracy classifiers. Typical misclassification patterns included:

- `siren` → `children_playing`
- `engine_idling` → `air_conditioner`
- `jackhammer` → `drilling`

These align with the acoustic similarity patterns observed in standard confusion matrices.

---

## Conclusions

This project demonstrates that audio classification via Mel-Spectrograms and CNNs is a viable and effective approach for urban sound recognition. Key takeaways:

**Architecture matters.** The deep, multi-path architecture of InceptionV3 substantially outperforms a custom 4-block CNN on this task, particularly on the more challenging minority classes.

**Transfer learning generalizes well to audio.** Despite ImageNet weights being trained on visual images, they provide a strong initialization that meaningfully improves both accuracy and adversarial robustness for mel-spectrogram-based classification.

**Augmentation trades off differently across architectures.** For simpler models, augmentation reliably improves test generalization. For complex models, augmentation effects are more nuanced — improving validation but potentially introducing overfitting on the augmented distribution.

**Adversarial robustness is not free.** High accuracy does not imply high robustness. The Custom CNN and InceptionV3 without transfer learning remain highly susceptible to DeepFool-style attacks, underscoring the importance of evaluating models beyond standard accuracy metrics when deploying in sensitive environments.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.x | Primary language |
| PyTorch | Deep learning framework |
| torchvision | InceptionV3 architecture and pre-trained weights |
| Librosa | Audio loading and Mel-Spectrogram extraction |
| NumPy | Numerical operations and array storage |
| pandas | Metadata management |
| scikit-learn | Label encoding, evaluation utilities |
| matplotlib / seaborn | Visualization |
| tqdm | Progress tracking |
| torchinfo | Model summary |
| soundata | Dataset download utility |
| SciPy | Spectrogram stretching (zoom interpolation) |
| Jupyter Notebook | Interactive development |

---

## References & Resources

- [Librosa Documentation](https://librosa.org/doc/0.11.0/index.html)
- [PyTorch Documentation](https://docs.pytorch.org/docs/stable/index.html)
- [PyTorch InceptionV3](https://docs.pytorch.org/vision/main/models/inception.html)
- [UrbanSound8K Dataset](https://urbansounddataset.weebly.com/download-urbansound8k.html)
- Moosavi-Dezfooli, S., Fawzi, A., & Frossard, P. (2016). *DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks*. CVPR 2016.
- Park, D. S., et al. (2019). *SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition*. Interspeech 2019.
- Szegedy, C., et al. (2016). *Rethinking the Inception Architecture for Computer Vision*. CVPR 2016.

---

## Team

| Name | Role |
|---|---|
| Hugo Duarte de Sousa | ML Pipeline, CNN Architecture,  CNN Training Infrastructure, CNN Data Preprocessing, CNN Model Evaluation |
| Mariana de Sousa Serralheiro |  DeepFool Implementation and Analysis|
| Tiago Lemos Silva | RNN Feature Extraction, RNN Training Infrastructure, RNN Training Infrastructure, RNN Data Preprocessing, RNN Model Evaluation |

**Course:** Machine Learning II — CC3043 PL1
