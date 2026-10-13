# Oxford-IIIT Pet: 37-Breed Classifier (PyTorch Transfer Learning)

Fine-grained image classification of 37 cat and dog breeds. The model starts from an ImageNet-pretrained ResNet-50 and is fine-tuned in two stages. On the official held-out test split it reaches **93.1 % top-1 accuracy** and **99.4 % top-5 accuracy**.

| Metric (official test split, n = 3,669) | Value |
|---|---|
| Top-1 accuracy | **0.9313** |
| Top-5 accuracy | **0.9940** |
| Macro F1 | **0.9302** |
| Best validation accuracy | 0.9538 (epoch 14 of 15) |

---

## Contents

| File | Description |
|---|---|
| `oxford_pets_classifier.ipynb` | Clean, documented version of the pipeline (no outputs). Start here. |
| `Pets_classification.ipynb` | The executed run (Colab, G4 GPU) with all outputs, plots and results reported below. |
| `best_pet_model.pt` | Best checkpoint by validation accuracy. Created when the notebook runs. |
| `data/` | Dataset cache. Downloaded automatically (~800 MB) on the first run. |

---

## Dataset

[Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/) (Parkhi et al., 2012): 37 breeds (12 cat, 25 dog), about 200 images per breed. It is loaded through `torchvision.datasets.OxfordIIITPet`.

| Split | Images | Use |
|---|---|---|
| `trainval` → train (80 %) | 2,944 | Model training |
| `trainval` → validation (20 %) | 736 | Model selection and early stopping |
| `test` (official) | 3,669 | Final evaluation, used **once** |

The train/validation split is **stratified** by breed (`random_state=42`), so every class appears in both parts in the same proportions. Classes are nearly balanced, with 93 to 100 trainval images per breed.

The dataset also provides pixel-level trimaps (pet / background / border). This study does not use them yet (see [Future work](#future-work)).

---

## Method

### 1. Preprocessing and augmentation

| | Training | Validation / Test |
|---|---|---|
| Resize / crop | `RandomResizedCrop(224, scale=(0.6, 1.0))` | `Resize(256)` → `CenterCrop(224)` |
| Flip | `RandomHorizontalFlip` | none |
| Policy | `TrivialAugmentWide` | none |
| Normalisation | ImageNet mean / std | ImageNet mean / std |

The `trainval` files are loaded **twice**: once with the augmented transform (for training) and once with the clean transform (for validation). Validation therefore measures performance on unaltered images, even though it shares source files with training.

### 2. Model

- **Backbone:** ResNet-50 with `IMAGENET1K_V2` weights, 23.6 M parameters. When no CUDA GPU is available, the notebook automatically switches to the lighter **EfficientNet-B0**.
- **Head:** the original 1000-way classifier is replaced with `Dropout(0.3) → Linear(2048, 37)`.

### 3. Two-stage training

**Stage 1: linear probe (frozen backbone), 3 epochs.**
Only the new head is trained (AdamW, lr = 1e-3, OneCycle). The randomly initialised head is fitted to the pretrained features first, so its large early gradients do not damage the backbone weights. This stage alone reaches 91.8 % validation accuracy.

**Stage 2: full fine-tuning, up to 12 epochs.**
All layers are unfrozen, with **discriminative learning rates**:
- backbone: 1e-4 (small, to preserve the pretrained features)
- head: 5e-4

Other settings: OneCycle schedule (20 % warm-up), weight decay 1e-4, **label smoothing 0.1**, mixed precision (AMP) on CUDA, and **early stopping** on validation accuracy (patience 4). The checkpoint with the best validation accuracy is saved and reloaded for testing.

> **Note on the training curves.** Training accuracy is computed on *augmented* images. In the early epochs it is often *lower* than validation accuracy. This is expected and does not indicate a bug.

### 4. Evaluation

Evaluation runs on the official test split, which is not used before this point. The notebook reports:
- top-1 and top-5 accuracy, macro F1
- a per-class precision / recall / F1 report
- a row-normalised confusion matrix and the 10 most-confused breed pairs
- a gallery of the 12 most confident misclassifications

---

## Results

### Training trajectory

| Stage | Epoch | Train loss / acc | Val loss / acc |
|---|---|---|---|
| Head | 1 | 3.130 / 0.342 | 2.153 / 0.864 |
| Head | 3 | 1.397 / 0.878 | 1.307 / 0.918 |
| Fine-tune | 1 (4) | 1.224 / 0.894 | 0.992 / 0.933 |
| Fine-tune | 6 (9) | 0.802 / 0.974 | 0.847 / 0.951 |
| Fine-tune | 11 (14) | 0.741 / 0.990 | 0.839 / **0.954** ✓ best |
| Fine-tune | 12 (15) | 0.739 / 0.990 | 0.837 / 0.952 |

Each epoch took about 5 s on a Colab G4 GPU. Early stopping did not trigger, so all 12 fine-tuning epochs ran. Fine-tuning added about 3.6 points of validation accuracy over the linear probe (91.8 % → 95.4 %).

### Per-class performance

Most breeds score F1 ≥ 0.95. Leonberger, Keeshond, Pug, Samoyed and Japanese Chin are close to perfect. The weakest classes are visually similar breeds:

| Breed | Precision | Recall | F1 |
|---|---|---|---|
| American Pit Bull Terrier | 0.851 | 0.570 | **0.683** |
| Staffordshire Bull Terrier | 0.703 | 0.719 | **0.711** |
| Ragdoll | 0.771 | 0.740 | **0.755** |
| American Bulldog | 0.722 | 0.960 | 0.824 |
| Bengal | 0.754 | 0.920 | 0.829 |

### Most-confused pairs (true → predicted)

| Count | True breed | Predicted as |
|---|---|---|
| 22 | American Pit Bull Terrier | Staffordshire Bull Terrier |
| 16 | American Pit Bull Terrier | American Bulldog |
| 15 | Staffordshire Bull Terrier | American Bulldog |
| 15 | Ragdoll | Birman |
| 13 | Egyptian Mau | Bengal |
| 9 | Staffordshire Bull Terrier | American Pit Bull Terrier |
| 9 | Maine Coon | Ragdoll |
| 7 | Birman | Ragdoll |
| 7 | Abyssinian | Bengal |
| 6 | Maine Coon | Bengal |

**Interpretation.** Most errors fall into two groups of closely related breeds:
- **Bully breeds** (Pit Bull ↔ Staffordshire ↔ American Bulldog). These breeds share build, coat and head shape, and even human annotators find them hard to separate. American Bulldog absorbs many of these errors (recall 0.96, precision only 0.72).
- **Long-haired and spotted cats** (Ragdoll ↔ Birman ↔ Maine Coon; Egyptian Mau / Abyssinian → Bengal). Ragdoll and Birman are both colour-pointed, long-haired breeds. Bengal attracts other spotted or ticked coats.

The errors therefore reflect real visual ambiguity between breeds, not random noise.

### Example prediction on an external image

A pug photo not taken from the dataset was classified as **Pug (0.946)**. The next four candidates (Siamese, Shiba Inu, Chihuahua, Persian) each received ≤ 0.003.

---

## How to run

**Google Colab (recommended).** Open `oxford_pets_classifier.ipynb`, select *Runtime → Change runtime type* and pick a GPU (T4 works; this run used G4), then *Run all*. The full run takes a few minutes, most of it spent downloading the dataset.

**Locally.**
```bash
pip install torch torchvision scikit-learn matplotlib pillow
jupyter notebook oxford_pets_classifier.ipynb
```
The notebook selects the device automatically (CUDA → Apple MPS → CPU). Off-GPU it uses EfficientNet-B0, and results will differ somewhat from the ResNet-50 numbers above.

**Classify your own photo** after training:
```python
predict_image("my_cat.jpg")   # shows the image and prints the top-5 breeds with probabilities
```

### Configuration

All hyperparameters are in the `CFG` dictionary in the first code cell:

| Key | Value | Meaning |
|---|---|---|
| `arch` | `resnet50` (GPU) / `efficientnet_b0` | Backbone |
| `img_size` | 224 | Input resolution |
| `batch_size` | 32 | |
| `val_frac` | 0.2 | Validation fraction of `trainval` |
| `head_epochs` | 3 | Stage 1 epochs |
| `ft_epochs` | 12 | Stage 2 epochs (maximum) |
| `head_lr` / `backbone_lr` | 1e-3 / 1e-4 | Peak learning rates |
| `weight_decay` | 1e-4 | AdamW |
| `label_smoothing` | 0.1 | |
| `patience` | 4 | Early-stopping patience |

### Reproducibility

Python, NumPy and PyTorch seeds are fixed (`SEED = 42`), and the train/validation split is deterministic. GPU kernels and data-loader workers are not fully deterministic, so repeated runs may vary by a few tenths of a percent. Reported results come from PyTorch 2.11 (CUDA 13.0) on a Colab G4 GPU.

---

## Future work

- **Stronger backbones:** `convnext_tiny` or `efficientnet_v2_s` (torchvision or `timm`) usually add a few points.
- **Ablation:** report the linear probe (stage 1 only) against full fine-tuning as a formal baseline comparison.
- **Interpretability:** use Grad-CAM (`pytorch-grad-cam`) to check that the model attends to the animal rather than the background, especially for the bully-breed confusions.
- **Use the trimaps:** train a multi-task model with `target_types=["category", "segmentation"]`, or crop to the animal before classifying.
- **Test-time augmentation:** average predictions over a horizontal flip for a cheap improvement.

---

## References

- O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar. *Cats and Dogs.* CVPR 2012.
- K. He, X. Zhang, S. Ren, J. Sun. *Deep Residual Learning for Image Recognition.* CVPR 2016.
- M. Tan, Q. V. Le. *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.* ICML 2019.
- S. G. Müller, F. Hutter. *TrivialAugment: Tuning-free Yet State-of-the-Art Data Augmentation.* ICCV 2021.
- L. N. Smith, N. Topin. *Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates* (OneCycle). 2019.

## Author

Walusimbi Uthman
