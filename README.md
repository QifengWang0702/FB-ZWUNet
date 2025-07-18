# FB-ZWUNet: Corpus Callosum Segmentation for Prenatal Diagnostics

This repository presents the outcomes of the study:

> **"FB-ZWUNet: A deep learning network for corpus callosum segmentation in fetal brain ultrasound images for prenatal diagnostics"**  
> *Qifeng Wang, Dan Zhao, Hao Ma, Bin Liu*  
> Published in **Biomedical Signal Processing and Control**, 2025.  
> DOI: [10.1016/j.bspc.2025.107499](https://doi.org/10.1016/j.bspc.2025.107499)

---

## Overview
FB-ZWUNet is a specialized deep learning framework for **automated segmentation of the corpus callosum (CC)** in **fetal brain mid-sagittal ultrasound images (FBMS)**, designed to assist **prenatal diagnostics**.

![Network Overview](media/network_overview.png)

Key components:
- **Zernike Attention Module (ZAM)** – Shape-aware feature extraction.
- **Wavelet Attention Module (WAM)** – Multi-scale feature fusion.
- **Morphological Constraint Module (MCM)** – Edge and region refinement.

The model is trained on the **FB-CC Dataset** (1,336 annotated FBMS images, 18–32 weeks gestation), outperforming state-of-the-art networks in **Dice (0.8743)** and **IoU (0.7813)**, while maintaining fast inference (135ms).

---

## Highlights
- Outperforms UNet, TransUNet, Swin-UNet, SAM, and other baselines.
- Integrated into a **real-time CAD system** for prenatal brain assessment.
- Demonstrates robust performance across varying gestational ages and devices (GE Voluson E8/E10, Samsung WS80A).

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/<your-username>/FB-ZWUNet.git
cd FB-ZWUNet

# Option 1: Install via pip
pip install -r src/env/requirements.txt

# Option 2: Create conda environment
conda env create -f src/env/environment.yml
conda activate fb-zwunet
```
---
## Dataset Preparation
Organize your dataset as:
```bash
datasets/
  ├── train/
  │    ├── images/   # training ultrasound images
  │    └── labels/   # corresponding segmentation masks
  ├── val/
  │    ├── images/
  │    └── labels/
  └── test/
       ├── images/
       └── labels/
```
Images should be normalized (e.g., using src/common/normalizeImages.py) and resized to 128x128.

---

## Training
Use main-3.0.py to train the network. You can specify the model architecture via the --model argument.
For the recommended Shape-Improved ZAM-WAM UNet:
```bash
python src/main-3.0.py \
  --model Shape_Improved_ZAM_WAM_Unet \
  --train datasets/train \
  --val datasets/val \
  --epochs 100 \
  --batch-size 8 \
  --lr 1e-4 \
  --output checkpoints/fb-zwunet
```
All logs, checkpoints, and training curves will be stored under checkpoints/fb-zwunet/.

---

## Testing
To evaluate the trained model on the test set:
```bash
python src/test-3.0.py \
  --model Shape_Improved_ZAM_WAM_Unet \
  --weights checkpoints/fb-zwunet/best_model.pth \
  --data datasets/test/images \
  --output results/
```
The script will output:
-	Quantitative metrics (Dice, IoU, Hausdorff Distance) in results/metrics.txt.
- Visual segmentation overlays in results/visuals/.

---
## Demonstration
<p align="center">
  <img src="media/demo.png" alt="Watch the demo" width="600"/><br>
  <img src="media/res.png" alt="Watch the results" width="800"/>
</p>

---

## Citation

If you use this work, please cite:

```bibtex
@article{wang2025fbzwunet,
  title={FB-ZWUNet: A deep learning network for corpus callosum segmentation in fetal brain ultrasound images for prenatal diagnostics},
  author={Wang, Qifeng and Zhao, Dan and Ma, Hao and Liu, Bin},
  journal={Biomedical Signal Processing and Control},
  volume={104},
  pages={107499},
  year={2025},
  doi={10.1016/j.bspc.2025.107499}
}
```

---

## Contact
For **data requests, academic collaborations, or presentation materials**:  
- **Email**: [wqf970702@mail.dlut.edu.cn](mailto:wqf970702@mail.dlut.edu.cn)  
- **ResearchGate**: [Qifeng Wang](https://www.researchgate.net/profile/Qifeng-Wang-9?ev=hdr_xprf)
