# ViSpa: Visual-Spatial Interactive Fusion for 3D UAV Localization in mmWave Communications

This repository contains the official PyTorch implementation of **ViSpa**, a Visual-Spatial Interactive Fusion framework for 3D UAV localization in millimeter-wave (mmWave) communication systems.

## 📋 Overview

High-precision low-cost Uncrewed Aerial Vehicle (UAV) localization is critical for emerging mmWave communication systems. **ViSpa** addresses the fundamental "Cone of Confusion" ambiguity inherent in ULA-based mmWave localization by jointly exploiting:

- **Beamspace Spatial Features**: RSS measurements from mmWave beam sweeping
- **Visual Observations**: Images from a single base-station-mounted camera

### Key Contributions

1. **Geometric Analysis**: Formal proof of the "Cone of Confusion" ambiguity in ULA-based 3D localization (Proposition 1)
2. **BiFiLM Module**: Bidirectional Feature-wise Linear Modulation for cross-modal representation enhancement
3. **LBF Module**: Low-Rank Bilinear Fusion for efficient mutual interaction modeling
4. **State-of-the-Art Performance**: <2m XY-plane error, ~6m altitude error on DeepSense 6G dataset

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ViSpa Framework                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐        ┌──────────────┐                          │
│  │   Visual     │        │   Spatial    │                          │
│  │ Observations │        │    Beam      │                          │
│  │   (Image)    │        │ Measurements │                          │
│  └──────┬───────┘        └──────┬───────┘                          │
│         │                       │                                   │
│         ▼                       ▼                                   │
│  ┌──────────────┐        ┌──────────────┐                          │
│  │  ConvNeXt-2D │        │  ConvNeXt-1D │                          │
│  │   Encoder    │        │   Encoder    │                          │
│  │     (fv)     │        │     (fb)     │                          │
│  └──────┬───────┘        └──────┬───────┘                          │
│         │                       │                                   │
│         └───────────┬───────────┘                                   │
│                     ▼                                               │
│         ┌─────────────────────┐                                     │
│         │   BiFiLM Module     │                                     │
│         │ (Cross-Modal        │                                     │
│         │  Conditioning)      │                                     │
│         └──────────┬──────────┘                                     │
│                    ▼                                                │
│         ┌─────────────────────┐                                     │
│         │   LBF Module        │                                     │
│         │ (Low-Rank Bilinear  │                                     │
│         │  Fusion)            │                                     │
│         └──────────┬──────────┘                                     │
│                    ▼                                                │
│         ┌─────────────────────┐                                     │
│         │   Regression Head   │                                     │
│         │   (MLP → 3D Pos)    │                                     │
│         └─────────────────────┘                                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## 📊 Results

### Localization Performance

| Setting | Mean MPE (m) | Mean MAE (m) | XY-axis MAE (m) | Z-axis MAE (m) |
|---------|--------------|--------------|-----------------|----------------|
| **ViSpa (Proposed)** | **7.32** | **3.10** | **1.58** | **6.14** |
| w/o BiFiLM | 8.23 | 3.44 | 1.59 | 7.16 |
| w/o LBF | 7.84 | 3.30 | 1.63 | 6.64 |
| Attention-based [2] | 8.27 | 3.49 | 1.68 | 7.10 |
| w/o Visual | 26.05 | 10.42 | 3.23 | 24.81 |
| w/o Beam | 17.77 | 8.00 | 6.61 | 10.77 |

### Top-k Beam Analysis

| Top-k | X-axis MAE (m) | Y-axis MAE (m) | Z-axis MAE (m) |
|-------|----------------|----------------|----------------|
| Top-1 | ~1.5 | ~1.5 | ~8.0 |
| Top-8 | ~1.3 | ~1.3 | ~7.0 |
| Top-16 | ~1.2 | ~1.2 | ~6.0 |
| Top-32 | ~1.2 | ~1.2 | ~6.0 |
| Top-64 | ~1.2 | ~1.2 | ~6.0 |

## 🚀 Getting Started

### Prerequisites

```bash
# Python 3.10+
pip install torch torchvision numpy pandas scikit-learn matplotlib scipy pillow
```

### Dataset

This project uses **Scenario 23** from the [DeepSense 6G](https://www.deepsense6g.net/scenarios/Scenarios%2020-29/scenario-23) dataset.

1. Download the dataset from the official website
2. Organize your data as follows:

```
data/
├── scenario23_dev/
│   └── unit1/
│       └── camera_data/
│           ├── 000001.jpg
│           ├── 000002.jpg
│           └── ...
├── input_rss.csv          # Beam RSS measurements (64 beams)
└── output_3dlocation.csv  # Ground truth 3D coordinates
```

### Training

```python
python main.py
```

**Hyperparameters:**
- Batch Size: 64
- Learning Rate: 8×10⁻⁴
- Weight Decay: 1×10⁻³
- Optimizer: AdamW
- Scheduler: ReduceLROnPlateau
- Epochs: 200 (with early stopping, patience=30)
- Image Size: 320×180

### Inference

```python
import torch
from main import MultimodalFusionNet

# Load pretrained model
model = MultimodalFusionNet(out_dim=3)
model.load_state_dict(torch.load('best_multimodal_modelv3_topk.pth'))
model.eval()

# Inference
with torch.no_grad():
    pred_position = model(image_tensor, beam_rss_tensor)
```

## 📁 Repository Structure

```
ViSpa-UAVLoc/
├── main.py                           # Training and evaluation script
├── best_multimodal_modelv3_topk.pth  # Pretrained model weights
├── input_rss.csv                     # Sample input RSS data
├── output_3dlocation.csv             # Sample output coordinates
└── README.md                         # This file
```

## 🔧 Model Components

### BiFiLM (Bidirectional Feature-wise Linear Modulation)

```python
class BiFiLM(nn.Module):
    """
    Bidirectional FiLM conditioning between signal and image branches.
    
    - s_feat: (B, Cs, L) - Signal feature map
    - i_feat: (B, Ci, H, W) - Image feature map
    
    Produces modulated features through cross-modal conditioning.
    """
```

### LowRankBilinearFusion

```python
class LowRankBilinearFusion(nn.Module):
    """
    Low-rank bilinear fusion for efficient cross-modal interaction.
    
    - Captures multiplicative correlations between modalities
    - Controls model complexity through rank decomposition
    """
```

### ConvNeXt Encoders

- **ConvNeXtBlock1D**: For 1D beam RSS signal encoding
- **ConvNeXtBlock2D**: For 2D visual image encoding

## 📈 Visualization Outputs

The training script generates:

1. `uav_xy_projection.png` - XY plane projection of predictions vs ground truth
2. `uav_error_cdf.png` - CDF of 3D positioning errors with percentile markers
3. `uav_positioning_plot_data.mat` - MATLAB-compatible data for custom visualization

## 🔗 Related Resources

- [DeepSense 6G Dataset](https://www.deepsense6g.net/)
- [FiLM: Visual Reasoning with a General Conditioning Layer](https://arxiv.org/abs/1709.07871)
- [ConvNeXt: A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- DeepSense 6G team for providing the real-world multimodal dataset
- This work was supported by [funding information if applicable]
