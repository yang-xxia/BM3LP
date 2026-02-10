# 🏗️ BM3LP

**Bridge Maintenance Multimodal Link Prediction (BM3LP)** via spatiotemporal feature fusion and cross-modal contrastive interaction.

## 🔍 Overview

The BM3LP (Bridge Maintenance Multimodal Link Prediction) model aims to tackle the inconsistency across image modalities, as well as the challenges in expressing and reasoning spatiotemporal characteristics of bridge defects. By incorporating both image and text modalities, the model integrates spatiotemporal semantic information to provide an accurate understanding of defects, ultimately contributing to more effective treatment decision-making in bridge maintenance.

- 📷 **Graph Structure Context**: Utilizes a Depth-First Search (DFS) encoder and Relational Graph Attention Network (RGAT) to capture multi-hop dependencies across defect and structural data.  
- 🌐 **Multimodal knowledge graphs**, integrating visual object detection, label co-occurrence statistics, and text-based knowledge graph embeddings  
- 🔁 **Dual-channel GCNs** to propagate features and model hierarchical dependencies across structural and defect labels  
- 🧠 **Attention-based fusion** to adaptively integrate multimodal semantic information for robust joint prediction  

---

## 🖼️ Graphical Abstract

<p align="center">
  <img src="assets/graphical_abstract.png" width="700">
</p>

---

## 🔧 Environment Configuration

This project was developed and tested under the following environment:

- python == 3.8.2  
- torch == 2.2.0  
- torchvision == 0.17.0  
- numpy == 1.24.0  

💡 Optional: You can install dependencies with pip install -r requirements.txt or manually as shown below.

## 📁 Dataset Setup

The dataset contains **1,463 bridge inspection images**, each annotated with **hierarchical labels** across three levels:  
**structural region → component → defect type**

### 🔽 Download

The full bridge inspection dataset (`BDSJR_dataset.tar.gz`) is available via cloud drive:

🔗 [Cloud Drive Download](https://pan.baidu.com/s/1KUGbvK1DHudWw7j4nlBt1Q) *(access code required)*

> 📌 **Note**: The extraction code is not publicly available. Please contact the authors for academic or collaborative use.  
> 📧 Contact Email: `yxxia@mails.cqjtu.edu.cn`

---

### 🗂 Directory Structure

After unzipping the dataset to the project root directory, it should have the following structure:

```bash
dataset/
├── Annotations/                 # raw label files for each image (multi-level: structural region/component/defect)
├── files/                       # preprocessed label CSVs
│   ├── classification_trainval.csv   # training/validation labels in multi-label format
│   └── classification_test.csv       # test labels in multi-label format
├── JPEGImages/                  # raw bridge inspection images
├── pool_pkls/                   # region-wise image features (Faster R-CNN)
├── co_occurrence_matrix.pkl     # label co-occurrence matrix (label correlation prior)
├── ent_emb.pkl                  # label embeddings from knowledge graph (used as semantic prior)
└── T-G-adj.pkl                  # adjacency matrix of textual graph (structural region–component–defect hierarchy)
```
---

### ✅ Usage Notes

- All annotations and features are **preprocessed** and ready to use.
- No additional conversion or annotation processing is required.


## 📥 Backbone Model: MambaVision-B-1K

The proposed **MKGM** model adopts **[MambaVision-B-1K](https://huggingface.co/nvidia/MambaVision-B-1K)** as the image feature extraction backbone.  
This pretrained model can be easily downloaded and used via [Hugging Face Transformers](https://huggingface.co) or the `timm` library.

```python
from timm import create_model
model = create_model('mambavision_b_1k', pretrained=True)
