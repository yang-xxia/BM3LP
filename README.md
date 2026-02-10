# 🏗️ BM3LP

**Bridge Maintenance Multimodal Link Prediction (BM3LP)** via spatiotemporal feature fusion and cross-modal contrastive interaction.

## 🔍 Overview

The BM3LP (Bridge Maintenance Multimodal Link Prediction) model aims to tackle the inconsistency across image modalities, as well as the challenges in expressing and reasoning spatiotemporal characteristics of bridge defects. By incorporating both image and text modalities, the model integrates spatiotemporal semantic information to provide an accurate understanding of defects, ultimately contributing to more effective treatment decision-making in bridge maintenance.

- 📷 **Graph Structure Encoder**: Uses DFS and RGAT to capture multi-hop dependencies between defect and structural data, overcoming limitations in long-range hierarchical modeling.
- 🌐 **Spatiotemporal Feature Modeling**: Combines CLIP and ST-GAT to jointly model image and text modalities, generating virtual embeddings to address missing image data and enhance spatiotemporal evolution.  
- 🔁 **cross-modal contrastive interaction**: Applies a progressive cross-modal contrast interaction strategy to improve feature retention, modality similarity, and cross-modal semantic association, addressing issues like feature forgetting and modality imbalance.  
- 🧠 **MMKG & Link Prediction**: Constructs a Bridge Maintenance MMKG and BriM dataset, demonstrating model advantages through multiple comparative experiments.

## 🔧 Environment Configuration

This project was developed and tested under the following environment:

- python == 3.8.2  
- torch==2.3.0
- torch-geometric==2.6.1
- transformers==4.46.3
- numpy==1.24.4
- scipy==1.10.1
- scikit-learn==1.3.2
- pandas==2.0.3
- matplotlib==3.7.5
- seaborn==0.13.2
- tqdm==4.67.1
- requests==2.32.3  

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
