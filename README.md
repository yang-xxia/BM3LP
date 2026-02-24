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

💡 You can install dependencies with pip install -r requirements.txt

## 📁 Dataset Setup

The **Bridge Maintenance Multimodal Knowledge Graph (BM3KG)** currently contains 852 entities, with three modalities: graph structure, images, and text. The dataset includes 9 main relations and 2,150 knowledge triples, and is continually expanding.

> 📌 **Note**: The dataset is not publicly available for direct download. For access, please contact the authors for academic and collaborative use. 

---

The dataset has the following structure:

```bash
dataset/BriM/
├── entity_image/                # Image files corresponding to each entity
├── data.txt                     # All knowledge triples in the dataset
├── entity2id.txt                # Mapping of entity names to IDs
├── entity_description.txt       # Text descriptions for each entity
├── relation2id.txt              # Mapping of relationships to IDs
├── img_features.pkl             # Image embeddings for entities (learned using CLIP and ST-GAT)
├── text_features.pkl            # Text embeddings for entities (learned using CLIP and ST-GAT)
├── train.txt                    # Training set (split from data.txt)
├── val.txt                      # Validation set (split from data.txt)
└── test.txt                     # Test set (split from data.txt)
```
---

## 📥 Image-Text Contrastive Embedding: CLIP (ViT-B/16)

The **BM3LP** model leverages **[CLIP ViT-B/16](https://huggingface.co/openai/clip-vit-base-patch16)** to learn image-text contrastive embeddings, enhancing cross-modal alignment for multimodal defect understanding.
