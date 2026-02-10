import os
import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import pandas as pd
import numpy as np
import pickle

# 
clip_model = CLIPModel.from_pretrained("/home/nlp/NLP-Group/YXX/IMF-Pytorch-main/clip-vit-base-patch16")
processor = CLIPProcessor.from_pretrained("/home/nlp/NLP-Group/YXX/IMF-Pytorch-main/clip-vit-base-patch16")

# 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model.to(device)

# 
description_file = '/home/nlp/NLP-Group/YXX/IMF-Pytorch-main/dataset/B2M/BriM852/entity_description.txt'
descriptions = pd.read_csv(description_file, sep='\t', header=None, names=['id', 'description'])

# 
image_dir = '/home/nlp/NLP-Group/YXX/IMF-Pytorch-main/dataset/B2M/BriM852/entity_image'

# 
def preprocess_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

# 
class STGATLayer(nn.Module):
    def __init__(self, input_dim):
        super(STGATLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=8, dropout=0.1)
    
    def forward(self, nodes_features, adjacency_matrix):
        # nodes_features shape: (num_nodes, input_dim)
        # adjacency_matrix shape: (num_nodes, num_nodes)
        # 
        attention_output, _ = self.attention(nodes_features.unsqueeze(1), nodes_features.unsqueeze(1), nodes_features.unsqueeze(1))
        attention_output = attention_output.squeeze(1)
        # 
        updated_features = torch.matmul(adjacency_matrix, attention_output)
        return updated_features

# 
st_gat_layer = STGATLayer(input_dim=512).to(device)

# 
image_embeddings = []
text_embeddings = []

# 
image_projection = nn.Linear(512, 4096).to(device)

# 
fusion_layer = nn.Linear(1024, 512).to(device)

# 
for idx in range(852):
    # 
    description_row = descriptions[descriptions['id'] == idx]
    if not description_row.empty:
        description = description_row['description'].values[0]
        
        # ）
        single_image_path = os.path.join(image_dir, f"{idx}.jpg")
        
        # 
        multi_image_dir = os.path.join(image_dir, str(idx))
        is_multi_image = os.path.exists(multi_image_dir) and os.path.isdir(multi_image_dir)
        
        # 
        image_embs = []
        
        if is_multi_image:
            # 
            image_files = [f for f in os.listdir(multi_image_dir) if f.endswith('.jpg')]
            image_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))  # 按序号排序
            
            # 
            for image_file in image_files:
                image_path = os.path.join(multi_image_dir, image_file)
                image = preprocess_image(image_path)
                
                if image is not None:
                    # 
                    inputs = processor(images=image, return_tensors="pt").to(device)
                    with torch.no_grad():
                        image_features = clip_model.get_image_features(**inputs)
                        image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # 归一化
                        image_embs.append(image_features.squeeze(0))  # 确保形状为 [512]
                else:
                    # 
                    image_embs.append(torch.normal(0, 0.01, (512,)).to(device))  # 512维
        
        elif os.path.exists(single_image_path):
            # 
            image = preprocess_image(single_image_path)
            
            if image is not None:
                # 
                inputs = processor(images=image, return_tensors="pt").to(device)
                with torch.no_grad():
                    image_features = clip_model.get_image_features(**inputs)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # 归一化
                    image_embs.append(image_features.squeeze(0))  # 确保形状为 [512]
            else:
                # 
                image_embs.append(torch.normal(0, 0.01, (512,)).to(device))  # 512维
        
        else:
            # 
            image_embs.append(torch.normal(0, 0.01, (512,)).to(device))
        
        # 
        with torch.no_grad():
            text_inputs = processor(text=description, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
            text_embedding = clip_model.get_text_features(**text_inputs).squeeze(0)  # 确保形状为 [512]
        
        # 
        if len(image_embs) > 1:
            print(f"Processing entity {idx} with {len(image_embs)} images.")
            # 
            num_nodes = len(image_embs)
            adjacency_matrix = torch.ones((num_nodes, num_nodes))  # 
            
            # 
            nodes_features = torch.stack(image_embs, dim=0)  # shape: (num_nodes, 512)
            print(f"Nodes features shape: {nodes_features.shape}")
            
            # 
            with torch.no_grad():
                updated_features = st_gat_layer(nodes_features, adjacency_matrix.to(device))
            print(f"Updated features shape: {updated_features.shape}")
            
            # 
            fused_features = []
            for i in range(updated_features.size(0)):
                # 
                img_feat = updated_features[i]  # shape: [512]
                txt_feat = text_embedding  # shape: [512]
                fused = torch.cat([img_feat, txt_feat], dim=-1)  # shape: [1024]
                fused = fusion_layer(fused.unsqueeze(0))  # shape: [1, 512]
                fused_features.append(fused.squeeze(0))  # shape: [512]
            
            fused_features = torch.stack(fused_features, dim=0)  # shape: (num_nodes, 512)
            print(f"Fused features shape: {fused_features.shape}")
            
            # 
            image_embedding = torch.mean(fused_features, dim=0, keepdim=True)  # shape: [1, 512]
            print(f"Pooled image embedding shape: {image_embedding.shape}")
        else:
            # 
            image_emb = image_embs[0]  # shape: [512]
            text_emb = text_embedding  # shape: [512]
            
            # 
            fused = torch.cat([image_emb, text_emb], dim=-1)  # shape: [1024]
            image_embedding = fusion_layer(fused.unsqueeze(0))  # shape: [1, 512]
        
        # 
        image_embedding = image_projection(image_embedding)  # shape: [1, 4096]
        
        # 
        image_embeddings.append({
            'id': idx,
            'image_embedding': image_embedding.squeeze(0).detach().cpu().numpy()  # shape: [4096]
        })
        text_embeddings.append({
            'id': idx,
            'text_embedding': text_embedding.detach().cpu().numpy()  # shape: [512]
        })

# 
with open('/home/nlp/NLP-Group/YXX/IMF-Pytorch-main/dataset/B2M/BriM852/image_embeddings_stc_domain_pre.pkl', 'wb') as f:
    pickle.dump(image_embeddings, f)

with open('/home/nlp/NLP-Group/YXX/IMF-Pytorch-main/dataset/B2M/BriM852/text_embeddings_stc_domain_pre.pkl', 'wb') as f:
    pickle.dump(text_embeddings, f)

print("Image and text embeddings generated and saved successfully.")
