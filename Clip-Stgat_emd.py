import os
import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import pandas as pd
import numpy as np
import pickle

# 加载 CLIP 模型和处理器
clip_model = CLIPModel.from_pretrained("/home/nlp/NLP-Group/YXX/IMF-Pytorch-main/clip-vit-base-patch16")
processor = CLIPProcessor.from_pretrained("/home/nlp/NLP-Group/YXX/IMF-Pytorch-main/clip-vit-base-patch16")

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model.to(device)

# 加载实体描述
description_file = '/home/nlp/NLP-Group/YXX/IMF-Pytorch-main/dataset/B2M/BriM852/entity_description.txt'
descriptions = pd.read_csv(description_file, sep='\t', header=None, names=['id', 'description'])

# 图像文件夹路径
image_dir = '/home/nlp/NLP-Group/YXX/IMF-Pytorch-main/dataset/B2M/BriM852/entity_image'

# 预处理图像
def preprocess_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

# ST-GAT 层
class STGATLayer(nn.Module):
    def __init__(self, input_dim):
        super(STGATLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=8, dropout=0.1)
    
    def forward(self, nodes_features, adjacency_matrix):
        # nodes_features shape: (num_nodes, input_dim)
        # adjacency_matrix shape: (num_nodes, num_nodes)
        # 使用 attention 进行特征更新
        attention_output, _ = self.attention(nodes_features.unsqueeze(1), nodes_features.unsqueeze(1), nodes_features.unsqueeze(1))
        attention_output = attention_output.squeeze(1)
        # 应用邻接矩阵
        updated_features = torch.matmul(adjacency_matrix, attention_output)
        return updated_features

# 初始化 ST-GAT 层
st_gat_layer = STGATLayer(input_dim=512).to(device)

# 初始化嵌入列表
image_embeddings = []
text_embeddings = []

# 定义线性层以调整图像嵌入维度
image_projection = nn.Linear(512, 4096).to(device)

# 融合文本特征的线性层
fusion_layer = nn.Linear(1024, 512).to(device)

# 遍历每个实体
for idx in range(852):
    # 获取实体描述
    description_row = descriptions[descriptions['id'] == idx]
    if not description_row.empty:
        description = description_row['description'].values[0]
        
        # 构造可能的图像路径（单张图像情况）
        single_image_path = os.path.join(image_dir, f"{idx}.jpg")
        
        # 检查实体是否有对应的多张图像文件夹
        multi_image_dir = os.path.join(image_dir, str(idx))
        is_multi_image = os.path.exists(multi_image_dir) and os.path.isdir(multi_image_dir)
        
        # 初始化图像嵌入列表
        image_embs = []
        
        if is_multi_image:
            # 获取实体下所有图像文件，并按序号排序
            image_files = [f for f in os.listdir(multi_image_dir) if f.endswith('.jpg')]
            image_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))  # 按序号排序
            
            # 遍历实体下的每张图像
            for image_file in image_files:
                image_path = os.path.join(multi_image_dir, image_file)
                image = preprocess_image(image_path)
                
                if image is not None:
                    # 生成图像嵌入
                    inputs = processor(images=image, return_tensors="pt").to(device)
                    with torch.no_grad():
                        image_features = clip_model.get_image_features(**inputs)
                        image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # 归一化
                        image_embs.append(image_features.squeeze(0))  # 确保形状为 [512]
                else:
                    # 生成接近零的随机向量
                    image_embs.append(torch.normal(0, 0.01, (512,)).to(device))  # 512维
        
        elif os.path.exists(single_image_path):
            # 单张图像情况
            image = preprocess_image(single_image_path)
            
            if image is not None:
                # 生成图像嵌入
                inputs = processor(images=image, return_tensors="pt").to(device)
                with torch.no_grad():
                    image_features = clip_model.get_image_features(**inputs)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # 归一化
                    image_embs.append(image_features.squeeze(0))  # 确保形状为 [512]
            else:
                # 生成接近零的随机向量
                image_embs.append(torch.normal(0, 0.01, (512,)).to(device))  # 512维
        
        else:
            # 实体没有对应的图像，生成接近零的随机向量
            image_embs.append(torch.normal(0, 0.01, (512,)).to(device))
        
        # 生成文本嵌入
        with torch.no_grad():
            text_inputs = processor(text=description, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
            text_embedding = clip_model.get_text_features(**text_inputs).squeeze(0)  # 确保形状为 [512]
        
        # 如果实体下有多张图像，进行时空关联特征捕捉
        if len(image_embs) > 1:
            print(f"Processing entity {idx} with {len(image_embs)} images.")
            # 构建邻接矩阵（假设所有图像相互关联）
            num_nodes = len(image_embs)
            adjacency_matrix = torch.ones((num_nodes, num_nodes))  # 全连接图
            
            # 将图像嵌入组合成一个矩阵
            nodes_features = torch.stack(image_embs, dim=0)  # shape: (num_nodes, 512)
            print(f"Nodes features shape: {nodes_features.shape}")
            
            # 使用 ST-GAT 更新节点特征
            with torch.no_grad():
                updated_features = st_gat_layer(nodes_features, adjacency_matrix.to(device))
            print(f"Updated features shape: {updated_features.shape}")
            
            # 将文本嵌入广播到每个图像节点，并与图像特征融合
            fused_features = []
            for i in range(updated_features.size(0)):
                # 确保两个张量的维度一致
                img_feat = updated_features[i]  # shape: [512]
                txt_feat = text_embedding  # shape: [512]
                fused = torch.cat([img_feat, txt_feat], dim=-1)  # shape: [1024]
                fused = fusion_layer(fused.unsqueeze(0))  # shape: [1, 512]
                fused_features.append(fused.squeeze(0))  # shape: [512]
            
            fused_features = torch.stack(fused_features, dim=0)  # shape: (num_nodes, 512)
            print(f"Fused features shape: {fused_features.shape}")
            
            # 对融合后的特征进行池化，得到一个综合的图像嵌入
            image_embedding = torch.mean(fused_features, dim=0, keepdim=True)  # shape: [1, 512]
            print(f"Pooled image embedding shape: {image_embedding.shape}")
        else:
            # 单张图像或没有图像的情况
            image_emb = image_embs[0]  # shape: [512]
            text_emb = text_embedding  # shape: [512]
            
            # 融合图像和文本特征
            fused = torch.cat([image_emb, text_emb], dim=-1)  # shape: [1024]
            image_embedding = fusion_layer(fused.unsqueeze(0))  # shape: [1, 512]
        
        # 将图像嵌入调整为 4096 维
        image_embedding = image_projection(image_embedding)  # shape: [1, 4096]
        
        # 保存嵌入 - 关键修改：添加.detach()
        image_embeddings.append({
            'id': idx,
            'image_embedding': image_embedding.squeeze(0).detach().cpu().numpy()  # shape: [4096]
        })
        text_embeddings.append({
            'id': idx,
            'text_embedding': text_embedding.detach().cpu().numpy()  # shape: [512]
        })

# 将嵌入保存为 pkl 文件
with open('/home/nlp/NLP-Group/YXX/IMF-Pytorch-main/dataset/B2M/BriM852/image_embeddings_stc_domain_pre.pkl', 'wb') as f:
    pickle.dump(image_embeddings, f)

with open('/home/nlp/NLP-Group/YXX/IMF-Pytorch-main/dataset/B2M/BriM852/text_embeddings_stc_domain_pre.pkl', 'wb') as f:
    pickle.dump(text_embeddings, f)

print("Image and text embeddings generated and saved successfully.")