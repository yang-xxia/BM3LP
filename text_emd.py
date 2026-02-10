import pickle
import numpy as np

# 加载现有的 text_embeddings768.pkl 文件
with open('/home/nlp/NLP-Group/YXX/IMF-Pytorch-main/dataset/B2M/BriM852/text_embeddings_stc_domain_pre.pkl', 'rb') as f:
    original_embeddings = pickle.load(f)

# 提取嵌入部分
text_embeddings = []
for entry in original_embeddings:
    embedding = entry['text_embedding'].astype(np.float32)  # 确保是 float32 类型
    # 使用 squeeze 去掉多余的维度
    embedding = embedding.squeeze()  # 将 (1, 768) 转换为 (768,)
    text_embeddings.append(embedding)

# 使用 np.array() 创建统一的 NumPy 数组
text_embeddings = np.array(text_embeddings, dtype=np.float32)

# 保存新的只包含嵌入的 pkl 文件，使用新的文件名
with open('/home/nlp/NLP-Group/YXX/IMF-Pytorch-main/dataset/B2M/BriM852/text_embeddings_array_domain_pre.pkl', 'wb') as f:
    pickle.dump(text_embeddings, f)

print("Text embeddings saved successfully as NumPy array in text_embeddings_array.pkl.")