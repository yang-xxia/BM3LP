import pickle
import numpy as np

with open('/home/nlp/NLP-Group/YXX/IMF-Pytorch-main/dataset/B2M/BriM852/image_embeddings_stc_domain_pre.pkl', 'rb') as f:
    original_embeddings = pickle.load(f)

image_embeddings = []
for entry in original_embeddings:
    embedding = entry['image_embedding'].astype(np.float32) 
    # 
    embedding = embedding.squeeze()  
    image_embeddings.append(embedding)

image_embeddings = np.array(image_embeddings, dtype=np.float32)

with open('/home/nlp/NLP-Group/YXX/IMF-Pytorch-main/dataset/B2M/BriM852/image_embeddings_array_domain_pre.pkl', 'wb') as f:
    pickle.dump(image_embeddings, f)

print("Image embeddings saved successfully as NumPy array in image_embeddings_array.pkl.")
