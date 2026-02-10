import pickle
import numpy as np

with open('/home/nlp/NLP-Group/YXX/IMF-Pytorch-main/dataset/B2M/BriM852/text_embeddings_stc_domain_pre.pkl', 'rb') as f:
    original_embeddings = pickle.load(f)

text_embeddings = []
for entry in original_embeddings:
    embedding = entry['text_embedding'].astype(np.float32)  
   
    embedding = embedding.squeeze()  
    text_embeddings.append(embedding)

text_embeddings = np.array(text_embeddings, dtype=np.float32)

with open('/home/nlp/NLP-Group/YXX/IMF-Pytorch-main/dataset/B2M/BriM852/text_embeddings_array_domain_pre.pkl', 'wb') as f:
    pickle.dump(text_embeddings, f)

print("Text embeddings saved successfully as NumPy array in text_embeddings_array.pkl.")
