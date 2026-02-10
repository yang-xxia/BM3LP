import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
file_path = '/home/nlp/NLP-Group/YXX/BM3LP/dataset/BriM751/data.txt'
data = pd.read_csv(file_path, delimiter='\t')  # 根据实际情况调整分隔符

# First split: 70% train, 30% temp
train_data, temp_data = train_test_split(
    data, 
    test_size=0.3, 
    random_state=42, 
    shuffle=True
)

# Second split: from 30% temp -> 10% val, 20% test
# 10 / (10 + 20) = 1/3
val_data, test_data = train_test_split(
    temp_data, 
    test_size=2/3, 
    random_state=42, 
    shuffle=True
)

# Output sizes
print(f'Training Set Size: {len(train_data)}')
print(f'Validation Set Size: {len(val_data)}')
print(f'Testing Set Size: {len(test_data)}')

# Save to files
train_data.to_csv('/home/nlp/NLP-Group/YXX/BM3LP/dataset/BriM751/train.txt', index=False, sep='\t')
val_data.to_csv('/home/nlp/NLP-Group/YXX/BM3LP/dataset/BriM751/val.txt', index=False, sep='\t')
test_data.to_csv('/home/nlp/NLP-Group/YXX/BM3LP/dataset/BriM751/test.txt', index=False, sep='\t')
