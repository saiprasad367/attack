import pandas as pd
from transformers import BertTokenizer
import numpy as np

# Load CSIC 2010 dataset (assumes CSV format with 'request' and 'label' columns)
def load_data(file_path):
    data = pd.read_csv(file_path)  # Adjust based on actual dataset format
    return data['request'].values, data['label'].values

# Preprocess HTTP requests for BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def preprocess_requests(requests, max_length=128):
    encodings = tokenizer(
        requests.tolist(),
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='tf'
    )
    return encodings['input_ids'], encodings['attention_mask']

if __name__ == "__main__":
    requests, labels = load_data('data/shopeasy_web_requests.csv')
    input_ids, attention_masks = preprocess_requests(requests)
    np.save('data/input_ids.npy', input_ids.numpy())
    np.save('data/attention_masks.npy', attention_masks.numpy())
    np.save('data/labels.npy', labels)