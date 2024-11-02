# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from data_preprocessing import preprocess_text, clean_text
from seq2seq_model import Encoder, Decoder, Seq2Seq
import pandas as pd
from sklearn.model_selection import train_test_split
import re

# Hyperparameters
INPUT_DIM = 10000  # Vocabulary size
OUTPUT_DIM = 10000  # Vocabulary size
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HIDDEN_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
LEARNING_RATE = 0.001
N_EPOCHS = 10
BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dummy Dataset (Replace with actual data)
# For illustration, let's create a small dataset
# In practice, use a large dataset like CNN/DailyMail

data = {
    'article': [
        "Artificial intelligence is transforming industries. It is used in healthcare to assist doctors.",
        "Machine learning models require large amounts of data. Data preprocessing is crucial for model performance."
    ],
    'summary': [
        "AI is revolutionizing industries, especially in healthcare by assisting doctors.",
        "Machine learning needs extensive data, and data preprocessing is essential for effective models."
    ]
}

df = pd.DataFrame(data)

# Preprocess the data
df['article_clean'] = df['article'].apply(clean_text)
df['summary_clean'] = df['summary'].apply(clean_text)

# Tokenize the data
df['article_tokens'] = df['article_clean'].apply(preprocess_text)
df['summary_tokens'] = df['summary_clean'].apply(preprocess_text)

# Build Vocabulary (Simple example)
from collections import Counter

def build_vocab(tokenized_texts, max_size=10000, min_freq=2):
    counter = Counter()
    for tokens in tokenized_texts:
        counter.update(tokens)
    vocab = {'<pad>': 0, '<sos>':1, '<eos>':2, '<unk>':3}
    idx = 4
    for word, freq in counter.most_common(max_size):
        if freq >= min_freq:
            vocab[word] = idx
            idx +=1
    return vocab

vocab = build_vocab(df['article_tokens'].tolist() + df['summary_tokens'].tolist())

# Save vocab for later use
import json
with open('vocab.json', 'w') as f:
    json.dump(vocab, f)

# Define Dataset Class
class SummarizationDataset(Dataset):
    def __init__(self, articles, summaries, vocab):
        self.articles = articles
        self.summaries = summaries
        self.vocab = vocab
    
    def __len__(self):
        return len(self.articles)
    
    def numericalize(self, tokens):
        return [self.vocab.get(token, self.vocab['<unk>']) for token in tokens]
    
    def __getitem__(self, idx):
        article = self.numericalize(self.articles[idx])
        summary = [self.vocab['<sos>']] + self.numericalize(self.summaries[idx]) + [self.vocab['<eos>']]
        return torch.tensor(article), torch.tensor(summary)

# Split the data
train_articles, val_articles, train_summaries, val_summaries = train_test_split(
    df['article_tokens'].tolist(),
    df['summary_tokens'].tolist(),
    test_size=0.2,
    random_state=42
)

# Create Datasets and DataLoaders
train_dataset = SummarizationDataset(train_articles, train_summaries, vocab)
val_dataset = SummarizationDataset(val_articles, val_summaries, vocab)

def collate_fn(batch):
    articles, summaries = zip(*batch)
    article_lengths = [len(a) for a in articles]
    summary_lengths = [len(s) for s in summaries]
    
    max_article_len = max(article_lengths)
    max_summary_len = max(summary_lengths)
    
    padded_articles = [torch.cat([a, torch.zeros(max_article_len - len(a)).long()]) for a in articles]
    padded_summaries = [torch.cat([s, torch.zeros(max_summary_len - len(s)).long()]) for s in summaries]
    
    return torch.stack(padded_articles), torch.stack(padded_summaries)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# Initialize the Model
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HIDDEN_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HIDDEN_DIM, N_LAYERS, DEC_DROPOUT)

model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)

# Initialize Optimizer and Loss Function
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index=0)

# Training Loop
for epoch in range(N_EPOCHS):
    model.train()
    epoch_loss = 0
    
    for batch_idx, (src, trg) in enumerate(train_loader):
        src = src.to(DEVICE)
        trg = trg.to(DEVICE)
        
        optimizer.zero_grad()
        output = model(src, trg)
        
        # Output: [batch_size, trg_len, output_dim]
        output_dim = output.shape[-1]
        output = output[:,1:,:].reshape(-1, output_dim)  # Exclude first token
        trg = trg[:,1:].reshape(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for src, trg in val_loader:
            src = src.to(DEVICE)
            trg = trg.to(DEVICE)
            
            output = model(src, trg, teacher_forcing_ratio=0)  # No teacher forcing for validation
            
            output_dim = output.shape[-1]
            output = output[:,1:,:].reshape(-1, output_dim)
            trg = trg[:,1:].reshape(-1)
            
            loss = criterion(output, trg)
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(val_loader)
    
    print(f'Epoch [{epoch+1}/{N_EPOCHS}], Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
