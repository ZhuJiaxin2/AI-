import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer
import math
import torch.nn.functional as F
import collections

def read_imdb(data_dir):
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    def read_data(data_dir):
        data, labels = [], []
        for label in ['pos', 'neg']:
            folder_name = os.path.join(data_dir, label)
            for file in os.listdir(folder_name):
                with open(os.path.join(folder_name, file), 'rb') as f:
                    review = f.read().decode('utf-8').replace('\n','')
                    data.append(review)
                    labels.append(1 if label == 'pos' else 0)
        return data, labels
    train_data, train_labels = read_data(train_dir)
    test_data, test_labels = read_data(test_dir)
    return train_data, train_labels, test_data, test_labels

def tokenize(lines): 
    """将文本行拆分为单词或字符词元"""
    return [line.split() for line in lines]

class MyTokenizer:
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率排序
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # 未知词元的索引为0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # 未知词元的索引为0
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs

    def encode(self, text, max_len):
        # Tokenize the text and pad/truncate to max_len
        if not isinstance(text, str):
            text = str(text)
        tokens = [self.token_to_idx[word] for word in text.split()]
        if len(tokens) < max_len:
            tokens += [0] * (max_len - len(tokens))
        else:
            tokens = tokens[:max_len]
        return tokens

    def encode_plus(self, text, max_len):
        # Encode the text and return the input_ids and attention_mask
        input_ids = self.encode(text, max_len)
        attention_mask = [1] * len(input_ids)
        if len(input_ids) < max_len:
            input_ids += [0] * (max_len - len(input_ids))
            attention_mask += [0] * (max_len - len(attention_mask))
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
    
    
def count_corpus(tokens):  #@save
    """统计词元的频率"""
    # 这里的tokens是1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成一个列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

# def tokenize(texts, tokenizer):
#     # Tokenize the texts using the provided tokenizer
#     tokenized_texts = []
#     for text in texts:
#         tokenized_text = tokenizer.encode_plus(text, max_len=512)
#         tokenized_texts.append(tokenized_text)
#     return tokenized_texts

class IMDBDataset(Dataset):
    def __init__(self, data, labels, tokenizer, max_len):
        self.data = data
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data[index]
        label = self.labels[index]
        encoding = self.tokenizer.encode_plus(
            text,
            max_len=self.max_len
        )
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'label': torch.tensor(label, dtype=torch.long)
        }

class BertForSequenceClassification(nn.Module):
    def __init__(self, num_classes, vocab_size, hidden_size, num_layers, num_heads, max_len, dropout):
        super(BertForSequenceClassification, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.encoder = Encoder(hidden_size, num_layers, num_heads, dropout)
        self.pooler = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.max_len = max_len

    def forward(self, input_ids, attention_mask):
        input_embed = self.embedding(input_ids)
        input_embed *= attention_mask.unsqueeze(-1)
        input_embed = input_embed.transpose(0, 1)
        encoder_output = self.encoder(input_embed)
        pooled_output = self.pooler(encoder_output[-1])
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits
    
class Encoder(nn.Module):
    def __init__(self, hidden_size, num_layers, num_heads, dropout):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(hidden_size, num_heads, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x
    
class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(hidden_size, num_heads)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.self_attn(x)
        x = self.norm1(x + residual)
        x = self.dropout1(x)
        residual = x
        x = self.feed_forward(x)
        x = self.norm2(x + residual)
        x = self.dropout2(x)
        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        batch_size, seq_len, hidden_size = x.size()
        q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_size)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        output = self.out(attn_output)
        return output
    
class PositionwiseFeedForward(nn.Module):
    def __init__(self, hidden_size):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size * 4)
        self.fc2 = nn.Linear(hidden_size * 4, hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    label = [item['label'] for item in batch]
    input_ids = pad_sequence(input_ids, batch_first=True)
    attention_mask = pad_sequence(attention_mask, batch_first=True)
    label = torch.stack(label)
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'label': label
    }

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0
    train_acc = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        label = batch['label'].to(device)
        optimizer.zero_grad()
        output = model(input_ids, attention_mask)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pred = torch.argmax(output, dim=1)
        train_acc += torch.sum(pred == label).item()
    train_loss /= len(train_loader)
    train_acc /= len(train_loader.dataset)
    return train_loss, train_acc

def evaluate(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    test_acc = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label = batch['label'].to(device)
            output = model(input_ids, attention_mask)
            loss = criterion(output, label)
            test_loss += loss.item()
            pred = torch.argmax(output, dim=1)
            test_acc += torch.sum(pred == label).item()
    test_loss /= len(test_loader)
    test_acc /= len(test_loader.dataset)
    return test_loss, test_acc


def main():
    data_dir = './aclImdb'
    train_data, train_labels, test_data, test_labels = read_imdb(data_dir)
    train_features = tokenize(train_data)
    train_vocab = MyTokenizer(train_features, min_freq=5, reserved_tokens=['<pad>'])
    test_features = tokenize(test_data)
    test_vocab = MyTokenizer(test_features, min_freq=5, reserved_tokens=['<pad>'])
    train_dataset = IMDBDataset(train_data, train_labels, train_vocab, max_len=512)
    test_dataset = IMDBDataset(test_data, test_labels, test_vocab, max_len=512)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = BertForSequenceClassification(num_classes=2, vocab_size=30522, hidden_size=768, num_layers=12, num_heads=12, max_len=512, dropout=0.1).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    num_epochs = 5
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')

if __name__ == '__main__':
    main()
