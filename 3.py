import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import re
import os

# 读取IMDB数据集
data_dir = './aclImdb'

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

train_data, train_labels, test_data, test_labels = read_imdb(data_dir)

# 划分训练集和测试集
X_train, X_val, y_train, y_val = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)

# 加载 BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 定义数据集类
class IMDBDataset(Dataset):
    def __init__(self, X, y, tokenizer, max_len):
        self.X = X
        self.y = y
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        text = self.X[idx]
        label = self.y[idx]

        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'][0],
            'attention_mask': inputs['attention_mask'][0],
            'token_type_ids': inputs['token_type_ids'][0],
            'label': torch.tensor(label, dtype=torch.long)
        }

# 定义模型
class BERTClassifier(nn.Module):
    def __init__(self, bert_model, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits

# 初始化模型和优化器
model = BERTClassifier(BertModel.from_pretrained('bert-base-uncased'), num_classes=2)
optimizer = optim.Adam(model.parameters(), lr=2e-5)

# 定义训练函数
def train(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, token_type_ids)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print('stepped foward')

        total_loss += loss.item()

    return total_loss / len(dataloader)

# 定义测试函数
def test(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    total_correct = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, dim=1)
            total_correct += (predicted == labels).sum().item()

    return total_loss / len(dataloader), total_correct / len(dataloader.dataset)

# 训练模型
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model.to(device)

train_dataset = IMDBDataset(X_train, y_train, tokenizer, max_len=512)
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

val_dataset = IMDBDataset(X_val, y_val, tokenizer, max_len=512)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

criterion = nn.CrossEntropyLoss()
num_epochs = 5
for epoch in range(num_epochs):
    print(torch.backends.mps.is_available())
    train_loss = train(model, train_dataloader, optimizer, criterion)
    val_loss, val_acc = test(model, val_dataloader, criterion)
    print(f'Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}')

# 在测试集上评估模型
test_dataset = IMDBDataset(test_data, test_labels, tokenizer, max_len=512)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

test_loss, test_acc = test(model, test_dataloader, criterion)
print(f'Test Loss = {test_loss:.4f}, Test Acc = {test_acc:.4f}')

# 保存模型
torch.save(model.state_dict(), 'imdb_bert_model.pth')