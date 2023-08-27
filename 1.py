import torch
import torch.nn as nn
from torch.utils import data
import os
import re
import collections
import numpy as np
import matplotlib.pyplot as plt
from d2l import torch as d2l#for glove 权重

data_dir = './aclImdb'

def read_imdb(data_dir):
    #作用：data按文件装入一个list
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
    train_data = read_data(train_dir)
    test_data = read_data(test_dir)
    return train_data, test_data

def tokenize(data_list):
    #[[split的评论]]
    # filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    data_token = []
    filters = ['!','"','#','$','%','&','\(','\)','\*','\+',',','-','\.','/',':',';','<','=','>','\?','@'
        ,'\[','\\','\]','^','_','`','\{','\|','\}','~','\t','\n','\x97','\x96','”','“',]
    for text in data_list:
        # sub方法是替换
        text = re.sub("<.*?>"," ",text,flags=re.S)	# 去掉<...>中间的内容，主要是文本内容中存在<br/>等内容
        text = re.sub("|".join(filters)," ",text,flags=re.S)	# 替换掉特殊字符，'|'是把所有要匹配的特殊字符连在一起
        data_token.append(text.split())
    return data_token

class Vocab: 
    """文本词表"""
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

def count_corpus(tokens):  #@save
    """统计词元的频率"""
    # 这里的tokens是1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成一个列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

def truncate_pad(line, num_steps, padding_token):
    """截断或填充文本序列"""
    if len(line) > num_steps:
        return line[:num_steps]  # 截断
    return line + [padding_token] * (num_steps - len(line))  # 填充

def load_array(data_arrays, batch_size, is_train=True):
    #TODO torch.utils.data
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

train_data, test_data = read_imdb(data_dir)
train_tokens, test_tokens = tokenize(train_data[0]), tokenize(test_data[0])
vocab = Vocab(train_tokens, min_freq=5, reserved_tokens=['<pad>'])

# print(vocab[train_tokens[:2]])
# plt.plot([len(line) for line in train_tokens])
# plt.show()

num_steps = 500
train_features = torch.tensor([truncate_pad(vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])#vocab[line_list]中输入会输出相应索引
test_features = torch.tensor([truncate_pad(vocab[line], num_steps, vocab['<pad>']) for line in test_tokens])

batch_size=64
train_iter = load_array((train_features, torch.tensor(train_data[1])), batch_size)
test_iter = load_array((test_features, torch.tensor(test_data[1])), batch_size, is_train=False)


class BiRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens,
                 num_layers, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 将bidirectional设置为True以获取双向循环神经网络
        self.encoder = nn.LSTM(embed_size, num_hiddens, num_layers=num_layers,
                                bidirectional=True)
        self.decoder = nn.Linear(4 * num_hiddens, 2)

    def forward(self, inputs):
        # inputs的形状是（批量大小，时间步数）
        # 因为长短期记忆网络要求其输入的第一个维度是时间维，
        # 所以在获得词元表示之前，输入会被转置。
        # 输出形状为（时间步数，批量大小，词向量维度）
        embeddings = self.embedding(inputs.T)
        self.encoder.flatten_parameters()
        # 返回上一个隐藏层在不同时间步的隐状态，
        # outputs的形状是（时间步数，批量大小，2*隐藏单元数）
        outputs, _ = self.encoder(embeddings)
        # 连结初始和最终时间步的隐状态，作为全连接层的输入，
        # 其形状为（批量大小，4*隐藏单元数）
        encoding = torch.cat((outputs[0], outputs[-1]), dim=1)
        outs = self.decoder(encoding)
        return outs
    

embed_size, num_hiddens, num_layers = 100, 100, 2
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
net = BiRNN(len(vocab), embed_size, num_hiddens, num_layers)

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.LSTM:
        for param in m._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(m._parameters[param])

net.apply(init_weights)

glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]
net.embedding.weight.data.copy_(embeds)
net.embedding.weight.requires_grad = False


#@save
def train_batch(net, X, y, loss, trainer, device):
    X = X.to(device)
    y = y.to(device)
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = accuracy(pred, y)
    return train_loss_sum, train_acc_sum

class Accumulator: 
    def __init__(self, n):
        self.data = [0.0] * n
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    def __getitem__(self, idx):
        return self.data[idx]
    
def accuracy(y_hat, y): #@save """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net, data_iter, device):
    if isinstance(net, torch.nn.Module):
        net.eval() # 将模型设置为评估模式
    metric = Accumulator(2) # 正确预测数、预测总数 
    with torch.no_grad():
        for X, y in data_iter:
            X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), y.numel())    
    return metric[0] / metric[1]


lr, num_epochs = 0.01, 5
loss = nn.CrossEntropyLoss(reduction="none")
trainer = torch.optim.Adam(net.parameters(), lr=lr)

for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}:')
    for i, (features, labels) in enumerate(train_iter):
        metric = Accumulator(4)
        # features = features.float()
        # features = torch.nn.functional.normalize(features)#值域矫正：对输入数据进行标准化，缩放到均值0、方差1范围内。避免loss出现nan
        # features = features.long()
        l, acc = train_batch(net, features, labels, loss, trainer, device)
        metric.add(l, acc, labels.shape[0], labels.numel())
        # test_acc = evaluate_accuracy(net, test_iter, device)
        print(f'loss {metric[0] / metric[2]:.3f}, train acc '
            f'{metric[1] / metric[3]:.3f}')
            #, test acc {test_acc:.3f}')