import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import optim

class Net:
    def __init__(self, net, params, device):
        self.net = net
        self.params = params
        self.device = device

    def train(self, data):
        n_epoch = self.params['n_epoch']
        self.clf = self.net().to(self.device)
        self.clf.train()
        optimizer = optim.SGD(self.clf.parameters(), **self.params['optimizer_args'])

        if isinstance(data, DataLoader):
            loader = data
        else:
            loader = DataLoader(data, shuffle=True, **self.params['train_args'])

        for epoch in tqdm(range(1, n_epoch + 1), ncols=100):
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                out, e1 = self.clf(x)
                loss = F.cross_entropy(out, y)
                loss.backward()
                optimizer.step()

    def predict(self, data):
        self.clf.eval()
        if isinstance(data, DataLoader):
            n_total = sum(batch[0].size(0) for batch in data)
            loader = data
        else:
            n_total = len(data)
            loader = DataLoader(data, shuffle=False, **self.params['test_args'])

        preds = torch.zeros(n_total, dtype=torch.long)
        with torch.no_grad():
            for x, y, idxs in loader:
                x = x.to(self.device)
                out, e1 = self.clf(x)
                pred = out.max(1)[1].cpu()
                preds[idxs] = pred
        return preds

    def predict_prob(self, data):
        self.clf.eval()
        if isinstance(data, DataLoader):
            loader = data
        else:
            loader = DataLoader(data, shuffle=False, **self.params['test_args'])

        all_probs = []
        with torch.no_grad():
            for x, y, idxs in loader:
                x = x.to(self.device)
                out, _ = self.clf(x)
                probs = F.softmax(out, dim=1).cpu()
                all_probs.append(probs)
        return torch.cat(all_probs, dim=0)

    def predict_prob_dropout(self, data, n_drop=10):
        self.clf.train()
        if isinstance(data, DataLoader):
            loader = data
            n_total = sum(batch[0].size(0) for batch in data)
        else:
            loader = DataLoader(data, shuffle=False, **self.params['test_args'])
            n_total = len(data)

        # Descobre número de classes numa passada rápida
        with torch.no_grad():
            sample_batch = next(iter(loader))
            x_sample = sample_batch[0].to(self.device)
            out_sample, _ = self.clf(x_sample)
            num_classes = out_sample.size(1)

        probs = torch.zeros(n_total, num_classes)
        for _ in range(n_drop):
            with torch.no_grad():
                for x, y, idxs in loader:
                    x = x.to(self.device)
                    out, _ = self.clf(x)
                    p = F.softmax(out, dim=1).cpu()
                    probs[idxs] += p
        probs /= n_drop
        return probs

    def predict_prob_dropout_split(self, data, n_drop=10):
        self.clf.train()
        if isinstance(data, DataLoader):
            loader = data
            n_total = sum(batch[0].size(0) for batch in data)
        else:
            loader = DataLoader(data, shuffle=False, **self.params['test_args'])
            n_total = len(data)

        with torch.no_grad():
            sample_batch = next(iter(loader))
            x_sample = sample_batch[0].to(self.device)
            out_sample, _ = self.clf(x_sample)
            num_classes = out_sample.size(1)

        probs = torch.zeros(n_drop, n_total, num_classes)
        for i in range(n_drop):
            with torch.no_grad():
                for x, y, idxs in loader:
                    x = x.to(self.device)
                    out, _ = self.clf(x)
                    p = F.softmax(out, dim=1).cpu()
                    probs[i][idxs] = p
        return probs

    def get_embeddings(self, data):
        self.clf.eval()
        if isinstance(data, DataLoader):
            loader = data
            n_total = sum(batch[0].size(0) for batch in data)
        else:
            loader = DataLoader(data, shuffle=False, **self.params['test_args'])
            n_total = len(data)

        embedding_dim = self.clf.get_embedding_dim()
        embeddings = torch.zeros(n_total, embedding_dim)
        with torch.no_grad():
            for x, y, idxs in loader:
                x = x.to(self.device)
                out, e1 = self.clf(x)
                embeddings[idxs] = e1.cpu()
        return embeddings
        

class MNIST_Net(nn.Module):
    def __init__(self):
        super(MNIST_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        e1 = F.relu(self.fc1(x))
        x = F.dropout(e1, training=self.training)
        x = self.fc2(x)
        return x, e1

    def get_embedding_dim(self):
        return 50


class SVHN_Net(nn.Module):
    def __init__(self):
        super(SVHN_Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1152, 400)
        self.fc2 = nn.Linear(400, 50)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
        x = x.view(-1, 1152)
        x = F.relu(self.fc1(x))
        e1 = F.relu(self.fc2(x))
        x = F.dropout(e1, training=self.training)
        x = self.fc3(x)
        return x, e1

    def get_embedding_dim(self):
        return 50


class CIFAR10_Net(nn.Module):
    def __init__(self):
        super(CIFAR10_Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(1024, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 1024)
        e1 = F.relu(self.fc1(x))
        x = F.dropout(e1, training=self.training)
        x = self.fc2(x)
        return x, e1

    def get_embedding_dim(self):
        return 50

