import numpy as np
import torch
import time
from collections import Counter
from torchvision import datasets

class Data:
    def __init__(self, X_train, Y_train, X_test, Y_test, handler):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.handler = handler

        self.n_pool = len(X_train)
        self.n_test = len(X_test)

        self.labeled_idxs = np.zeros(self.n_pool, dtype=bool)
        self.labels = np.full(self.n_pool, None, dtype=object)  # Inicializa com None

    def initialize_labels(self, num):
        t_ini = time.time()

        tmp_idxs = np.arange(self.n_pool)
        np.random.shuffle(tmp_idxs)
        selected_idxs = tmp_idxs[:num]

        self.labeled_idxs[selected_idxs] = True

        labels_selecionados = self.Y_train[selected_idxs]
        self.update_labeled_data(selected_idxs, labels_selecionados)

        t_fim = time.time()

    def get_labeled_data(self):
        labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs]
        return labeled_idxs, self.handler(self.X_train[labeled_idxs], self.Y_train[labeled_idxs])

    def get_unlabeled_data(self):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        return unlabeled_idxs, self.handler(self.X_train[unlabeled_idxs], self.Y_train[unlabeled_idxs])

    def get_train_data(self):
        return self.labeled_idxs.copy(), self.handler(self.X_train, self.Y_train)

    def get_test_data(self):
        return self.handler(self.X_test, self.Y_test)

    def cal_test_acc(self, preds):
        return 1.0 * (self.Y_test==preds).sum().item() / self.n_test

    def subsample(self, indexes):
        """
        Retorna um subconjunto dos dados baseado nos índices fornecidos
        e mantém os índices originais para referência.
        """
        return Data(
            self.X_train[indexes],
            self.Y_train[indexes],
            self.X_test,
            self.Y_test,
            self.handler
        ), indexes

    def get_labels(self, indexes):
        """
        Retorna os rótulos (labels) correspondentes aos índices fornecidos.
        """
        return self.Y_train[indexes].tolist()

    def update_labeled_data(self, indices, labels, test_preds=None):
        """
        Atualiza o conjunto rotulado com os índices e rótulos dados, e retorna:
         - rótulos_atualizados: quantos rótulos mudaram (None->novo ou corrigido)
         - porcentagem_corrigida: % de rótulos no batch que eram diferentes dos anteriores
         - novas_classes_descobertas: conjunto de classes recém vistas
         - distrib_classes: contagem de cada classe no pool rotulado
         - num_rotulos_adicionados: tamanho do batch
         - porcentagem_rotulada: % do pool total que já está rotulado
         - acuracia_por_classe: dict {classe: acurácia no teste} se test_preds fornecido

        :param indices: array de índices do pool que serão rotulados
        :param labels:  array de rótulos (mesmo tamanho de indices)
        :param test_preds: array de predições (numpy ou torch) para X_test, opcional
        """
        # 1) marcas como rotulados e guarda antigos
        self.labeled_idxs[indices] = True
        labels_antigos = self.labels[indices]

        # 2) antes de atualizar, captura o conjunto de classes já vistas
        classes_antes = set(
            lbl for lbl in self.labels
            if lbl is not None
        )

        # 3) faz a atualização dos novos rótulos
        self.labels[indices] = labels

        # 4) estatísticas de correção
        num_rotulos_adicionados = len(indices)
        num_corrigidos = int(np.sum(labels_antigos != labels))
        porcentagem_corrigida = (
            100.0 * num_corrigidos / num_rotulos_adicionados
            if num_rotulos_adicionados else 0.0
        )

        # 5) novas classes descobertas (apenas as do batch)
        novas_classes_descobertas = set(labels) - classes_antes

        # 6) distribuição de classes no pool rotulado
        rotulados = [lbl for lbl in self.labels if lbl is not None]
        distrib_classes = dict(Counter(rotulados))

        # 7) porcentagem do pool já rotulado
        porcentagem_rotulada = (
            100.0 * np.count_nonzero(self.labeled_idxs) / self.labels.size
        )

        # 8) acurácia por classe no teste (se fornecido)
        acuracia_por_classe = {}
        if test_preds is not None:
            y_true = (
                self.Y_test.numpy()
                if hasattr(self.Y_test, 'numpy')
                else np.array(self.Y_test)
            )
            y_pred = (
                test_preds.numpy()
                if hasattr(test_preds, 'numpy')
                else np.array(test_preds)
            )
            classes = np.unique(y_true)
            for cls in classes:
                mask = (y_true == cls)
                if mask.sum() > 0:
                    acc = (y_pred[mask] == cls).sum() / mask.sum()
                    acuracia_por_classe[int(cls)] = float(acc)

        return {
            "rótulos_atualizados":       num_corrigidos,
            "porcentagem_corrigida":      porcentagem_corrigida,
            "novas_classes_descobertas":  novas_classes_descobertas,
            "distrib_classes":            distrib_classes,
            "num_rotulos_adicionados":    num_rotulos_adicionados,
            "porcentagem_rotulada":       porcentagem_rotulada,
            "acuracia_por_classe":        acuracia_por_classe
        }


    def get_data_and_labels(self, indexes):
      """
      Retorna os dados e rótulos correspondentes aos índices fornecidos.

      :param indexes: Lista ou array de índices.
      :return: Dados e rótulos correspondentes aos índices.
      """
      X_subset = self.X_train[indexes]
      Y_subset = self.Y_train[indexes]
      return X_subset, Y_subset


def get_MNIST(handler):
    raw_train = datasets.MNIST('./data/MNIST', train=True, download=True)
    raw_test = datasets.MNIST('./data/MNIST', train=False, download=True)
    return Data(raw_train.data[:40000], raw_train.targets[:40000], raw_test.data[:40000], raw_test.targets[:40000], handler)

def get_FashionMNIST(handler):
    raw_train = datasets.FashionMNIST('./data/FashionMNIST', train=True, download=True)
    raw_test = datasets.FashionMNIST('./data/FashionMNIST', train=False, download=True)
    return Data(raw_train.data[:40000], raw_train.targets[:40000], raw_test.data[:40000], raw_test.targets[:40000], handler)

def get_SVHN(handler):
    data_train = datasets.SVHN('./data/SVHN', split='train', download=True)
    data_test = datasets.SVHN('./data/SVHN', split='test', download=True)
    return Data(data_train.data[:40000], torch.from_numpy(data_train.labels)[:40000], data_test.data[:40000], torch.from_numpy(data_test.labels)[:40000], handler)

def get_CIFAR10(handler):
    data_train = datasets.CIFAR10('./data/CIFAR10', train=True, download=True)
    data_test = datasets.CIFAR10('./data/CIFAR10', train=False, download=True)
    return Data(data_train.data[:40000], torch.LongTensor(data_train.targets)[:40000], data_test.data[:40000], torch.LongTensor(data_test.targets)[:40000], handler)

