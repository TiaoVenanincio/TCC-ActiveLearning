from torchvision import transforms
import numpy as np
from handlers import MNIST_Handler, SVHN_Handler, CIFAR10_Handler
from data import get_MNIST, get_FashionMNIST, get_SVHN, get_CIFAR10
from nets import Net, MNIST_Net, SVHN_Net, CIFAR10_Net
from query_strategies import RandomSampling, LeastConfidence, MarginSampling, EntropySampling, \
                             LeastConfidenceDropout, MarginSamplingDropout, EntropySamplingDropout, \
                             KMeansSampling, KCenterGreedy, BALDDropout, \
                             AdversarialBIM, AdversarialDeepFool

params = {
    'MNIST': {
        'n_epoch': 10,
        'train_args':    {'batch_size': 64,   'num_workers': 1},
        'test_args':     {'batch_size': 1000, 'num_workers': 1},
        'optimizer_args':{'lr': 0.01, 'momentum': 0.5}
    },
    'FashionMNIST': {
        'n_epoch': 10,
        'train_args':    {'batch_size': 64,   'num_workers': 1},
        'test_args':     {'batch_size': 1000, 'num_workers': 1},
        'optimizer_args':{'lr': 0.01, 'momentum': 0.5}
    },
    'SVHN': {
        'n_epoch': 20,
        'train_args':    {'batch_size': 64,   'num_workers': 1},
        'test_args':     {'batch_size': 1000, 'num_workers': 1},
        'optimizer_args':{'lr': 0.01, 'momentum': 0.5}
    },
    'CIFAR10': {
        'n_epoch': 20,
        'train_args':    {'batch_size': 64,   'num_workers': 1},
        'test_args':     {'batch_size': 1000, 'num_workers': 1},
        'optimizer_args':{'lr': 0.05, 'momentum': 0.3}
    }
}


def get_handler(name):
    if name == 'MNIST':
        return MNIST_Handler
    elif name == 'FashionMNIST':
        return MNIST_Handler
    elif name == 'SVHN':
        return SVHN_Handler
    elif name == 'CIFAR10':
        return CIFAR10_Handler
    else:
        raise NotImplementedError(f"Handler para dataset '{name}' não foi implementado.")


def get_dataset(name):
    if name == 'MNIST':
        return get_MNIST(get_handler(name))
    elif name == 'FashionMNIST':
        return get_FashionMNIST(get_handler(name))
    elif name == 'SVHN':
        return get_SVHN(get_handler(name))
    elif name == 'CIFAR10':
        return get_CIFAR10(get_handler(name))
    else:
        raise NotImplementedError(f"Dataset '{name}' não foi reconhecido em get_dataset.")

def get_net(name, device):
    """
    Retorna uma instância de Net(...) cujo argumento 'net'
    é a classe apropriada (MNIST_Net, SVHN_Net, CIFAR10_Net ou Parasitos_Net),
    usando os params definidos acima.
    """
    if name == 'MNIST':
        return Net(MNIST_Net, params[name], device)
    elif name == 'FashionMNIST':
        return Net(MNIST_Net, params[name], device)
    elif name == 'SVHN':
        return Net(SVHN_Net, params[name], device)
    elif name == 'CIFAR10':
        return Net(CIFAR10_Net, params[name], device)
    else:
        raise NotImplementedError(f"Net para dataset '{name}' não foi implementada.")

def get_params(name):
    try:
        return params[name]
    except KeyError:
        raise NotImplementedError(f"Parâmetros para '{name}' não encontrados em params.")

def get_strategy(name):
    if name == "RandomSampling":
        return RandomSampling
    elif name == "LeastConfidence":
        return LeastConfidence
    elif name == "MarginSampling":
        return MarginSampling
    elif name == "EntropySampling":
        return EntropySampling
    elif name == "LeastConfidenceDropout":
        return LeastConfidenceDropout
    elif name == "MarginSamplingDropout":
        return MarginSamplingDropout
    elif name == "EntropySamplingDropout":
        return EntropySamplingDropout
    elif name == "KMeansSampling":
        return KMeansSampling
    elif name == "KCenterGreedy":
        return KCenterGreedy
    elif name == "BALDDropout":
        return BALDDropout
    elif name == "AdversarialBIM":
        return AdversarialBIM
    elif name == "AdversarialDeepFool":
        return AdversarialDeepFool
    else:
        raise NotImplementedError(f"Estratégia '{name}' não foi implementada em get_strategy.")


# Função utilitária para calcular acurácia por classe
def accuracy_per_class(y_true, y_pred):
    # Se for tensor do PyTorch, converte para numpy
    if hasattr(y_true, "numpy"):
        y_true = y_true.numpy()
    if hasattr(y_pred, "numpy"):
        y_pred = y_pred.numpy()

    classes = np.unique(y_true)
    acc = {}
    for cls in classes:
        mask = (y_true == cls)
        if mask.sum() > 0:
            acc[int(cls)] = float((y_pred[mask] == cls).sum() / mask.sum())
    return acc
