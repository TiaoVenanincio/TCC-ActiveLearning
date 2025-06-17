import numpy as np
from .strategy import Strategy

class MarginSampling(Strategy):
    def __init__(self, dataset, net):
        super(MarginSampling, self).__init__(dataset, net)

    def query(self, n, dataset_subsample=None):
        """
        Seleciona os n exemplos com maior margem de incerteza (menor diferença entre
        as duas maiores probabilidades). Usa `dataset_subsample` se fornecido.
        """
        if dataset_subsample is None:
            unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        else:
            unlabeled_idxs, unlabeled_data = dataset_subsample.get_unlabeled_data()

        # obtém probabilidades [N, C]
        probs = self.net.predict_prob(unlabeled_data)
        if hasattr(probs, 'numpy'):
            probs = probs.numpy()

        # para cada amostra, obtém os dois maiores valores
        # ordena colunas em ordem decrescente e pega top2
        sorted_idx = np.argsort(-probs, axis=1)
        top1 = probs[np.arange(probs.shape[0]), sorted_idx[:, 0]]
        top2 = probs[np.arange(probs.shape[0]), sorted_idx[:, 1]]
        margins = top1 - top2

        # menor margem = maior incerteza
        local_idxs = np.argsort(margins)[:n]

        return [int(unlabeled_idxs[i]) for i in local_idxs]

