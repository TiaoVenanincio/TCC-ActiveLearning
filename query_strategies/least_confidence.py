import numpy as np
from .strategy import Strategy

class LeastConfidence(Strategy):
    def __init__(self, dataset, net):
        super(LeastConfidence, self).__init__(dataset, net)

    def query(self, n, dataset_subsample=None):
        """
        Seleciona os n exemplos com menor confiança (maior incerteza).
        Se `dataset_subsample` for fornecido, faz a query nesse subconjunto;
        caso contrário, usa o dataset original.
        """
        # obtém índices e dados não rotulados
        if dataset_subsample is None:
            unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        else:
            unlabeled_idxs, unlabeled_data = dataset_subsample.get_unlabeled_data()

        # usa o wrapper Net para obter probabilidades
        probs = self.net.predict_prob(unlabeled_data)
        if hasattr(probs, 'numpy'):
            probs = probs.numpy()

        # confiança = probabilidade máxima por amostra
        confidences = np.max(probs, axis=1)
        # seleciona os n menores (maior incerteza)
        local_idxs = np.argsort(confidences)[:n]

        # mapeia para índices globais
        return [int(unlabeled_idxs[i]) for i in local_idxs]
