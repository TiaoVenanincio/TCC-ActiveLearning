import time
import pandas as pd
import numpy as np
import torch
from utils import get_dataset, get_net

def executar_supervisionado_puro(dataset_name="MNIST", seed=1, n_epoch=1):
    print(f"\nIniciando Execução Supervisionado Puro no dataset: {dataset_name}")

    # ------------------------------------------------------------------------
    # 1) Setup de sementes e device
    # ------------------------------------------------------------------------
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.enabled = False

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # ------------------------------------------------------------------------
    # 2) Preparar dataset e modelo
    # ------------------------------------------------------------------------
    dataset = get_dataset(dataset_name)
    net = get_net(dataset_name, device)

    # Marcar todos os dados como rotulados
    dataset.initialize_labels(dataset.n_pool)

    # ------------------------------------------------------------------------
    # 3) Treinar com todos os dados rotulados
    # ------------------------------------------------------------------------
    # get_labeled_data() retorna (índices, DataLoader)
    _, loader_labeled = dataset.get_labeled_data()

    t_ini = time.time()
    net.train(loader_labeled)  # treina em 100% dos dados rotulados
    t_fim = time.time()
    tempo_total = t_fim - t_ini

    # ------------------------------------------------------------------------
    # 4) Avaliar no conjunto de teste
    # ------------------------------------------------------------------------
    loader_test = dataset.get_test_data()  # DataLoader do conjunto de teste
    preds = net.predict(loader_test)
    acuracia_teste = dataset.cal_test_acc(preds)
    print(f"\nAcurácia no conjunto de teste com 100% dos dados rotulados: {acuracia_teste:.4f}")

    # ------------------------------------------------------------------------
    # 5) Salvar resultado em DataFrame
    # ------------------------------------------------------------------------
    df = pd.DataFrame([{
        'Round': 0,
        'Acurácia Conjunto de Teste': acuracia_teste,
        'Porcentagem Rotulada': "100%",
        'Tempo Total Execução (s)': tempo_total,
        'Modo': 'Supervisionado Puro'
    }])

    nome_arquivo = f'resultado_supervisionado_puro_{dataset_name.lower()}.xlsx'
    df.to_excel(nome_arquivo, index=False)
    print(f"Resultado salvo em: {nome_arquivo}")

    return df

