import numpy as np
import pandas as pd
import torch
import time
from utils import get_dataset, get_net, get_strategy, accuracy_per_class

def executar_estrategia(args, acuracia_teto=None):

    print("\nIniciando com estratégia:", args['strategy_name'])

    # ------------------------------------------------------------------------
    # 1) Setup de sementes e device
    # ------------------------------------------------------------------------
    np.random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    torch.backends.cudnn.enabled = False

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # ------------------------------------------------------------------------
    # 2) Carregar dataset, rede e estratégia
    # ------------------------------------------------------------------------
    dataset = get_dataset(args['dataset_name'])
    try:
        print(f"         Tamanho X_train: {dataset.X_train.shape}, Y_train: {dataset.Y_train.shape}")
        print(f"         Tamanho X_test:  {dataset.X_test.shape},  Y_test:  {dataset.Y_test.shape}")
    except:
        labeled_idxs, _ = dataset.get_labeled_data()
        print(f"         labeled_idxs (round 0 antes): {labeled_idxs.shape}")

    net = get_net(args['dataset_name'], device)

    strategy_cls = get_strategy(args['strategy_name'])
    strategy = strategy_cls(dataset, net)

    # ------------------------------------------------------------------------
    # 3) Inicializa rótulos iniciais
    # ------------------------------------------------------------------------
    classes_antes_round0 = set(lbl for lbl in dataset.labels if lbl is not None)
    dataset.initialize_labels(args['n_init_labeled'])
    labeled_mask = dataset.labeled_idxs
    num_rotulados = int(np.sum(labeled_mask))

    t_ini = time.time()
    resultados = []

    # ------------------------------------------------------------------------
    # Round 0 (treino inicial / sem seleções de query)
    # ------------------------------------------------------------------------
    print("\n=== Round 0: Treino inicial e avaliação antes de qualquer query ===")
    # Obtém DataLoader dos dados rotulados
    labeled_idxs, loader_labeled = dataset.get_labeled_data()
    net.train(loader_labeled)

    # Coleta índices e labels iniciais
    initial_labeled_idxs = labeled_idxs
    #print(f"[DEBUG] initial_labeled_idxs:", initial_labeled_idxs)

    initial_labels = dataset.get_labels(initial_labeled_idxs)
    #print(f"[DEBUG] initial_labels (primeiras 10):", initial_labels[:10], "...")

    stats_round_0 = dataset.update_labeled_data(initial_labeled_idxs, initial_labels)

    # Predição no conjunto de teste
    loader_test = dataset.get_test_data()  # retorna apenas o DataLoader
    preds = net.predict(loader_test)
    accuracy = dataset.cal_test_acc(preds)

    # Avaliação nos dados rotulados (Round 0)
    labeled_data, labeled_labels = dataset.get_data_and_labels(initial_labeled_idxs)
    labeled_loader = dataset.handler(labeled_data, labeled_labels)
    labeled_preds = net.predict(labeled_loader)
    labeled_accuracy = (labeled_labels == labeled_preds).float().mean().item()

    acc_por_classe = accuracy_per_class(dataset.Y_test, preds)
    stats_round_0['novas_classes_descobertas'] = set(initial_labels) - classes_antes_round0

    resultados.append({
        "Round": 0,
        "Acurácia Conjunto de Teste": accuracy,
        "Acurácia Dados Rotulados": labeled_accuracy,
        "Novas Classes Descobertas": stats_round_0['novas_classes_descobertas'],
        "Distribuição Classes": stats_round_0['distrib_classes'],
        "Rótulos Atualizados": stats_round_0['rótulos_atualizados'],
        "Rótulos Adicionados": stats_round_0['num_rotulos_adicionados'],
        "Porcentagem Rotulada": f"{stats_round_0['porcentagem_rotulada']:.2f}%",
        "Porcentagem Corrigida": f"{100.00:.2f}%",
        "Acurácia por classe": acc_por_classe
    })

    teto_atingido = False
    contador_extras = 0

    # ------------------------------------------------------------------------
    # Loop de rounds de seleção (Round 1 até n_round)
    # ------------------------------------------------------------------------
    for rd in range(1, args['n_round'] + 1):
        print(f"\nIniciando Round {rd} --------------------------------")
        if teto_atingido and contador_extras >= 3:
            print("Parando após atingir acurácia teto + 3 rounds.")
            break

        unlabeled_idxs = np.where(~dataset.labeled_idxs)[0]

        t_selecao_ini = time.time()
        query_idxs = strategy.query(args['n_query'])
        t_selecao_fim = time.time()

        try:
            strategy.update(query_idxs)
        except AttributeError:
            print("[DEBUG] strategy.update() não definido para esta estratégia.")

        labeled_idxs, loader_labeled = dataset.get_labeled_data()
        net.train(loader_labeled)

        loader_test = dataset.get_test_data()
        preds = net.predict(loader_test)
        accuracy = dataset.cal_test_acc(preds)

        query_data, query_labels = dataset.get_data_and_labels(query_idxs)
        query_loader = dataset.handler(query_data, query_labels)
        query_preds = net.predict(query_loader)
        query_accuracy = (query_labels == query_preds).float().mean().item()

        labels_selecionados = dataset.get_labels(query_idxs)

        stats = dataset.update_labeled_data(query_idxs, labels_selecionados)

        if not teto_atingido and acuracia_teto and accuracy >= acuracia_teto:
            print(f"Atingiu acurácia do supervisionado: {accuracy:.4f} >= {acuracia_teto:.4f}")
            teto_atingido = True
        elif teto_atingido:
            contador_extras += 1

        pct_corr = (1.0 - query_accuracy) * 100.0
        acc_por_classe = accuracy_per_class(dataset.Y_test, preds)

        resultados.append({
            "Round": rd,
            "Acurácia Conjunto de Teste": accuracy,
            "Acurácia Dados Rotulados": query_accuracy,
            "Novas Classes Descobertas": stats['novas_classes_descobertas'],
            "Distribuição Classes": stats['distrib_classes'],
            "Rótulos Atualizados": stats['rótulos_atualizados'],
            "Rótulos Adicionados": stats['num_rotulos_adicionados'],
            "Porcentagem Rotulada": f"{stats['porcentagem_rotulada']:.2f}%",
            "Tempo Seleção": f"{(t_selecao_fim - t_selecao_ini):.2f}s",
            "Porcentagem Corrigida": f"{pct_corr:.2f}%",
            "Acurácia por classe": acc_por_classe
        })

    # ------------------------------------------------------------------------
    # Finalização e gravação de resultados
    # ------------------------------------------------------------------------
    t_fim = time.time()
    tempo_total = t_fim - t_ini
    if resultados:
        resultados[-1]['Tempo Total Execução (s)'] = tempo_total
    print(f"\nTempo total de execução: {tempo_total:.2f} segundos")

    df_resultados = pd.DataFrame(resultados)
    nome_arquivo = f"resultados_{args['strategy_name'].lower()}_{args['dataset_name'].lower()}.xlsx"
    df_resultados.to_excel(nome_arquivo, index=False)
    print(f"Finalizado. Acurácia final: {accuracy:.4f}")
    print(f"Resultados salvos em: {nome_arquivo}")

    return df_resultados
