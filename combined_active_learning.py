import numpy as np
import pandas as pd
import torch
import time
from utils import get_dataset, get_net, get_strategy, accuracy_per_class

def executar_estrategia_composta(args):

    print(f"\nExecutando estratégia composta: {args['strategy_prefilter']} ➝ {args['strategy_refine']}")

    # ------------------------------------------------------------------------
    # 1) Setup de sementes e device
    # ------------------------------------------------------------------------
    np.random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    torch.backends.cudnn.enabled = False

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # ------------------------------------------------------------------------
    # 2) Carregar dataset, rede e estratégias
    # ------------------------------------------------------------------------
    dataset = get_dataset(args['dataset_name'])
    net = get_net(args['dataset_name'], device)

    strategy_prefilter = get_strategy(args['strategy_prefilter'])(dataset, net)
    strategy_refine   = get_strategy(args['strategy_refine'])(dataset, net)

    # ------------------------------------------------------------------------
    # 3) Inicializa rótulos iniciais
    # ------------------------------------------------------------------------
    classes_antes_round0 = set(lbl for lbl in dataset.labels if lbl is not None)
    dataset.initialize_labels(args['n_init_labeled'])
    t_ini = time.time()
    resultados = []
    classes_conhecidas = set()
    historico_classes_por_iteracao = []

    # ------------------------------------------------------------------------
    # Round 0: Treino inicial apenas com prefilter
    # ------------------------------------------------------------------------
    print("Round 0")
    strategy_prefilter.train()  # treina nos rótulos iniciais

    # Predição no teste usando prefilter
    loader_test = dataset.get_test_data()
    preds_test = strategy_prefilter.predict(loader_test)
    accuracy_test = dataset.cal_test_acc(preds_test)

    # Estatísticas Round 0
    initial_labeled_idxs = np.where(dataset.labeled_idxs)[0]
    initial_labels = dataset.get_labels(initial_labeled_idxs)
    stats_round_0 = dataset.update_labeled_data(initial_labeled_idxs, initial_labels)

    # Avaliação nos rótulos iniciais
    labeled_data, labeled_labels = dataset.get_data_and_labels(initial_labeled_idxs)
    loader_labeled_eval = dataset.handler(labeled_data, labeled_labels)
    preds_labeled = strategy_prefilter.predict(loader_labeled_eval)
    acc_labeled = (torch.tensor(labeled_labels) == preds_labeled).float().mean().item()

    acc_por_classe = accuracy_per_class(dataset.Y_test, preds_test)
    stats_round_0['novas_classes_descobertas'] = set(initial_labels) - classes_antes_round0

    resultados.append({
        "Round": 0,
        "Acurácia Conjunto de Teste": accuracy_test,
        "Acurácia Dados Rotulados": acc_labeled,
        "Novas Classes Descobertas": stats_round_0['novas_classes_descobertas'],
        "Distribuição Classes": stats_round_0['distrib_classes'],
        "Rótulos Atualizados": stats_round_0['rótulos_atualizados'],
        "Rótulos Adicionados": stats_round_0['num_rotulos_adicionados'],
        "Porcentagem Rotulada": f"{stats_round_0['porcentagem_rotulada']:.2f}%",
        "Porcentagem Corrigida": f"{100.00:.2f}%",
        "Acurácia por classe": acc_por_classe
    })

    # ------------------------------------------------------------------------
    # Rounds de Pré-filtro + Refinamento
    # ------------------------------------------------------------------------
    for rd in range(1, args['n_round'] + 1):
        print(f"\nRound {rd} - Pré-filtro + refinamento")

        # Pré-filtro: seleciona mais exemplos
        n_prefilter = args['n_query'] * 3
        prefilter_idxs = strategy_prefilter.query(n_prefilter)

        # Subsample local
        dataset_subsample, local_indices = dataset.subsample(prefilter_idxs)
        local_to_global = {loc: glob for loc, glob in enumerate(prefilter_idxs)}

        # Refinamento no subconjunto
        final_query_local_idxs = strategy_refine.query(args['n_query'], dataset_subsample=dataset_subsample)
        final_query_local_idxs = [idx for idx in final_query_local_idxs if idx < len(prefilter_idxs)]
        if not final_query_local_idxs:
            print("Nenhum índice selecionado no refinamento.")
            continue

        final_query_global_idxs = [local_to_global[lidx] for lidx in final_query_local_idxs]

        # Atualizar rótulos no dataset original
        labels_selecionados = dataset.get_labels(final_query_global_idxs)
        classes_conhecidas.update(labels_selecionados)
        historico_classes_por_iteracao.append(list(classes_conhecidas))

        stats = dataset.update_labeled_data(final_query_global_idxs, labels_selecionados)

        # Notificar estratégias
        try:
            strategy_prefilter.update(final_query_global_idxs)
        except AttributeError:
            pass
        try:
            strategy_refine.update(final_query_global_idxs)
        except AttributeError:
            pass

        t_selecao_fim = time.time()

        # Treinar ambas as estratégias com todos os rotulados
        strategy_prefilter.train()
        strategy_refine.train()

        # Predição no teste usando refinamento
        loader_test = dataset.get_test_data()
        preds_test = strategy_refine.predict(loader_test)
        accuracy_test = dataset.cal_test_acc(preds_test)

        # Avaliação nos exemplos query
        query_data, query_labels = dataset.get_data_and_labels(final_query_global_idxs)
        query_loader = dataset.handler(query_data, query_labels)
        preds_query = strategy_refine.predict(query_loader)
        acc_query = (torch.tensor(query_labels) == preds_query).float().mean().item()

        acc_por_classe = accuracy_per_class(dataset.Y_test, preds_test)

        resultados.append({
            "Round": rd,
            "Acurácia Conjunto de Teste": accuracy_test,
            "Acurácia Dados Rotulados": acc_query,
            "Novas Classes Descobertas": stats['novas_classes_descobertas'],
            "Distribuição Classes": stats['distrib_classes'],
            "Rótulos Atualizados": stats['rótulos_atualizados'],
            "Rótulos Adicionados": stats['num_rotulos_adicionados'],
            "Porcentagem Rotulada": f"{stats['porcentagem_rotulada']:.2f}%",
            "Tempo Seleção": f"{(t_selecao_fim - t_ini):.2f}s",
            "Porcentagem Corrigida": f"{(1.0 - acc_query) * 100.0:.2f}%",
            "Acurácia por classe": acc_por_classe
        })

    # ------------------------------------------------------------------------
    # Finalização e gravação de resultados
    # ------------------------------------------------------------------------
    t_fim = time.time()
    tempo_total = t_fim - t_ini
    if resultados:
        resultados[-1]['Tempo Total Execução (s)'] = tempo_total
    print(f"\nTempo total: {tempo_total:.2f}s")

    df_resultados = pd.DataFrame(resultados)
    print("\nEstatísticas do Treinamento:")
    print(df_resultados.to_string(index=False))

    nome_arquivo = f"resultados_composto_{args['strategy_prefilter'].lower()}_{args['strategy_refine'].lower()}_{args['dataset_name'].lower()}.xlsx"
    df_resultados.to_excel(nome_arquivo, index=False)
    print(f"Resultados salvos em: {nome_arquivo}")

    return df_resultados
