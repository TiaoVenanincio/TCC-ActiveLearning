import numpy as np
import pandas as pd
import torch
import time
from utils import get_dataset, get_net, get_strategy, accuracy_per_class

def executar_estrategia_uniao(args):

    print(f"\nExecutando união entre estratégias: {args['strategy_a']} ∪ {args['strategy_b']}")

    # ------------------------------------------------------------------------
    # 1) Setup de sementes e device
    # ------------------------------------------------------------------------
    np.random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    torch.backends.cudnn.enabled = False

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # ------------------------------------------------------------------------
    # 2) Carregar dataset e dois wrappers Net independentes
    # ------------------------------------------------------------------------
    dataset = get_dataset(args['dataset_name'])
    net_a = get_net(args['dataset_name'], device)
    net_b = get_net(args['dataset_name'], device)

    # ------------------------------------------------------------------------
    # 3) Instanciar duas estratégias distintas
    # ------------------------------------------------------------------------
    strategy_a = get_strategy(args['strategy_a'])(dataset, net_a)
    strategy_b = get_strategy(args['strategy_b'])(dataset, net_b)

    # ------------------------------------------------------------------------
    # 4) Inicializa rótulos iniciais
    # ------------------------------------------------------------------------
    classes_antes_round0 = set(lbl for lbl in dataset.labels if lbl is not None)
    dataset.initialize_labels(args['n_init_labeled'])
    t_ini = time.time()
    resultados = []

    # ------------------------------------------------------------------------
    # Round 0: treinar ambas as estratégias nos rótulos iniciais
    # ------------------------------------------------------------------------
    print("Round 0")
    strategy_a.train()
    strategy_b.train()

    # Predição no conjunto de teste usando strategy_a (por exemplo)
    loader_test = dataset.get_test_data()
    preds_test = strategy_a.predict(loader_test)
    accuracy_test = dataset.cal_test_acc(preds_test)

    # Estatísticas Round 0: confirma labels iniciais
    initial_labeled_idxs = np.where(dataset.labeled_idxs)[0]
    initial_labels = dataset.get_labels(initial_labeled_idxs)
    stats_round_0 = dataset.update_labeled_data(initial_labeled_idxs, initial_labels)

    # Avaliação nos dados rotulados
    labeled_data, labeled_labels = dataset.get_data_and_labels(initial_labeled_idxs)
    loader_labeled_eval = dataset.handler(labeled_data, labeled_labels)
    preds_labeled = strategy_a.predict(loader_labeled_eval)
    acc_labeled = (torch.tensor(labeled_labels) == preds_labeled).float().mean().item()

    # Acurácia por classe no teste
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
    # Rounds de união entre as duas estratégias
    # ------------------------------------------------------------------------
    for rd in range(1, args['n_round'] + 1):
        print(f"\nRound {rd} - Selecionando {args['n_query']} por união...")

        # Cada estratégia seleciona args['n_query'] * args['n'] candidatos
        idxs_a = strategy_a.query(int(args['n_query'] * args['n']))
        idxs_b = strategy_b.query(int(args['n_query'] * args['n']))

        # União dos índices
        uniao = list(set(idxs_a).union(set(idxs_b)))
        n_pool = len(dataset.Y_train)
        uniao = [idx for idx in uniao if 0 <= idx < n_pool]

        if len(uniao) < args['n_query']:
            complemento = list(set(range(n_pool)) - set(uniao) - set(np.where(dataset.labeled_idxs)[0]))
            np.random.shuffle(complemento)
            final_idxs = uniao + complemento[: args['n_query'] - len(uniao)]
        else:
            final_idxs = uniao[: args['n_query']]


        # --------------------------------------------------------------------
        # 1) Atualizar rótulos do dataset
        # --------------------------------------------------------------------
        t_selecao_ini = time.time()
        labels_selecionados = dataset.get_labels(final_idxs)
        stats = dataset.update_labeled_data(final_idxs, labels_selecionados)

        # Notificar estratégias sobre novos rótulos
        try:
            strategy_a.update(final_idxs)
        except AttributeError:
            pass
        try:
            strategy_b.update(final_idxs)
        except AttributeError:
            pass

        t_selecao_fim = time.time()

        # --------------------------------------------------------------------
        # 2) Treinar novamente ambas as estratégias com todos os rotulados
        # --------------------------------------------------------------------
        strategy_a.train()
        strategy_b.train()

        # --------------------------------------------------------------------
        # 3) Predição no teste usando strategy_a
        # --------------------------------------------------------------------
        loader_test = dataset.get_test_data()
        preds_test = strategy_a.predict(loader_test)
        accuracy_test = dataset.cal_test_acc(preds_test)

        # --------------------------------------------------------------------
        # 4) Avaliação nos exemplos selecionados (final_idxs)
        # --------------------------------------------------------------------
        query_data, query_labels = dataset.get_data_and_labels(final_idxs)
        query_loader = dataset.handler(query_data, query_labels)
        preds_query = strategy_a.predict(query_loader)
        acc_query = (torch.tensor(query_labels) == preds_query).float().mean().item()

        # --------------------------------------------------------------------
        # 5) Acurácia por classe no teste
        # --------------------------------------------------------------------
        acc_por_classe = accuracy_per_class(dataset.Y_test, preds_test)

        # --------------------------------------------------------------------
        # 6) Armazenar estatísticas
        # --------------------------------------------------------------------
        resultados.append({
            "Round": rd,
            "Acurácia Conjunto de Teste": accuracy_test,
            "Acurácia Dados Rotulados": acc_query,
            "Novas Classes Descobertas": stats['novas_classes_descobertas'],
            "Distribuição Classes": stats['distrib_classes'],
            "Rótulos Atualizados": stats['rótulos_atualizados'],
            "Rótulos Adicionados": stats['num_rotulos_adicionados'],
            "Porcentagem Rotulada": f"{stats['porcentagem_rotulada']:.2f}%",
            "Tempo Seleção": f"{(t_selecao_fim - t_selecao_ini):.2f}s",
            "Porcentagem Corrigida": f"{(1.0 - acc_query) * 100.0:.2f}%",
            "Acurácia por classe": acc_por_classe
        })

    # ------------------------------------------------------------------------
    # Finalização e gravação de resultados
    # ------------------------------------------------------------------------
    t_fim = time.time()
    tempo_total = t_fim - t_ini
    print(f"\nTempo total: {tempo_total:.2f}s")
    resultados[-1]['Tempo Total Execução (s)'] = tempo_total

    df_resultados = pd.DataFrame(resultados)
    print("\nEstatísticas do Treinamento:")
    print(df_resultados.to_string(index=False))

    nome_arquivo = f"resultados_uniao_{args['strategy_a'].lower()}_{args['strategy_b'].lower()}_{args['dataset_name'].lower()}.xlsx"
    df_resultados.to_excel(nome_arquivo, index=False)
    print(f"Resultados salvos em: {nome_arquivo}")

    return df_resultados
