from execute_strategy import executar_estrategia
from supervised_all_samples import executar_supervisionado_puro
from combined_active_learning import executar_estrategia_composta
from intersect_active_learning import executar_estrategia_intersecao
from union_active_learning import executar_estrategia_uniao

# Variáveis globais:
SEED = 1
ROUNDS = 2
INIT_LABELED = 200
QUERY = 200
DATSET = "MNIST"


if __name__ == "__main__":
    # necessário no Windows para multiprocess
    from multiprocessing import freeze_support
    freeze_support()

    # Aqui vamos executar o método supervisionado que já existe no método train em nets.py
    # O objetivo é obter a acurácia usando todas as amostras do dataset 
    df = executar_supervisionado_puro(DATSET, SEED, n_epoch=1)
    acuracia_teto = df['Acurácia Conjunto de Teste'].iloc[0]

    # Em seguida, vamos executar cada estratégia individualmente:
    estrategias = ["LeastConfidence", "MarginSampling", "EntropySampling","LeastConfidenceDropout", "MarginSamplingDropout", "EntropySamplingDropout", "KMeansSampling", "BALDDropout"]
    for estrategia in estrategias:
        args_individual = {
            'seed': SEED,
            'n_init_labeled': INIT_LABELED,
            'n_query': QUERY,
            'n_round': ROUNDS,
            'dataset_name': DATSET,
            'strategy_name': estrategia
        }
        #executar_estrategia(args_individual, acuracia_teto=acuracia_teto)

    args_composta1 = {
        'seed': 1,
        'n_init_labeled': INIT_LABELED,
        'n_query': QUERY,
        'n_round': ROUNDS,
        'dataset_name': DATSET,
        'strategy_prefilter': "MarginSampling",
        'strategy_refine': "LeastConfidence"
    }

    executar_estrategia_composta(args_composta1)

    args_composta2 = {
        'seed': 1,
        'n_init_labeled': INIT_LABELED,
        'n_query': QUERY,
        'n_round': ROUNDS,
        'dataset_name': DATSET,
        'strategy_prefilter': "LeastConfidence",
        'strategy_refine': "MarginSampling"
    }

    executar_estrategia_composta(args_composta2)

    args_intersecao = {
        'seed': 1,
        'n_init_labeled': INIT_LABELED,
        'n_query': QUERY,
        'n_round': ROUNDS, #Serão selecionados (n_query * n) amostras de cada estrategia para depois buscar a interseção
        'n': 10,
        'dataset_name': DATSET,
        'strategy_a': "LeastConfidence",
        'strategy_b': "MarginSampling"
    }

    executar_estrategia_intersecao(args_intersecao)

    args_uniao = {
        'seed': 1,
        'n_init_labeled': INIT_LABELED,
        'n_query': QUERY,
        'n_round': ROUNDS, #amostras selecionadas por cada estratégia = (n_query * n )
        'n': 0.5,
        'dataset_name': DATSET,
        'strategy_a': "LeastConfidence",
        'strategy_b': "MarginSampling"
    }

    executar_estrategia_uniao(args_uniao)
