from execute_strategy import executar_estrategia
from supervised_all_samples import executar_supervisionado_puro

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
        args = {
            'seed': SEED,
            'n_init_labeled': INIT_LABELED,
            'n_query': QUERY,
            'n_round': ROUNDS,
            'dataset_name': DATSET,
            'strategy_name': estrategia
        }
        executar_estrategia(args, acuracia_teto=acuracia_teto)