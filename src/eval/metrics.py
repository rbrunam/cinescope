import numpy as np

def recall_at_k(pred_items, true_item, k=10):
    # pred_items: lista/array de itens recomendados (top-K)
    # true_item: id do item verdadeiro
    return 1.0 if true_item in list(pred_items[:k]) else 0.0

def ndcg_at_k(pred_items, true_item, k=10):
    # DCG: 1 / log2(1 + rank)
    try:
        rank = list(pred_items[:k]).index(true_item) + 1
        return 1.0 / np.log2(rank + 1.0)
    except ValueError:
        return 0.0
