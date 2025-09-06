import argparse
import os
import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import CosineRecommender
from src.data.loaders import (
    load_ratings, index_encode, temporal_user_split, build_item_user_matrix
)
from src.eval.metrics import recall_at_k, ndcg_at_k

# --------- Baseline: Popularidade ---------
class PopularityRecommender:
    def __init__(self, pop_order):
        self.pop_order = np.array(pop_order, dtype=np.int32)

    def recommend(self, u, user_items, N=10):
        seen = user_items[u].indices
        mask = ~np.isin(self.pop_order, seen, assume_unique=False)
        cand = self.pop_order[mask]
        top = cand[:N]
        scores = np.arange(len(top))[::-1]
        return top, scores

# --------- Util: pegar fatores (U,V) compatíveis com a matriz ----------
def get_uv_for_trainmat(als, train_mat):
    """
    Retorna (U, V) onde:
      U -> (n_users_mat, f)
      V -> (n_items_mat, f)
    Corrige automaticamente se o ALS tiver treinado com orientação trocada.
    """
    U_m, V_m = np.asarray(als.user_factors), np.asarray(als.item_factors)
    n_items_mat, n_users_mat = train_mat.shape

    if U_m.shape[0] == n_users_mat and V_m.shape[0] == n_items_mat:
        # orientação 'normal'
        return U_m, V_m
    elif U_m.shape[0] == n_items_mat and V_m.shape[0] == n_users_mat:
        # trocado: inverta
        return V_m, U_m
    else:
        raise ValueError(
            f"Incompatibilidade de dimensões: "
            f"user_factors={U_m.shape}, item_factors={V_m.shape}, "
            f"train_mat(items,users)={train_mat.shape}"
        )

# --------- Avaliações dedicadas ----------
def evaluate_popularity(pop_order, train_mat, test_df, k=10):
    user_items = train_mat.T.tocsr()  # usuários x itens
    recalls, ndcgs = [], []
    pop = PopularityRecommender(pop_order)
    for _, row in test_df.iterrows():
        u = int(row["u"]); true_i = int(row["i"])
        recs, _ = pop.recommend(u, user_items, N=k)
        recs = list(map(int, recs))
        recalls.append(recall_at_k(recs, true_i, k=k))
        ndcgs.append(ndcg_at_k(recs, true_i, k=k))
    return float(np.mean(recalls)), float(np.mean(ndcgs))

def evaluate_item_item(model, train_mat, test_df, k=10):
    user_items = train_mat.T.tocsr()
    recalls, ndcgs = [], []
    for _, row in test_df.iterrows():
        u = int(row["u"]); true_i = int(row["i"])
        recs, _ = model.recommend(u, user_items, N=k)
        recs = list(map(int, recs))
        recalls.append(recall_at_k(recs, true_i, k=k))
        ndcgs.append(ndcg_at_k(recs, true_i, k=k))
    return float(np.mean(recalls)), float(np.mean(ndcgs))

def evaluate_als_manual(als, train_mat, test_df, k=10):
    """
    Usa U,V coerentes com a matriz (corrige orientação).
    Calcula scores = U[u] @ V^T e remove itens vistos.
    """
    user_items = train_mat.T.tocsr()
    U, V = get_uv_for_trainmat(als, train_mat)   # U: (n_users,f), V: (n_items,f)
    Vt = V.T                                     # (f, n_items)
    recalls, ndcgs = [], []
    for _, row in test_df.iterrows():
        u = int(row["u"]); true_i = int(row["i"])
        scores = U[u] @ Vt                       # (n_items,)
        seen = user_items[u].indices             # ids de itens
        if seen.size:
            scores[seen] = -np.inf
        N = min(k, scores.shape[0])
        idx = np.argpartition(-scores, N-1)[:N]
        idx = idx[np.argsort(-scores[idx])]
        recs = list(map(int, idx))
        recalls.append(recall_at_k(recs, true_i, k=k))
        ndcgs.append(ndcg_at_k(recs, true_i, k=k))
    return float(np.mean(recalls)), float(np.mean(ndcgs))

def main():
    # Evita overthreading do OpenBLAS no ALS
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings", default="data/ratings.csv")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--factors", type=int, default=64)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--reg", type=float, default=0.01)
    parser.add_argument("--sample_users", type=int, default=None)
    args = parser.parse_args()

    print("[CineScope] carregando ratings:", args.ratings)
    ratings = load_ratings(args.ratings)

    if args.sample_users:
        users = ratings["userId"].drop_duplicates().sample(args.sample_users, random_state=42)
        ratings = ratings[ratings["userId"].isin(users)]

    ratings_enc, uid_map, iid_map = index_encode(ratings)
    train_df, test_df = temporal_user_split(ratings_enc)

    n_users = len(uid_map)
    n_items = len(iid_map)
    print(f"[CineScope] n_users={n_users} | n_items={n_items} | train={len(train_df)} | test={len(test_df)}")

    # Matriz itens x usuários (CSR int32/float32 garantida em loaders.py)
    train_mat = build_item_user_matrix(train_df, n_users=n_users, n_items=n_items, weight_col="rating")

    # --------- Popularidade ---------
    print("[CineScope] avaliando Popularidade ...")
    pop_scores = (train_mat.sum(axis=1)).A1
    pop_order = np.argsort(-pop_scores)
    rec_pop, ndcg_pop = evaluate_popularity(pop_order, train_mat, test_df, k=args.k)
    print(f"[CineScope] Popularidade | Recall@{args.k}: {rec_pop:.4f} | NDCG@{args.k}: {ndcg_pop:.4f}")

    # --------- Item-Item (com fallback no Windows) ---------
    item_item_metrics = None
    try:
        print("[CineScope] treinando Item-Item (Cosine) ...")
        item_item = CosineRecommender(K=100)
        item_item.fit(train_mat)
        rec_item, ndcg_item = evaluate_item_item(item_item, train_mat, test_df, k=args.k)
        print(f"[CineScope] Item-Item    | Recall@{args.k}: {rec_item:.4f} | NDCG@{args.k}: {ndcg_item:.4f}")
        item_item_metrics = {"recall": rec_item, "ndcg": ndcg_item}
    except ValueError as e:
        if "Buffer dtype mismatch" in str(e):
            print("[CineScope] Aviso: problema de dtype no Windows p/ Item-Item. Pulando.")
        else:
            raise

    # --------- ALS (com orientação corrigida) ---------
    print("[CineScope] treinando ALS ...")
    als = AlternatingLeastSquares(factors=args.factors, regularization=args.reg, iterations=args.iters)
    als.fit(train_mat)
    rec_als, ndcg_als = evaluate_als_manual(als, train_mat, test_df, k=args.k)
    print(f"[CineScope] ALS         | Recall@{args.k}: {rec_als:.4f} | NDCG@{args.k}: {ndcg_als:.4f}")

    # --------- Salvar métricas ---------
    os.makedirs("artifacts", exist_ok=True)
    metrics = {
        "k": args.k,
        "popularity": {"recall": rec_pop, "ndcg": ndcg_pop},
        "als": {"recall": rec_als, "ndcg": ndcg_als},
        "n_users": n_users,
        "n_items": n_items,
        "train_size": len(train_df),
        "test_size": len(test_df)
    }
    if item_item_metrics is not None:
        metrics["item_item"] = item_item_metrics

    import json
    with open("artifacts/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print("[CineScope] métricas salvas em artifacts/metrics.json")

if __name__ == "__main__":
    main()
