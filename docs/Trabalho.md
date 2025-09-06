\# 5. TABELA DE RESULTADOS



| Modelo            | K  | Recall@K | NDCG@K | Observações                                      |

|-------------------|----|----------|--------|--------------------------------------------------|

| Popularidade      | 10 | 0.0675   | 0.0365 | baseline simples                                 |

| ALS Implícito     | 10 | 0.0975   | 0.0516 | factors=32, iters=5, sample\_users=2000           |

| Item-Item Cosine  | 10 | –        | –      | pulado (bug de dtype do implicit no Windows)     |



