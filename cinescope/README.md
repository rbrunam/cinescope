# CineScope — Recomendador com MovieLens

CineScope é um projeto acadêmico-prático para estudar **sistemas de recomendação** usando o dataset **MovieLens (ml-latest)**. O foco é comparar **baselines** (Popularidade e Item-Item) com um modelo **colaborativo implícito (ALS)**, medir **HR@K** e **NDCG@K** e publicar resultados de forma reprodutível.

## Dataset
Baixe o `ml-latest.zip` (ou `ml-latest-small.zip`) do site do MovieLens e **extraia** os CSVs em `data/`. O projeto usa principalmente `ratings.csv` e `movies.csv`.

> ⚠️ Os arquivos da pasta `data/` são ignorados pelo Git (ver `.gitignore`).

## Como rodar
```bash
# 1) ambiente
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt

# 2) coloque ratings.csv em data/
# 3) execute os baselines + ALS e métricas
python src/run_baselines.py --k 10 --factors 64 --iters 20
```

## Estrutura
```
cinescope/
  README.md
  requirements.txt
  .gitignore
  data/              # coloque aqui ratings.csv (MovieLens)
  artifacts/         # modelos/resultados (não versionados)
  notebooks/
  src/
    run_baselines.py
    data/loaders.py
    eval/metrics.py
```

## Métricas
- **Recall@K (HR@K)** e **NDCG@K** com *split temporal por usuário* (o último item avaliado de cada usuário vai para teste).

## Roadmap
- Híbrido leve (ALS + tags/gêneros).
- Re-ranking temporal por popularidade recente.
- Avaliação por cobertura e novidade.

## Licença
MIT — use à vontade para fins acadêmicos e de aprendizagem.
