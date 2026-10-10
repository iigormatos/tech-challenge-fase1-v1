# Tech Challenge — Fase 1 | IA para Diagnóstico de Câncer de Mama

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iigormatos/tech-challenge-fase1-v1/blob/joao-augusto/notebooks/01_principal.ipynb)

Sistema de apoio ao diagnóstico de câncer de mama (**maligno × benigno**) com Machine Learning sobre
dados tabulares (*Breast Cancer Wisconsin*). Projeto da Pós-Tech FIAP — IA para Devs.

## Problema

Hospitais precisam triar exames de câncer de mama com rapidez e segurança. Este projeto treina e avalia
modelos de classificação que, a partir de 30 medidas numéricas de núcleos celulares (raio, textura,
perímetro, área, concavidade etc.), estimam se um tumor é **maligno** ou **benigno**.

A métrica prioritária é o **recall da classe maligna (sensibilidade)**: deixar de identificar um tumor
maligno (**falso negativo**) tem custo clínico muito maior do que um falso positivo. Por isso o modelo
é calibrado para minimizar falsos negativos mantendo a especificidade sob controle, e a **acurácia não
é usada como métrica principal** (a base é desbalanceada, ~63% benigno / ~37% maligno). O sistema é uma
**ferramenta de suporte** — o diagnóstico e o laudo final são sempre do médico.

## Estrutura do projeto

```text
src/
├── config.py          # semente (RANDOM_STATE) e caminhos (fonte única, sem hardcode)
└── data.py            # carga, tradução de colunas (PT) e split estratificado 70/15/15
notebooks/
└── 01_principal.ipynb # notebook auto-contido: EDA → pré-proc → modelos → avaliação → interpretação
tests/
└── test_data.py       # testes de src/data.py (pytest)
reports/
├── figures/           # figuras geradas (PNG)
├── relatorio_tecnico.md
└── roteiro_video.md
models/                # modelo final persistido (gerado ao rodar o notebook)
Dockerfile · requirements.txt
```

> **Organização do código (decisão do grupo).** Para manter o código didático, próximo das aulas, a
> maior parte da lógica vive em células do notebook (cada função definida uma única vez e reutilizada).
> Apenas `src/config.py` e `src/data.py` ficam em módulos — são o ponto único de semente/caminhos e a
> fronteira de carga/divisão dos dados.

## Como executar

Pré-requisito: o dataset já está em `data/cancer-mama-diagnostico/data.csv`.

### 1. Local (VSCode / Jupyter)

```bash
python -m venv .venv
source .venv/Scripts/activate      # Windows (Git Bash);  Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/01_principal.ipynb   # ou "Run All"
# Alternativa não-interativa:
jupyter nbconvert --to notebook --execute notebooks/01_principal.ipynb --output 01_principal.ipynb
```

### 2. Google Colab

Clique no badge acima. A primeira célula clona o repositório, instala as dependências de
`requirements.txt` e ajusta os caminhos automaticamente. Depois use **Ambiente de execução → Executar
tudo**.

### 3. Docker

```bash
docker build -t cancer-mama .
docker run --rm -v "$PWD/reports:/app/reports" -v "$PWD/models:/app/models" cancer-mama
```

O container executa o notebook do início ao fim via `nbconvert`, gerando as figuras em `reports/figures/`
e o modelo em `models/`.

## Testes

```bash
pytest tests/test_data.py
```

## Resultados

Ao rodar o notebook são produzidos: a EDA por classe (com discussão), a tabela comparativa dos quatro
modelos (validação × teste) com recall, precision, F1, especificidade, ROC-AUC, PR-AUC e falsos
negativos, as curvas ROC/PR/limiar, a interpretabilidade (SHAP global e individual, coeficientes da
Regressão Logística, *permutation importance*) e o modelo final persistido em `models/`. Veja o
[relatório técnico](reports/relatorio_tecnico.md).

## Modelos utilizados

Regressão Logística, Random Forest, SVM (RBF) e KNN (o edital exige duas ou mais técnicas). Os três
primeiros usam `class_weight="balanced"`; o KNN não suporta ponderação e depende apenas do limiar de
decisão escolhido na validação.

## Integrantes

Bianca Maciel (RM376200) · Eduardo O. Charrone (RM377641) · Guilherme Mendes Alburquerque (RM378723) ·
Igor Matos de Andrade (RM377344) · João Augusto Gonçalves de Aragão (RM377385)
