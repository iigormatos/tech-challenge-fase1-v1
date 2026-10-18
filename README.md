# Tech Challenge — Fase 1 | IA para Diagnóstico de Câncer de Mama

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iigormatos/tech-challenge-fase1-v1/blob/main/notebooks/01_principal.ipynb)

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
└── data.py            # carga tabular + split 70/15/15; catálogo de imagem + split por paciente (CNN)
notebooks/
├── 01_principal.ipynb # tarefa principal (tabular): EDA → pré-proc → modelos → avaliação → interpretação
└── 02_extra.ipynb     # EXTRA (visão computacional): CNN sobre mamografias (CBIS-DDSM)
reports/
└── figures/           # figuras geradas (PNG)

models/                # modelos persistidos (gerados ao rodar os notebooks)
Dockerfile · requirements.txt
```

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

## Resultados

Ao rodar o notebook são produzidos: a EDA por classe (com discussão), a tabela comparativa dos quatro
modelos (validação × teste) com recall, precision, F1, especificidade, ROC-AUC, PR-AUC e falsos
negativos, as curvas ROC/PR/limiar, a interpretabilidade (SHAP global e individual, coeficientes da
Regressão Logística, *permutation importance*) e o modelo final persistido em `models/`.

## Modelos utilizados

Regressão Logística, Random Forest, SVM (RBF) e KNN. Os três
primeiros usam `class_weight="balanced"`; o KNN não suporta ponderação e depende apenas do limiar de
decisão escolhido na validação.

## Item EXTRA — Detecção por visão computacional (CNN)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iigormatos/tech-challenge-fase1-v1/blob/main/notebooks/02_extra.ipynb)

`notebooks/02_extra.ipynb` adiciona o diagnóstico **maligno × benigno** a partir de **imagens de
mamografia** com **redes neurais convolucionais (CNN)**, usando o dataset **CBIS-DDSM** (recortes de
lesão). Segue os mesmos moldes do notebook principal: mesma narrativa seção a seção, prioridade no
**recall maligno**, limiar escolhido na validação e interpretação em markdown após cada resultado.

- **Stack**: TensorFlow/Keras. Modelo: **CNN**
- **Sem vazamento**: divisão **por paciente** (split oficial do dataset + validação por grupo); teste
  avaliado uma única vez.
- **Interpretabilidade**: **Grad-CAM** sobrepondo o mapa de calor à imagem (caso maligno e benigno).
- **Artefatos**: `models/modelo_cnn.keras`, `models/limiar_cnn.json`,
  `reports/tabela_comparativa_cnn.csv` e figuras `reports/figures/cnn_*.png`.

### Dataset de imagem (download automático)

O CBIS-DDSM (~5,8 GB) **não é versionado** (ver `.gitignore`). O notebook resolve a origem de forma
**híbrida**: usa a pasta local `data/cancer-mama-deteccao/` se existir; caso contrário, baixa
automaticamente via **kagglehub** (`awsaf49/cbis-ddsm-breast-cancer-image-dataset`) e **move o conteúdo
para `data/cancer-mama-deteccao/`**, deixando tudo na pasta do projeto (sem duplicar; nas próximas
execuções nada é baixado).

- **Local/Docker com os dados já presentes**: rápido e offline (coloque o dataset em
  `data/cancer-mama-deteccao/` com subpastas `csv/` e `jpeg/<SeriesUID>/`, ou aponte a env
  `CANCER_MAMA_IMG_DIR`).
- **Colab / máquina limpa**: o `kagglehub` baixa na primeira execução (alguns minutos) e cacheia;
  requer autenticação Kaggle (`kagglehub.login()` ou secrets `KAGGLE_USERNAME`/`KAGGLE_KEY`).
- **Docker**: o dataset **não** é embutido na imagem (ver `.dockerignore`). Monte a pasta local como
  volume para usar os dados sem baixar nada:
  ```bash
  docker run --rm -v "$PWD/data:/app/data" -v "$PWD/reports:/app/reports" \
    -v "$PWD/models:/app/models" cancer-mama \
    jupyter nbconvert --to notebook --execute notebooks/02_extra.ipynb --output 02_extra.ipynb
  ```
  Sem montar a pasta, o container baixaria do Kaggle (passe `-e KAGGLE_USERNAME=... -e KAGGLE_KEY=...`),
  re-baixando a cada execução.

Download manual (alternativa): [Kaggle — CBIS-DDSM Breast Cancer Image Dataset](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset/data).

#### Gerar a Kaggle API key

Só é necessária quando o dataset **não** está em disco (ex.: Colab novo). Passos:

1. Faça login em [kaggle.com](https://www.kaggle.com).
2. Clique no seu **perfil** (canto superior direito) → **Settings**.
3. Seção **API** → **Your API tokens** → **Generate new token**.
4. Dê um **nome** ao token e confirme. A chave aparece **na tela** — copie o `username` e a `key`
   (ela só é exibida uma vez).

Onde usar a chave copiada:

- **Colab**: rode `import kagglehub; kagglehub.login()` e cole `username`/`key`;
  **Secrets** do Colab as variáveis `KAGGLE_USERNAME` e `KAGGLE_KEY`.
- **Local/Docker** (se precisar): defina as variáveis de ambiente `KAGGLE_USERNAME` e `KAGGLE_KEY`,
  ou salve em `~/.kaggle/kaggle.json`.


### Executar o notebook extra

```bash
pip install -r requirements.txt   # inclui tensorflow-cpu e pillow
jupyter nbconvert --to notebook --execute notebooks/02_extra.ipynb --output 02_extra.ipynb
```

> **Nota**: o treino de CNN é pesado. Recomenda-se **Google Colab com GPU** para rodar em tempo
> razoável. Localmente (CPU) funciona, porém mais lento.

## Integrantes

Bianca Maciel (RM376200) · Eduardo O. Charrone (RM377641) · Guilherme Mendes Alburquerque (RM378723) ·
Igor Matos de Andrade (RM377344) · João Augusto Gonçalves de Aragão (RM377385)
