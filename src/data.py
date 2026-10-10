"""Carga, limpeza, tradução e divisão dos dados de câncer de mama.

Único ponto do projeto onde o CSV bruto é lido, as colunas são traduzidas para
``snake_case`` em português e os dados são divididos de forma estratificada em
treino / validação / teste. Mantido em ``src/`` por ser a fronteira onde um
vazamento começaria (constitution V).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import CANCER_MAMA_CSV, RANDOM_STATE

#: Nome da coluna-alvo já traduzida.
COLUNA_ALVO: str = "diagnostico"

#: Tradução das 30 features + alvo para ``snake_case`` em português (aplicada 1×).
TRADUCAO_COLUNAS: dict[str, str] = {
    "diagnosis": "diagnostico",
    # --- média (_mean → _medio) ---
    "radius_mean": "raio_medio",
    "texture_mean": "textura_media",
    "perimeter_mean": "perimetro_medio",
    "area_mean": "area_media",
    "smoothness_mean": "suavidade_media",
    "compactness_mean": "compacidade_media",
    "concavity_mean": "concavidade_media",
    "concave points_mean": "pontos_concavos_medio",
    "symmetry_mean": "simetria_media",
    "fractal_dimension_mean": "dimensao_fractal_media",
    # --- erro padrão (_se → _erro_padrao) ---
    "radius_se": "raio_erro_padrao",
    "texture_se": "textura_erro_padrao",
    "perimeter_se": "perimetro_erro_padrao",
    "area_se": "area_erro_padrao",
    "smoothness_se": "suavidade_erro_padrao",
    "compactness_se": "compacidade_erro_padrao",
    "concavity_se": "concavidade_erro_padrao",
    "concave points_se": "pontos_concavos_erro_padrao",
    "symmetry_se": "simetria_erro_padrao",
    "fractal_dimension_se": "dimensao_fractal_erro_padrao",
    # --- pior (_worst → _pior) ---
    "radius_worst": "raio_pior",
    "texture_worst": "textura_pior",
    "perimeter_worst": "perimetro_pior",
    "area_worst": "area_pior",
    "smoothness_worst": "suavidade_pior",
    "compactness_worst": "compacidade_pior",
    "concavity_worst": "concavidade_pior",
    "concave points_worst": "pontos_concavos_pior",
    "symmetry_worst": "simetria_pior",
    "fractal_dimension_worst": "dimensao_fractal_pior",
}

#: Codificação do alvo: maligno é a classe positiva (1).
MAPA_ALVO: dict[str, int] = {"M": 1, "B": 0}


def carregar_dados(caminho: Path = CANCER_MAMA_CSV) -> pd.DataFrame:
    """Lê o CSV, descarta colunas sem valor, traduz nomes e codifica o alvo.

    - Remove ``id`` (identificador) e colunas espúrias totalmente vazias
      (ex.: ``Unnamed: 32`` gerada pela vírgula final do cabeçalho).
    - Traduz as colunas uma única vez para ``snake_case`` em português.
    - Codifica ``diagnostico`` como ``1`` (maligno) / ``0`` (benigno).
    """
    df = pd.read_csv(caminho)

    # Descarta identificador e qualquer coluna "Unnamed" totalmente vazia.
    colunas_descartar = [c for c in df.columns if c == "id" or c.startswith("Unnamed")]
    df = df.drop(columns=colunas_descartar)

    df = df.rename(columns=TRADUCAO_COLUNAS)
    df[COLUNA_ALVO] = df[COLUNA_ALVO].map(MAPA_ALVO).astype(int)
    return df


def checar_qualidade(df: pd.DataFrame) -> dict:
    """Retorna diagnóstico de qualidade dos dados (não altera ``df``).

    Inclui contagem de nulos por coluna, total de nulos, linhas duplicadas e a
    distribuição absoluta e relativa das classes do alvo.
    """
    contagem_classes = df[COLUNA_ALVO].value_counts().sort_index()
    return {
        "nulos_por_coluna": df.isnull().sum(),
        "total_nulos": int(df.isnull().sum().sum()),
        "duplicatas": int(df.duplicated().sum()),
        "distribuicao_classes": contagem_classes,
        "distribuicao_classes_pct": (contagem_classes / len(df) * 100).round(2),
    }


def dividir_treino_val_teste(
    df: pd.DataFrame,
    alvo: str = COLUNA_ALVO,
    random_state: int = RANDOM_STATE,
):
    """Divide ``df`` em treino / validação / teste 70/15/15, estratificado.

    Feito em dois passos (70/30 e depois 30 → 15/15) sempre com ``stratify`` pelo
    alvo, preservando a proporção de classes (~63/37) nos três conjuntos. Nenhuma
    linha é removida.

    Retorna ``(X_treino, X_val, X_teste, y_treino, y_val, y_teste)``.
    """
    X = df.drop(columns=[alvo])
    y = df[alvo]

    X_treino, X_temp, y_treino, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=random_state
    )
    X_val, X_teste, y_val, y_teste = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=random_state
    )
    return X_treino, X_val, X_teste, y_treino, y_val, y_teste
