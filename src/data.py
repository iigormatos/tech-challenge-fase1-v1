# importação de bibliotecas
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .config import CANCER_MAMA_CSV, CANCER_MAMA_IMG_DIR, RANDOM_STATE

# Nome da coluna-alvo. Já traduzida.
COLUNA_ALVO: str = "diagnostico"

# Tradução das 30 variáveis características .
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

# Codificação do alvo: maligno é a classe positiva (1).
MAPA_ALVO: dict[str, int] = {"M": 1, "B": 0}

# Lê o CSV, descarta colunas sem valor, traduz nomes e codifica o alvo.
# Remove ``id`` (identificador) e colunas invalidas
def carregar_dados(caminho: Path = CANCER_MAMA_CSV) -> pd.DataFrame:
    df = pd.read_csv(caminho)

    # Descarta identificador e qualquer coluna "Unnamed" totalmente vazia.
    colunas_descartar = [c for c in df.columns if c == "id" or c.startswith("Unnamed")]
    df = df.drop(columns=colunas_descartar)

    df = df.rename(columns=TRADUCAO_COLUNAS)
    df[COLUNA_ALVO] = df[COLUNA_ALVO].map(MAPA_ALVO).astype(int)
    return df

def checar_qualidade(df: pd.DataFrame) -> dict:
    contagem_classes = df[COLUNA_ALVO].value_counts().sort_index()
    return {
        "nulos_por_coluna": df.isnull().sum(),
        "total_nulos": int(df.isnull().sum().sum()),
        "duplicatas": int(df.duplicated().sum()),
        "distribuicao_classes": contagem_classes,
        "distribuicao_classes_pct": (contagem_classes / len(df) * 100).round(2),
    }

# Divide ``df`` em treino / validação / teste 70/15/15
def dividir_treino_val_teste(
    df: pd.DataFrame,
    alvo: str = COLUNA_ALVO,
    random_state: int = RANDOM_STATE,
):
    X = df.drop(columns=[alvo])
    y = df[alvo]

    X_treino, X_temp, y_treino, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=random_state
    )
    X_val, X_teste, y_val, y_teste = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=random_state
    )
    return X_treino, X_val, X_teste, y_treino, y_val, y_teste


# ---------------------------------------------------------------------------
# Item EXTRA (visão computacional / CNN): catálogo de imagens do CBIS-DDSM.
MAPA_PATOLOGIA: dict[str, int] = {
    "MALIGNANT": 1,
    "BENIGN": 0,
    "BENIGN_WITHOUT_CALLBACK": 0,
}

# Arquivos de descrição de caso por tipo de achado (train/test oficiais do dataset).
_ARQUIVOS_CASO: dict[str, tuple[str, str]] = {
    "massa": ("mass_case_description_train_set.csv", "mass_case_description_test_set.csv"),
    "calcificacao": ("calc_case_description_train_set.csv", "calc_case_description_test_set.csv"),
}


def _mapa_uid_para_jpeg(csv_dir: Path) -> dict[str, str]:
    di = pd.read_csv(
        csv_dir / "dicom_info.csv",
        usecols=["image_path", "SeriesDescription", "SeriesInstanceUID"],
    )
    mapa: dict[str, str] = {}
    cropped = di[di["SeriesDescription"] == "cropped images"]
    for sub in (cropped, di):  # 1º passo: só cropped; 2º: fallback para qualquer série
        for uid, img in zip(sub["SeriesInstanceUID"], sub["image_path"]):
            if isinstance(img, str) and uid not in mapa:
                mapa[uid] = img.split("jpeg/")[-1]
    return mapa


def construir_catalogo_imagens(img_dir: Path = CANCER_MAMA_IMG_DIR) -> pd.DataFrame:
    img_dir = Path(img_dir)
    csv_dir = img_dir / "csv"
    jpeg_dir = img_dir / "jpeg"
    mapa = _mapa_uid_para_jpeg(csv_dir)

    registros: list[dict] = []
    descartados = 0
    for tipo, (arq_treino, arq_teste) in _ARQUIVOS_CASO.items():
        for origem, arq in (("train", arq_treino), ("test", arq_teste)):
            df = pd.read_csv(csv_dir / arq)
            col_dens = "breast_density" if "breast_density" in df.columns else "breast density"
            for _, r in df.iterrows():
                partes = str(r["cropped image file path"]).strip().split("/")
                if len(partes) < 2:
                    descartados += 1
                    continue
                rel = mapa.get(partes[-2])
                patologia = MAPA_PATOLOGIA.get(str(r["pathology"]).strip())
                if rel is None or patologia is None:
                    descartados += 1
                    continue
                caminho = jpeg_dir / Path(rel)
                if not caminho.exists():
                    descartados += 1
                    continue
                registros.append({
                    "caminho_imagem": str(caminho.resolve()),
                    "paciente": r["patient_id"],
                    "diagnostico": patologia,
                    "tipo_achado": tipo,
                    "vista": r.get("image view"),
                    "densidade": r.get(col_dens),
                    "origem": origem,
                })

    catalogo = pd.DataFrame(registros)
    catalogo.attrs["descartados"] = descartados
    return catalogo

# Divide o catálogo em treino / validação / teste SEM vazamento de paciente.
#  Garante, por asserção, que nenhum paciente aparece em mais de um conjunto.
def dividir_por_paciente(
    catalogo: pd.DataFrame,
    frac_val: float = 0.15,
    random_state: int = RANDOM_STATE,
):
    teste = catalogo[catalogo["origem"] == "test"].reset_index(drop=True)
    pacientes_teste = set(teste["paciente"])

    treino_val = catalogo[catalogo["origem"] == "train"].reset_index(drop=True)
    n_antes = len(treino_val)
    treino_val = treino_val[~treino_val["paciente"].isin(pacientes_teste)].reset_index(drop=True)
    removidos = n_antes - len(treino_val)

    gss = GroupShuffleSplit(n_splits=1, test_size=frac_val, random_state=random_state)
    idx_treino, idx_val = next(gss.split(treino_val, groups=treino_val["paciente"]))
    treino = treino_val.iloc[idx_treino].reset_index(drop=True)
    val = treino_val.iloc[idx_val].reset_index(drop=True)

    p_tr, p_val, p_te = set(treino["paciente"]), set(val["paciente"]), set(teste["paciente"])
    assert not (p_tr & p_val), "vazamento de paciente entre treino e validação"
    assert not (p_tr & p_te), "vazamento de paciente entre treino e teste"
    assert not (p_val & p_te), "vazamento de paciente entre validação e teste"
    treino.attrs["removidos_overlap_teste"] = removidos
    return treino, val, teste
