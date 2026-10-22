import os
from pathlib import Path

RANDOM_STATE: int = 42

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

#: Caminho do CSV de câncer de mama (env ``CANCER_MAMA_CSV`` sobrescreve o padrão do repo).
CANCER_MAMA_CSV: Path = Path(
    os.getenv("CANCER_MAMA_CSV", PROJECT_ROOT / "data" / "cancer-mama-diagnostico" / "data.csv")
)

#: Raiz do dataset de imagem (CBIS-DDSM) usado no notebook extra de CNN.
CANCER_MAMA_IMG_DIR: Path = Path(
    os.getenv("CANCER_MAMA_IMG_DIR", PROJECT_ROOT / "data" / "cancer-mama-deteccao")
)
