"""Configuração central do projeto: ponto único de semente e caminhos.
"""

import os
from pathlib import Path

#: Semente aleatória usada em todo o projeto.
RANDOM_STATE: int = 42

#: Raiz do repositório, resolvida a partir deste arquivo (independe do CWD).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

#: Caminho do CSV de câncer de mama (env ``CANCER_MAMA_CSV`` sobrescreve o padrão do repo).
CANCER_MAMA_CSV: Path = Path(
    os.getenv("CANCER_MAMA_CSV", PROJECT_ROOT / "data" / "cancer-mama-diagnostico" / "data.csv")
)
