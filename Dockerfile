# Imagem reprodutível do Tech Challenge Fase 1 — diagnóstico de câncer de mama.
# Executa o notebook principal do zero via nbconvert, reproduzindo todas as
# figuras (reports/figures) e o modelo persistido (models/).
FROM python:3.12-slim

# Evita prompts e bytecode .pyc; saída de log sem buffer.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependências do sistema para compilar/rodar a stack científica (slim é enxuto).
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Instala as dependências Python com versões fixas (cache de camada).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o restante do projeto.
COPY . .

# Executa o notebook principal do início ao fim; falha o build se qualquer célula quebrar.
CMD ["jupyter", "nbconvert", "--to", "notebook", "--execute", \
     "--ExecutePreprocessor.timeout=600", \
     "--output", "01_principal.executado.ipynb", \
     "notebooks/01_principal.ipynb"]
