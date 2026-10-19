# Code review: pipeline de diagnóstico de câncer de mama (tabular e CNN)

Revisão de `notebooks/01_principal.ipynb`, `notebooks/02_extra.ipynb`, `src/data.py` e `src/config.py`. Nada foi executado. Os únicos números de métrica citados (ROC-AUC ≈ 0,61 e recall ≈ 0,26 no teste) vêm da **versão anterior** do `02_extra`, com CNN do zero. O notebook agora usa transfer learning e ainda **não foi reexecutado**, então não há métricas da versão atual.

## 1. Resumo executivo e arquitetura

**Problema.** Classificação binária maligno × benigno, como suporte à triagem médica. A métrica prioritária é o recall da classe maligna, porque um falso negativo é um câncer não sinalizado. Há duas frentes:
- `01_principal`: dados tabulares do Wisconsin (569 pacientes, 30 variáveis, 63/37 benigno/maligno).
- `02_extra`: imagens de mamografia (recortes de lesão do CBIS-DDSM) com uma CNN pré-treinada (EfficientNetB0, transfer learning).

```
NOTEBOOK 01 (tabular)
data.csv → carregar_dados (drop id/Unnamed, traduz colunas, alvo M=1/B=0)
 → checar_qualidade + EDA (sem remover outliers)
 → split estratificado 70/15/15 (ANTES de transformar)
 → Pipeline[Imputer(mediana) → StandardScaler* → Modelo]
 → GridSearchCV(scoring=recall, StratifiedKFold 5, só no treino) × 4 modelos
 → cross_validate (recall/ROC-AUC, média±dp)
 → limiar por modelo na VALIDAÇÃO (max recall s.t. especificidade ≥ 0,90)
 → avaliação no TESTE (6 métricas + IC de Wilson + FN)
 → SHAP / coeficientes LR / permutation importance
 → seleção do "melhor" por recall na validação → joblib + limiar.json

NOTEBOOK 02 (imagem)
CBIS-DDSM (local ou kagglehub) → catálogo (UID → JPEG, rótulo por pathology)
 → split por paciente (teste oficial; val = 15% dos pacientes do treino)
 → tf.data (decode, resize 224, batch 32, prefetch)
 → EfficientNetB0 ImageNet (aug → base → GAP → Dropout 0,3 → sigmoid)
 → fase 1: base congelada, Adam 1e-3 (até 10 épocas)
 → fase 2: fine-tuning das últimas 40 camadas (BN congelada), Adam 1e-5
 → class_weight, EarlyStopping(val_auc)
 → limiar na validação → avaliação no teste → Grad-CAM (top_activation) → .keras + limiar_cnn.json
```
\* O scaler só entra em LR, SVM e KNN.

## 2. Auditoria por bloco funcional

### Notebook 01

| Bloco | O que faz | Técnica e "porquê" | Entrada → saída |
|---|---|---|---|
| Setup (células 2–4) | Detecta Colab, clona o repo, instala `shap` e `missingno`, fixa o seed e define o helper de figuras | `RANDOM_STATE` único em `config.py` dá reprodutibilidade. `PROJECT_ROOT` via `__file__` não depende do diretório de execução. | → `RANDOM_STATE`, `salvar_figura`, diretórios |
| Carga e qualidade (6) | Remove `id` e `Unnamed: 32`, traduz as colunas uma vez e codifica o alvo | O mapa explícito `M→1` deixa claro que a classe positiva é maligna, o que importa para o recall. `id` é um identificador sem poder preditivo. | `data.csv` → `df` (569×31) |
| EDA, outliers e correlação (8–17) | Gráfico de ausentes, boxplots, histogramas, correlação com o alvo e heatmap | Os outliers não são removidos, porque nos malignos são sinal real (tumores grandes). A correlação serve para entender os dados, não para selecionar variáveis. | `df` → figuras |
| Split (19) | Divide estratificadamente em 70/15/15, em dois passos, antes de qualquer transformação | Estratificar preserva os 63/37 nos três conjuntos, o que importa em teste pequeno (~86 amostras). Treino ajusta, validação serve para o limiar e o teste é usado uma vez. | `df` → 6 objetos X/y |
| `criar_pipeline` (19) | `Imputer(mediana) → [Scaler] → Modelo` | O `Pipeline` garante que o scaler é reajustado dentro de cada fold da CV, o que evita vazamento. | — |
| Catálogo e tuning (22–23) | Treina 4 modelos com `GridSearchCV(recall)` e depois roda `cross_validate` | LR é o baseline interpretável. RF captura não linearidade e não precisa de escala. SVM-RBF lida bem com 30 variáveis em n pequeno. KNN é o baseline de distância e usa só o limiar para tratar o desbalanceamento. | `X_treino` → `ajustados`, `cv_resultados` |
| Limiar (26) | Varre 0,01 a 0,99 e escolhe o maior recall com especificidade ≥ 0,90, na validação | O limiar de 0,5 é arbitrário com classes desbalanceadas e custos assimétricos. O piso de especificidade evita a solução degenerada "tudo maligno". | `ajustados`, `X_val` → `limiares` |
| Avaliação (29–31) | Recall com IC de Wilson, precision, F1, especificidade, ROC-AUC, PR-AUC e FN, mais curvas e matrizes de confusão | O IC de Wilson é adequado a proporções com n pequeno. Os AUCs são independentes de limiar. | → `tabela`, CSV, figuras |
| Interpretabilidade (34–37) | SHAP (TreeExplainer) global e individual, coeficientes padronizados da LR e permutation importance | Três visões independentes que convergem reforçam a conclusão. A importância de Gini é enviesada em variáveis colineares. | → figuras |
| Persistência (40) | Escolhe o melhor por recall na validação e salva `joblib` + JSON | O pipeline inteiro é salvo, então o pré-processamento viaja junto com o modelo. | → `models/` |

### Notebook 02

| Bloco | O que faz | Porquê | Entrada → saída |
|---|---|---|---|
| Setup e dataset híbrido | Limpa o cache de `src` em `sys.modules` e põe a raiz na frente do `sys.path`. Usa a pasta local ou baixa via `kagglehub` e **copia** o conteúdo para `data/` | A cópia evita erros de "cross-device link" e sistema somente leitura no Colab, ao custo de duplicar cerca de 6 GB em disco. | → `img_dir` |
| Catálogo (`src/data.py`) | Une massa e calcificação, resolve o JPEG pelo `SeriesInstanceUID`, mapeia a patologia (`BENIGN_WITHOUT_CALLBACK` vira 0) e descarta linhas sem arquivo | Centraliza a fronteira onde o vazamento começaria. | → `catalogo` |
| Split por paciente | Teste = split oficial. Validação = 15% dos pacientes do treino via `GroupShuffleSplit`. Pacientes do treino que aparecem no teste são removidos | Imagens do mesmo paciente são correlacionadas. Sem agrupar, as métricas ficam infladas. As asserções de disjunção tornam isso verificável. | → 3 DataFrames |
| `tf.data` | Lê o JPEG, usa 3 canais, redimensiona para 224 e aplica `prefetch`. Embaralha só no treino | Normalização e augmentation dentro do modelo mantêm o pipeline idêntico nos três conjuntos. | → `ds_*` |
| Modelo | Augmentation (flip horizontal e vertical, rotação 10%, zoom 10%, contraste 10%) → EfficientNetB0 (ImageNet, `include_top=False`) → `GlobalAveragePooling2D` → Dropout 0,3 → sigmoid | Com poucas imagens médicas, reaproveitar características da ImageNet costuma superar uma CNN do zero. O GAP elimina a camada densa gigante do `Flatten`. A EfficientNet normaliza a entrada (0–255) internamente. | → modelo e base |
| Treino em duas fases | Fase 1: base congelada, Adam 1e-3, até 10 épocas. Fase 2: últimas 40 camadas descongeladas, BN congelada, Adam 1e-5, até 40 épocas. `class_weight` balanceado e `EarlyStopping(val_auc, max)` com `restore_best_weights` | Treinar primeiro a cabeça evita que gradientes aleatórios destruam os pesos pré-treinados. O LR baixo no fine-tuning preserva o que foi aprendido. `val_auc` não depende de limiar nem de ponderação de classe. | → `historicos` (fases concatenadas) |
| Limiar, avaliação e Grad-CAM | Reaproveitam a lógica do 01. O Grad-CAM usa a camada `top_activation` da base aninhada e reaplica as camadas posteriores (GAP, Dropout, Dense) | Grad-CAM é o análogo visual do SHAP e apoia a auditoria clínica. A busca por `Conv2D` no nível superior não funciona com a base aninhada. | → figuras, `.keras` |

## 3. Modelagem e validação

- **Algoritmos (01).** LR, Random Forest, SVM-RBF e KNN, que cobrem modelos linear, de árvores, de kernel e baseado em instância. Para uma base pequena, quase linearmente separável e de 30 variáveis, a LR já tende a ser competitiva e é a mais explicável. O ganho de modelos complexos deve ser mostrado com os ICs.
- **Tuning.** `GridSearchCV` com `StratifiedKFold(5, shuffle)` e `scoring="recall"`, só no treino.
  - LR: `C ∈ {0,01; 0,1; 1; 10}`.
  - RF: `n_estimators ∈ {200, 400}` e `max_depth ∈ {None, 5, 10}`.
  - SVM: `C` e `gamma`.
  - KNN: `k ∈ {3, ..., 11}`.
- **CNN (02).** Sem busca de hiperparâmetros. LRs (1e-3 e 1e-5), número de camadas descongeladas (40), dropout e tamanho de entrada são fixos. Isso é uma limitação. Só a EfficientNetB0 foi tentada, sem comparação com outras bases (ResNet, DenseNet).
- **Métricas.**
  - O recall maligno é a prioridade, com o número de FN ao lado.
  - Especificidade é o contrapeso do recall.
  - PR-AUC é mais informativa que ROC-AUC com desbalanceamento, porque ignora os verdadeiros negativos.
  - Acurácia é descartada: o baseline "tudo benigno" dá 63% no 01 e ~61% no 02.
  - O IC de Wilson comunica a incerteza do recall.

## 4. Gaps técnicos e code smells

### Vazamento de dados
**Não encontrei vazamento relevante antes do split**:
- No 01, o split vem antes de qualquer transformação, e imputer e scaler ficam no `Pipeline`.
- No 02, o split é por paciente, com asserção.
- O limiar é escolhido na validação e aplicado intacto ao teste.

Pontos de atenção:
1. **Correlação com o alvo na base completa (01, célula 16).** É usada só para análise, mas inclui o teste. Para ser rigoroso, deveria usar apenas o treino.
2. **Reuso da validação (01 e 02).** A mesma validação escolhe o limiar e depois o "melhor modelo" (`max recall`). Isso é seleção otimista, e o recall máximo tende a empatar entre modelos (1,0), o que torna o critério quase arbitrário. Melhor usar PR-AUC ou recall com desempate.
3. **Validação pequena (01).** Com ~85 amostras e ~32 malignos, o limiar escolhido varrendo 99 pontos tem alta variância. Uma alternativa é obter as probabilidades por CV no treino (`cross_val_predict`) e escolher o limiar sobre elas.
4. **Pacientes removidos do treino (02).** Isso é correto, mas o número de imagens removidas deve ser reportado.
5. **Scoring do grid (01).** O tuning otimiza `recall` no limiar 0,5, e um modelo que classifica tudo como maligno tem recall 1,0. O piso de especificidade só atua depois do tuning. Um scoring como `average_precision` ou `roc_auc` seria mais estável.

### Problemas metodológicos específicos
- **`cross_validate` após `GridSearchCV` (01, célula 23).** É CV no mesmo conjunto usado para escolher os hiperparâmetros, então o recall de CV é otimista. O correto seria CV aninhada.
- **Reprodutibilidade parcial (02).** `set_random_seed` não garante determinismo em GPU (reconhecido no texto).
- **CNN do zero (resolvido no código, sem métricas ainda).** A versão anterior tinha ROC-AUC ≈ 0,61 e recall ≈ 0,26, e a causa provável era a arquitetura (`Flatten` gerando camada densa enorme, sem transfer learning). O notebook agora usa EfficientNetB0 com fine-tuning em duas fases. **Falta reexecutar** para confirmar o ganho.
- **Texto do notebook desatualizado.** A seção 10 (conclusão) ainda descreve a CNN do zero, com Adam 1e-4, `BatchNormalization`, ROC-AUC ≈ 0,61 e recall ≈ 0,26. Deve ser reescrita com os resultados da nova execução. O mesmo vale para a seção "Para uso em produção faltaria", que cita fine-tuning como pendente.
- **Limiar e especificidade ≥ 0,90 no 02.** Com o modelo anterior (AUC 0,61), esse piso era praticamente inatingível sem recall muito baixo, o que explica o 0,26. Com um AUC maior, o piso passa a ser viável. Vale conferir se o fallback de 0,5 não foi acionado.
- **Métrica do Keras.** O `Recall` do Keras no treino usa limiar 0,5, não o limiar final.
- **`class_weight` e validação.** A validação não é ponderada, então a loss de validação não é comparável com a de treino. O `EarlyStopping` em `val_auc` contorna esse ponto.
- **Fine-tuning.** Descongelar as últimas 40 camadas com ~1.500 imagens de treino pode gerar overfit. Acompanhar a diferença entre as curvas de treino e validação.
- **Dependência de rede.** `weights="imagenet"` baixa os pesos na primeira execução, o que falha em ambientes offline (Docker sem internet).
- **Domínio.** ImageNet é de fotos RGB, e mamografias são em tons de cinza replicados em 3 canais. O ganho do transfer learning existe, mas é menor que em imagens naturais.
- **Rótulos.** `BENIGN_WITHOUT_CALLBACK` como benigno é razoável, mas deve ser justificado na apresentação.

### Code smells
- **Variáveis globais nos notebooks.** `RANDOM_STATE`, `TAM_IMG`, `peso_classe`, `modelos` e `aumento` são globais, e funções como `treinar` dependem de `peso_classe`.
- **Duplicação entre os notebooks.** `escolher_limiar`, `avaliar`, `ic_recall_wilson`, `linha` e `salvar_figura` estão copiadas nos dois. Deveriam ir para `src/` (por exemplo, `src/avaliacao.py`).
- **Fallback silencioso.** Em `escolher_limiar`, `melhor_lim` começa em 0,5. Se nenhum limiar atinge o piso de especificidade, o código devolve 0,5 sem avisar.
- **Assinaturas diferentes** de `escolher_limiar` entre o 01 (pipeline) e o 02 (probabilidades), o que atrapalha a reutilização.
- **`construir_catalogo_imagens` usa `iterrows`** e `Path.exists()` por linha. Funciona, mas é lento.
- **Bootstrap do ambiente dentro do notebook.** O `shutil.rmtree` do clone no Colab apaga o diretório sem confirmação, e o `pip install` não fixa versões.
- **Dicionário de um único modelo** (`"CNN"`), mantido por uniformidade, adiciona complexidade sem uso.
- **Texto incompleto (resolvido):** a interpretação da seção 5 do 02, que tinha a referência faltando, foi reescrita para descrever o transfer learning.
- **Typo:** "redimensia" no comentário do `tf.data`.
- **Grad-CAM** usa sempre o primeiro caso de cada classe. O ideal seria mostrar acertos e erros (FN). Também depende do nome `top_activation` e do prefixo `efficientnet` do modelo, então quebra se a base for trocada.
- **Dois notebooks divergentes.** `_02_extra.ipynb` (versão Colab, CNN do zero, branch `igor`) e `02_extra.ipynb` coexistem. Convém consolidar em um só para evitar que as correções se percam.

### Memória e processamento
- **GridSearch (01):** RF é a etapa mais cara, mas é insignificante em ~400 linhas. O SVM com `probability=True` faz uma CV interna de 5 folds extra.
- **Pipeline de imagem:** o `tf.data` lê o JPEG do disco a cada época, sem `.cache()`. Esse é o gargalo de I/O mais provável. Com ~6 GB de dataset, o cache em memória é inviável, mas um cache em disco ou pré-redimensionar uma vez (TFRecord) ajudaria.
- **Redimensionamento para 224×224** pode distorcer a razão de aspecto e destruir microcalcificações, o que pode explicar parte do desempenho baixo.
- **Treino:** não há `mixed_precision` nem `ReduceLROnPlateau`. O fine-tuning de camadas da EfficientNetB0 em 224×224 é mais pesado que a CNN anterior e pede GPU (Colab).

## 5. Roteiro de apresentação (5 slides)

**Slide 1: problema e restrição clínica**
- "O erro caro é o falso negativo. Por isso otimizamos recall maligno, com um piso de especificidade para não alertar tudo."
- "A IA é suporte: a decisão final é do médico."
- "Acurácia seria enganosa, já que 'tudo benigno' acerta 63%."

**Slide 2: arquitetura e disciplina anti-vazamento**
- Mostrar o diagrama do pipeline.
- "O split vem antes de qualquer transformação, e o `Pipeline` reajusta o scaler em cada fold."
- "Nas imagens, dividimos por paciente e verificamos com asserção que nenhum paciente está em dois conjuntos."
- "Removemos pacientes do treino que apareciam no teste oficial."

**Slide 3: modelagem tabular**
- "Quatro famílias de modelos, com tuning por CV estratificada só no treino."
- "O limiar é escolhido na validação (especificidade ≥ 0,90) e aplicado sem mudança ao teste."
- "Reportamos recall com IC de Wilson, porque o teste é pequeno e diferenças de um ou dois casos não são conclusivas."
- "Não removemos outliers: nos malignos eles são sinal clínico. A versão anterior descartava mais da metade dos malignos."

**Slide 4: explicabilidade e leitura crítica**
- "SHAP, coeficientes da LR e permutation importance convergem em tamanho e forma do pior núcleo (`pontos_concavos_pior`, `perimetro_pior`, `raio_pior`)."
- "Raio, perímetro e área são quase colineares (r ≈ 0,99), então a leitura é por grupo."
- "Desempenho próximo de 100% reflete a separabilidade conhecida do Wisconsin e um teste pequeno, não um modelo infalível."

**Slide 5: extra de visão computacional, limites e próximos passos**
- "A primeira CNN, treinada do zero, obteve ROC-AUC ≈ 0,61 e recall ≈ 0,26: o pipeline estava correto, mas o modelo era fraco. Migramos para transfer learning (EfficientNetB0 com fine-tuning em duas fases)." Apresentar os números da nova execução, que ainda precisam ser gerados.
- "A acurácia (~62%) da primeira versão era próxima do baseline (~61%), o que confirma por que não a usamos."
- Próximos passos: comparar outras bases (ResNet, DenseNet) e tamanhos de entrada maiores, CV aninhada e `average_precision` no tuning, calibração de probabilidades, validação externa e monitoramento de drift.
- Pergunta provável da banca: "por que não SMOTE?" Resposta: o `class_weight` preserva a distribuição real e evita amostras sintéticas, que vazariam se aplicadas antes do split.

**Prioridades de correção, se houver tempo:**
1. Mover `escolher_limiar` e `avaliar` para `src/`.
2. Trocar o scoring do grid para `average_precision`.
3. Reexecutar o `02_extra` com transfer learning e atualizar a seção 10 do notebook e este documento com as métricas reais (transfer learning já implementado).
