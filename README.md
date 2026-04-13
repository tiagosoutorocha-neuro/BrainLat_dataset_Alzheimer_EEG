# 🧠 Pipeline ML para Detecção de Alzheimer via EEG

Classificador binário **Alzheimer (AD) vs. Controle Saudável (HC)** baseado em **features espectrais de EEG**, com validação **Leave-One-Subject-Out (LOSO)** e explicabilidade via **SHAP**.

Este repositório foi organizado a partir do notebook `Pipeline_ML_Alzheimer_Reorganizado (1).ipynb`, que implementa um fluxo completo:

1. download/preparo dos dados BrainLat;
2. pré-processamento do EEG;
3. extração de features espectrais;
4. benchmark de modelos de machine learning;
5. análise de interpretabilidade e relação com cognição.

---

## Visão geral

O objetivo do projeto é construir um classificador robusto para distinguir sujeitos com Doença de Alzheimer de controles saudáveis a partir de EEG em repouso. O pipeline foi desenhado para reduzir vazamento de dados e para avaliar desempenho em nível de **sujeito**, e não apenas de época.

### Principais escolhas metodológicas
- **Validação LOSO por sujeito**: cada fold deixa um sujeito inteiro para teste.
- **Normalização feita somente no treino**: evita data leakage.
- **Predição agregada por sujeito**: a decisão final vem da média das probabilidades das épocas.
- **Features espectrais consolidadas na literatura**: potência relativa por banda, razões espectrais e entropia espectral.
- **Explicabilidade**: SHAP, análise de erros e partial dependence plots.

---

## Estrutura esperada do projeto

```text
.
├── Pipeline_ML_Alzheimer_Reorganizado (1).ipynb
├── README.md
└── Dataset_EEG_Alzheimer/
    ├── dataset_eeg_alzheimer/
    │   ├── *.set
    │   └── *.fdt
    └── dataset_eeg_hc/
        ├── *.set
        └── *.fdt
```

### Arquivos gerados pelo notebook
- `eeg_features_brainlat_FULL_normalizado.csv`
- `eeg_features_brainlat_falhas_normalizado.csv`

---

## Requisitos

O notebook foi escrito em Python e usa bibliotecas comuns de ciência de dados e EEG.

### Dependências principais
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `mne`
- `scikit-learn`
- `xgboost`
- `shap`
- `neuroCombat` *(opcional)*

### Instalação sugerida
```bash
pip install numpy pandas matplotlib seaborn mne scikit-learn xgboost shap neuroCombat
```

Se estiver usando Jupyter/Colab, garanta que o kernel tenha acesso à biblioteca `mne` e aos arquivos `.set/.fdt`.

---

## Como usar este repositório

### 1) Baixe e organize o BrainLat
O notebook espera os dados em uma pasta local com esta lógica:

```text
Dataset_EEG_Alzheimer/
├── dataset_eeg_alzheimer/
└── dataset_eeg_hc/
```

Dentro de cada pasta devem estar os arquivos `*.set` e, idealmente, seus respectivos `*.fdt`.

### 2) Ajuste os caminhos na célula de configuração
Na **FASE 0**, altere apenas as variáveis de caminho para o seu ambiente local:

- `PASTA_RAIZ`
- `PASTA_AD`
- `PASTA_HC`

Também revise os caminhos usados na parte de análise cognitiva, caso a estrutura local seja diferente.

### 3) Execute o notebook do início ao fim
A ordem importa:

1. configuração global;
2. download/checagem dos dados;
3. exploração do dataset;
4. pré-processamento e extração de features;
5. benchmark de modelos;
6. explicabilidade e análise de resultados.

### 4) Confira os arquivos de saída
Ao final, o notebook salva:
- a base principal com todas as épocas e features;
- um CSV com falhas de leitura/processamento.

---

## Metodologia

## 1. Pré-processamento do EEG

O pipeline aplica os seguintes passos por arquivo `.set`:

1. **Leitura robusta do EEG**  
   O código tenta carregar o arquivo EEGLAB e, quando necessário, lida com a dependência do arquivo `.fdt`.

2. **Re-referência para a média global**  
   Padroniza o sinal entre canais.

3. **Filtro passa-faixa de 0.5 a 45 Hz**  
   Remove componentes fora da banda de interesse.

4. **Seleção apenas de canais EEG**  
   Exclui canais não neuronais como EOG, ECG e similares.

5. **Segmentação em épocas fixas de 4 segundos**  
   Cada sujeito é dividido em janelas temporais fixas.

6. **Normalização por sujeito via RMS global**  
   O sinal de cada sujeito é escalado para reduzir diferenças globais de amplitude.

## 2. Extração de features espectrais

As features são calculadas com **PSD via Welch** e agregadas por banda. O conjunto final utilizado no modelo é:

- `Rel_Theta_mean`
- `Rel_Alpha_mean`
- `Rel_Beta_mean`
- `Rel_Gamma_mean`
- `Razao_Theta_Alpha`
- `Razao_Theta_Beta`
- `Spectral_Entropy`

### Interpretação clínica esperada
- aumento relativo de **theta**: sinal compatível com lentificação cortical;
- redução de **alpha, beta e gamma**: compatível com disfunção neurofisiológica;
- aumento das razões **theta/alpha** e **theta/beta**: indicativo de desacoplamento e lentificação;
- alteração da **entropia espectral**: reflete mudança de complexidade do sinal.

---

## 3. Validação ML: LOSO por sujeito

A validação usada é **Leave-One-Subject-Out (LOSO)**.

### Como funciona
Em cada fold:
- um sujeito inteiro é deixado para teste;
- todos os demais sujeitos entram no treino;
- o `StandardScaler` é ajustado somente no treino;
- o modelo é treinado do zero;
- as probabilidades das épocas do sujeito de teste são médias;
- a decisão final é feita no nível do sujeito.

### Por que isso é importante
Em EEG, épocas do mesmo sujeito tendem a ser muito parecidas. Se o mesmo sujeito aparece no treino e no teste, o desempenho pode ficar artificialmente inflado. O LOSO evita esse problema e fornece uma estimativa mais realista da generalização.

---

## Modelos avaliados

O notebook compara três classificadores consolidados:

### 1. Random Forest
- robusto a não linearidades;
- boa interpretabilidade relativa;
- usado também na etapa de SHAP.

### 2. SVM com kernel RBF
- adequado para relações não lineares;
- foi calibrado para produzir probabilidades.

### 3. XGBoost
- modelo de boosting com forte capacidade preditiva;
- serve como baseline competitivo em dados tabulares.

---

## Resultados

Os resultados abaixo foram obtidos na validação LOSO por sujeito.

### Benchmark principal

| Modelo | AUC | Sensibilidade | Especificidade |
|---|---:|---:|---:|
| RandomForest | 0.701 | 0.600 | 0.625 |
| SVM_RBF | 0.684 | 0.657 | 0.656 |
| XGBoost | 0.662 | 0.600 | 0.531 |

### Matrizes de confusão
- **RandomForest**: 20 TN, 12 FP, 14 FN, 21 TP
- **SVM_RBF**: 21 TN, 11 FP, 12 FN, 23 TP
- **XGBoost**: 17 TN, 15 FP, 14 FN, 21 TP

### Leitura prática
- **RandomForest** teve o melhor AUC global.
- **SVM_RBF** apresentou o melhor equilíbrio entre sensibilidade e especificidade.
- **XGBoost** teve a menor especificidade, sugerindo maior tendência a falsos positivos.

---

## Análise EEG × cognição

O notebook também cruza a probabilidade de AD com medidas cognitivas.

### Correlações observadas
- `moca_total` vs. `P(AD)`: **rho = -0.354**, **p = 0.0101**, **n = 52**
- `ifs_total_score` vs. `P(AD)`: **rho = -0.404**, **p = 0.00267**, **n = 53**

### Interpretação
A relação negativa sugere que escores cognitivos mais altos tendem a aparecer com probabilidades menores de AD, o que é coerente com o comportamento esperado do classificador.

### Médias por grupo de erro
- **FN (AD→HC)**: MoCA média ≈ 16.45; IFS média ≈ 16.67
- **FP (HC→AD)**: MoCA média ≈ 25.25; IFS média ≈ 21.89
- **TN (HC→HC)**: MoCA média ≈ 25.80; IFS média ≈ 23.91
- **TP (AD→AD)**: MoCA média ≈ 17.39; IFS média ≈ 13.05

Isso reforça a ideia de que erros não são aleatórios: os falsos positivos e falsos negativos ocupam perfis cognitivos intermediários ou mais ambíguos.

---

## Explicabilidade com SHAP

A etapa de XAI usa **Random Forest** e agrega os valores SHAP de todos os folds LOSO.

### Ranking global de importância SHAP
1. `Rel_Theta_mean` — **0.0680**
2. `Razao_Theta_Alpha` — **0.0660**
3. `Razao_Theta_Beta` — **0.0601**
4. `Rel_Gamma_mean` — **0.0551**
5. `Rel_Beta_mean` — **0.0354**
6. `Rel_Alpha_mean` — **0.0313**
7. `Spectral_Entropy` — **0.0253**

### Interpretação
As features mais importantes apontam para um padrão clássico de AD:
- aumento de theta;
- aumento das razões theta/alpha e theta/beta;
- perda de complexidade em bandas mais rápidas.

Isso é coerente com a literatura sobre lentificação do EEG em neurodegeneração.

---

## Discussão

### O que os resultados sugerem
O pipeline consegue separar AD de HC com desempenho moderado e biologicamente plausível. O fato de as features mais importantes estarem associadas a **theta** e às razões espectrais reforça que o modelo está capturando um fenômeno neurofisiológico consistente, e não apenas ruído estatístico.

### Forças do projeto
- validação por sujeito, mais rigorosa do que validação por época;
- pipeline reprodutível e modular;
- uso de features interpretáveis;
- análise integrada entre EEG, cognição e explicabilidade.

### Limitações importantes
- desempenho ainda moderado;
- o conjunto usa apenas features espectrais resumidas, sem conectividade, temporalidade fina ou deep learning;
- parte dos arquivos HC falhou por ausência do `.fdt`, o que reduz o conjunto disponível;
- o corte em 0.5 para decisão binária é simples e pode ser otimizado;
- os PDPs no notebook são apenas ilustrativos e foram ajustados em todo o conjunto, não devendo ser usados como estimativa final de generalização.

### Possíveis extensões
- inclusão de conectividade funcional;
- extração de features por canal ou por região;
- ajuste de limiar com base em custo clínico;
- comparação com modelos adicionais;
- harmonização de domínio entre países/sítios com ComBat;
- calibração probabilística e análise de confiança.

---

## Saídas geradas

Ao final da execução, o notebook produz:

- **CSV principal** com todas as épocas e features;
- **CSV de falhas** com problemas de leitura/processamento;
- gráficos de:
  - distribuição das features;
  - correlação entre features;
  - curvas ROC;
  - matrizes de confusão;
  - distribuição das probabilidades;
  - ranking de benchmark;
  - SHAP global e summary plot;
  - análise de erros;
  - partial dependence plots;
  - relação EEG × cognição.

---

## Observações de reprodução

- Use `SEED = 42` para manter os experimentos mais estáveis.
- Execute o notebook inteiro em ordem.
- Verifique se o diretório dos dados contém os arquivos `.set` e `.fdt`.
- Se `SHAP` não estiver instalado, a parte de explicabilidade será ignorada até você instalar a dependência.

---

## Referência do notebook

Este repositório foi organizado a partir de um fluxo experimental para classificação de Alzheimer vs. HC com EEG, validado por sujeito e orientado para interpretabilidade.
