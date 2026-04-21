# Análise Exploratória Detalhada do FER2013

## Visão Geral

Esta pasta contém uma análise exploratória rigorosa do dataset FER2013 (Facial Expression Recognition 2013) com foco em gerar materiais de alta qualidade para inclusão na monografia.

## 📊 Artefatos Gerados

### Gráficos (Monografia-Ready)

| Arquivo | Descrição | Dimensões | Uso |
|---------|-----------|-----------|-----|
| `eda_01_distribuicao_classes.png` | Distribuição absoluta Train vs Test | 12×6" | Seção de metodologia/dataset |
| `eda_02_percentual_pie.png` | Percentual de distribuição (Pie charts) | 14×6" | Análise visual da proporção |
| `eda_03_desbalanceamento.png` | Desbalanceamento com destaque classes minoritárias | 12×6" | Discussão de desafios |
| `eda_04_amostras_visuais.png` | 4 amostras aleatórias por emoção (7×4 grid) | 10×14" | Visualização do dataset |
| `eda_05_intensidade_pixels.png` | Análise de intensidade de pixels | 14×5" | Características estatísticas |

**Especificações:**
- Resolução: 300 DPI (adequado para impressão)
- Formato: PNG com transparência
- Paleta: Husl (visualmente agradável)
- Tamanhos: Otimizados para diferentes layouts

### Relatórios

| Arquivo | Descrição | Tamanho |
|---------|-----------|---------|
| `eda_relatorio_completo.txt` | Relatório executivo estruturado | ~2.6 KB |
| `eda_statistics_summary.csv` | Estatísticas tabuladas (gerado automaticamente) | ~453 B |

## 🔍 Descobertas Principais

### Dataset Overview
- **Total**: 35.887 imagens
- **Treinamento**: 28.709 imagens
- **Validação**: 7.178 imagens
- **Classes**: 7 emoções (Anger, Disgust, Fear, Happy, Neutral, Sad, Surprise)
- **Formato**: Grayscale 48×48 pixels

### Distribuição de Classes (Train Set)

| Emoção | Quantidade | Percentual | Status |
|--------|-----------|-----------|--------|
| **Happy** | 7.215 | 25.13% | ✓ Super-representada |
| **Neutral** | 4.965 | 17.29% | ○ Balanceada |
| **Sad** | 4.830 | 16.82% | ○ Balanceada |
| **Fear** | 4.097 | 14.27% | ○ Balanceada |
| **Angry** | 3.995 | 13.92% | ○ Balanceada |
| **Surprise** | 3.171 | 11.05% | ○ Levemente sub-representada |
| **Disgust** | 436 | 1.52% | ✗ **Crítica: Sub-representada** |

### Desbalanceamento de Classes

**Ratio Máximo/Mínimo: 16.55x**

- Classe mais representada: HAPPY (7.215 imagens)
- Classe menos representada: DISGUST (436 imagens)

**Implicações:**
- Modelos treinados ingenuamente favorecerão classes maiores
- Necessidade de técnicas de balanceamento (class weights, oversampling, etc)
- Métricas: usar F1-score/Recall ao invés de apenas acurácia

### Integridade de Dados

✅ **Validação de Leakage**: 0 overlaps detectados
- Train set: 28.709 imagens únicas
- Test set: 7.178 imagens únicas
- Intersecção: 0 arquivos

**Conclusão**: Train e test sets estão perfeitamente separados. Nenhuma data leakage detectada.

### Características de Imagens

- **Tamanho**: Uniforme 48×48 pixels (grayscale)
- **Intensidade média**: ~73/255 (imagens levemente escuras)
- **Variação de intensidade**: Significativa entre classes
- **Qualidade**: In-the-wild (variações naturais de iluminação, pose, oclusões)

## 🛠️ Como Executar a EDA

### Option 1: Script Python (Recomendado)

```bash
cd /caminho/para/tcc-fer-deeplearning
./venv/Scripts/python.exe src/data/exploratory_analysis.py
```

**Tempo estimado:** ~30 segundos  
**Output:** Gráficos em `reports/figures/` + relatório em `reports/`

### Option 2: Jupyter Notebook

```bash
jupyter notebook notebooks/01_exploracao_de_dados/02_analise_exploratoria_detalhada.ipynb
```

**Vantagem:** Executar célula por célula, explorar interativamente

## 📋 Recomendações para Modelagem

Com base na análise, recomenda-se:

### 1. **Data Augmentation Agressivo**
```
- Rotação: ±20°
- Shifts: ±20% (horizontal e vertical)
- Zoom: 80-120%
- Shear: ±20%
- Brightness: ±20%
- Horizontal flip: Sim
```

### 2. **Balanceamento de Classes**
- Usar `class_weight='balanced'` durante treinamento
- Monitorar F1-score e Recall por classe
- Considerar SMOTE para classe minoritária (Disgust)

### 3. **Transfer Learning**
- ResNet50V2 ou EfficientNetB0 (ImageNet pre-training)
- Fine-tuning em 2 fases:
  - Fase 1 (Warmup): Treina apenas top classifier
  - Fase 2 (Fine-tuning): Descongela últimas camadas

### 4. **Validação Rigorosa**
- K-fold cross-validation estratificada
- Confusion matrix normalizada por classe
- Verificar leakage em cada fold

## 📚 Referências

- **Goodfellow et al. (2013)**: "Challenges in Representation Learning: A report on three machine learning contests" - Paper original do FER2013
- **Li & Deng (2020)**: "Deep Facial Expression Recognition: A Survey" - Análise comparativa de métodos
- **Plano de Projeto TCC 1**: Seção 6 - Fundamentação Teórica (Datasets e Protocolos de Avaliação)

## 📝 Notas Técnicas

### Configurações dos Gráficos
- **Style**: Seaborn v0_8-darkgrid
- **Paleta de cores**: Husl (visualmente agradável, acessível)
- **Font**: Sans-serif 11pt
- **Grid**: Ativado para legibilidade

### Reproducibilidade
- Todas as operações são determinísticas
- Seeds podem ser ajustadas em `exploratory_analysis.py`
- CSV de estatísticas pode ser usado para validação

## ✅ Checklist de Validação

- [x] Contagem de imagens verificada
- [x] Distribuição de classes plotada
- [x] Desbalanceamento quantificado (16.55x)
- [x] Data leakage verificado (0 overlaps)
- [x] Amostras visuais coletadas
- [x] Propriedades de imagens analisadas
- [x] Gráficos de qualidade monografia gerados
- [x] Relatório executivo criado
- [x] Recomendações documentadas

## 🎯 Próximos Passos

1. **✓ Análise Exploratória**: Concluída (este commit)
2. **→ Transfer Learning**: Implementação de ResNet50V2 + EfficientNetB0
3. **→ Treinamento**: Rodagem de modelos com validação rigorosa
4. **→ Análise Comparativa**: Comparação de resultados
5. **→ Monografia**: Integração dos gráficos/descobertas

---

**Última atualização:** 21 de abril de 2026  
**Responsável:** Análise Exploratória Automatizada (EDA Pipeline)  
**Status:** ✅ Concluído e Validado
