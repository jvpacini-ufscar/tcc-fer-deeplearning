# Reconhecimento de Emoções Humanas Usando Inteligência Artificial

Este repositório contém o código, os experimentos e a base de desenvolvimento para o meu Trabalho de Conclusão de Curso (TCC) em Engenharia de Computação (UFSCar). O projeto tem como foco a análise comparativa de diferentes arquiteturas de redes neurais profundas aplicadas ao Reconhecimento de Expressões Faciais (FER).

## 🗂️ Estrutura do Projeto

O repositório foi organizado utilizando as melhores práticas para projetos de Data Science e Machine Learning (Cookiecutter Data Science pattern), garantindo fácil expansão para novos modelos (RNNs, LSTMs, 3D CNNs) e datasets (RAF-DB, AffectNet, CK+).

```
tcc-fer-deeplearning/
├── README.md                   <- Este documento.
├── .gitignore                  <- Arquivos e pastas a serem ignorados pelo git (ex: datasets pesados, modelos treinados).
├── requirements.txt            <- Dependências do projeto para reprodução do ambiente.
├── data/                       <- Pasta para armazenamento local de dados (NÃO commitada no Git).
│   ├── raw/                    <- Dados originais (ex: pastas fer2013, raf-db).
│   ├── interim/                <- Dados intermediários.
│   └── processed/              <- Dados processados prontos para modelagem.
├── models/                     <- Modelos treinados (.keras, .h5, .pt) e pesos. (NÃO commitados)
├── notebooks/                  <- Jupyter Notebooks focados e numerados.
│   ├── 01_exploracao_de_dados/ <- EDA (Análise Exploratória) dos datasets.
│   ├── 02_pre_processamento/   <- Preparação de imagens, data augmentation, etc.
│   ├── 03_modelagem_cnn/       <- Experimentos e treinos de arquiteturas estáticas (VGG, ResNet, Custom).
│   ├── 04_modelagem_temporal/  <- Experimentos com sequências de vídeo (RNNs, LSTMs).
│   ├── 05_modelagem_hibrida/   <- Experimentos com abordagens multimodais/atenção.
│   └── 06_analise_e_comparacao/<- Geração de gráficos, tabelas e avaliação de métricas.
├── reports/                    <- Arquivos gerados para o TCC (ex: CSV de histórico). (Alguns versionados)
│   └── figures/                <- Gráficos, matrizes de confusão e plots para a monografia.
└── src/                        <- Código fonte reutilizável em Python (.py).
    ├── data/                   <- Scripts de extração e transformação.
    ├── models/                 <- Scripts com definição de arquiteturas.
    ├── train/                  <- Rotinas de treinamento.
    ├── evaluate/               <- Scripts de validação e testes.
    └── utils/                  <- Funções utilitárias diversas.
```

## 🚀 Como Executar

1. Crie um ambiente virtual e instale as dependências:
```bash
python -m venv venv
source venv/bin/activate  # ou venv\Scripts\activate no Windows
pip install -r requirements.txt
```

2. Baixe os datasets e extraia na pasta correspondente em `data/raw/` (ex: `data/raw/fer2013/train`).

3. Utilize os notebooks dentro da pasta `notebooks/` para rodar as etapas de exploração, treinamento e análise, na ordem em que foram numerados.

## 📅 Contexto (TCC 1 e TCC 2)

A fundamentação teórica, revisão sistemática da literatura e caracterização dos datasets foram estruturadas durante o TCC 1. Os esforços na fase atual (TCC 2) focam em:
* Implementar e comparar arquiteturas espaciais estáticas (CNNs, VGG16, ResNet50, EfficientNet).
* Implementar modelos de captura de características temporais (RNN/LSTM/CNNs 3D) para vídeos.
* Avaliar criticamente métricas de desempenho e resiliência a oclusões e viés de classe.

---
**Autor**: João Victor Pacini (RA 769729)  
**Orientador**: Prof. Alexandre Luís Magalhaes Levada  
**Instituição**: Universidade Federal de São Carlos (UFSCar)
