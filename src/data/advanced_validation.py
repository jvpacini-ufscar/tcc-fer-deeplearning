#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validacao Avancada e Analise de Cross-Validation Estratificada
Garante que os dados estao prontos para treinamento com estruturas robustas
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Configuracoes
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAIN_DIR = os.path.join(BASE_DIR, 'data/raw/fer2013/train')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
FIGURES_DIR = os.path.join(REPORTS_DIR, 'figures')

os.makedirs(FIGURES_DIR, exist_ok=True)

EMOTION_CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


def collect_all_images():
    """Coleta todos os caminhos de imagens com labels."""
    image_paths = []
    labels = []
    label_map = {emotion: idx for idx, emotion in enumerate(EMOTION_CLASSES)}
    
    for emotion in EMOTION_CLASSES:
        emotion_path = os.path.join(TRAIN_DIR, emotion)
        if os.path.exists(emotion_path):
            files = [f for f in os.listdir(emotion_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
            for fname in files:
                image_paths.append(os.path.join(emotion_path, fname))
                labels.append(label_map[emotion])
    
    return np.array(image_paths), np.array(labels)


def analyze_kfold_stratification(n_splits=5):
    """Analisa K-Fold estratificado e verifica distribuicao em cada fold."""
    print("\n" + "="*80)
    print("ANALISE DE CROSS-VALIDATION ESTRATIFICADA (K-FOLD)")
    print("="*80)
    
    image_paths, labels = collect_all_images()
    total_samples = len(image_paths)
    
    print(f"\n[INFO] Total de imagens: {total_samples}")
    print(f"[INFO] Numero de folds: {n_splits}")
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_data = []
    all_fold_distributions = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(image_paths, labels)):
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]
        
        # Conta distribuicao
        train_counts = np.bincount(train_labels, minlength=len(EMOTION_CLASSES))
        val_counts = np.bincount(val_labels, minlength=len(EMOTION_CLASSES))
        
        fold_data.append({
            'Fold': fold_idx + 1,
            'Train_Samples': len(train_idx),
            'Val_Samples': len(val_idx),
            'Train_Ratio': len(train_idx) / total_samples,
            'Val_Ratio': len(val_idx) / total_samples
        })
        
        # Armazena distribuicao por classe
        for class_idx, emotion in enumerate(EMOTION_CLASSES):
            all_fold_distributions.append({
                'Fold': fold_idx + 1,
                'Emocao': emotion.capitalize(),
                'Train_Count': train_counts[class_idx],
                'Train_Pct': (train_counts[class_idx] / len(train_idx) * 100),
                'Val_Count': val_counts[class_idx],
                'Val_Pct': (val_counts[class_idx] / len(val_idx) * 100)
            })
        
        print(f"\n[FOLD {fold_idx + 1}]")
        print(f"  Train: {len(train_idx):5d} amostras ({len(train_idx)/total_samples*100:5.1f}%)")
        print(f"  Val:   {len(val_idx):5d} amostras ({len(val_idx)/total_samples*100:5.1f}%)")
        
        # Verifica se distribuicao eh similar
        ratios = []
        for i in range(len(EMOTION_CLASSES)):
            if train_counts[i] > 0:
                ratio = (val_counts[i] / len(val_idx)) / (train_counts[i] / len(train_idx))
                ratios.append(ratio)
        
        min_ratio = min(ratios)
        max_ratio = max(ratios)
        
        print(f"  Consistencia: min_ratio={min_ratio:.3f}, max_ratio={max_ratio:.3f}")
        if 0.9 < min_ratio and max_ratio < 1.1:
            print(f"  [OK] Distribuicao estratificada balanceada!")
        else:
            print(f"  [AVISO] Distribuicao com desvio >10%")
    
    df_folds = pd.DataFrame(fold_data)
    df_distributions = pd.DataFrame(all_fold_distributions)
    
    return df_folds, df_distributions, skf, image_paths, labels


def plot_kfold_distribution(df_distributions):
    """Plota distribuicao de classes em cada fold."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.flatten()
    
    for fold_idx, fold_num in enumerate(sorted(df_distributions['Fold'].unique())):
        ax = axes[fold_idx]
        fold_data = df_distributions[df_distributions['Fold'] == fold_num]
        
        x = np.arange(len(EMOTION_CLASSES))
        width = 0.35
        
        train_vals = fold_data['Train_Pct'].values
        val_vals = fold_data['Val_Pct'].values
        
        ax.bar(x - width/2, train_vals, width, label='Train', alpha=0.8)
        ax.bar(x + width/2, val_vals, width, label='Val', alpha=0.8)
        
        ax.set_title(f'Fold {fold_num}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Percentual (%)', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([e[:3].upper() for e in EMOTION_CLASSES], rotation=45)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 35])
    
    axes[-1].axis('off')
    
    plt.suptitle('Distribuicao de Classes em Cada Fold (K-Fold Estratificado)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    fig_path = os.path.join(FIGURES_DIR, 'advanced_01_kfold_stratification.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[SALVO] {fig_path}")


def analyze_augmentation():
    """Analisa o efeito de data augmentation (antes/depois)."""
    print("\n" + "="*80)
    print("ANALISE DE DATA AUGMENTATION")
    print("="*80)
    
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    # Carrega uma imagem de exemplo
    sample_emotion_path = os.path.join(TRAIN_DIR, 'happy')
    sample_file = os.listdir(sample_emotion_path)[0]
    sample_img_path = os.path.join(sample_emotion_path, sample_file)
    
    img = Image.open(sample_img_path).convert('L')
    img_array = np.array(img).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=(0, -1))  # (1, 48, 48, 1)
    
    # Data augmentation
    augmentation_config = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    
    # Gera 8 augmentacoes
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    axes = axes.flatten()
    
    # Original
    axes[0].imshow(img_array[0, :, :, 0], cmap='gray')
    axes[0].set_title('Original', fontsize=11, fontweight='bold')
    axes[0].axis('off')
    
    # Augmentadas
    aug_idx = 1
    for img_aug in augmentation_config.flow(img_array, batch_size=1):
        if aug_idx >= 9:
            break
        axes[aug_idx].imshow(img_aug[0, :, :, 0], cmap='gray')
        axes[aug_idx].set_title(f'Augmentacao {aug_idx}', fontsize=10)
        axes[aug_idx].axis('off')
        aug_idx += 1
    
    plt.suptitle('Efeito de Data Augmentation (Amostra: Happy)', 
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    fig_path = os.path.join(FIGURES_DIR, 'advanced_02_augmentation_effect.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[SALVO] {fig_path}")
    print("\n[INFO] Augmentacao aplicada com sucesso. Variacoes:")
    print("  * Rotacao: +/-20°")
    print("  * Shifts: +/-20% (H/V)")
    print("  * Zoom: 80-120%")
    print("  * Shear: +/-20%")
    print("  * Brightness: 80-120%")
    print("  * Horizontal flip: Sim")


def generate_execution_plan():
    """Gera plano estruturado de execucao para Passo 2."""
    plan = """================================================================================
PLANO DE EXECUCAO ESTRUTURADO - PASSO 2: TRANSFER LEARNING
================================================================================

OBJETIVO
────────
Treinar dois modelos de deep learning (ResNet50V2 + EfficientNetB0) com 
validacao rigorosa e metricas confiáveis para atingir 60%+ de acuracia.

ARQUITETURA DE VALIDACAO (K-FOLD ESTRATIFICADO)
────────────────────────────────────────────────

[OK] Validacao: 5-Fold Estratificado
[OK] Separacao: 80% train / 20% val por fold
[OK] Estrategia: Mantem distribuicao de classes em cada fold
[OK] Random Seed: 42 (para reproducibilidade)

PLANO DE TREINAMENTO
────────────────────

FASE 1: RESNET50V2 COM FINE-TUNING
────────────────────────────────────

Arquitetura:
  ├─ Base: ResNet50V2 (ImageNet pre-training)
  ├─ Input: 224×224×3 RGB
  ├─ Top: GlobalAveragePooling2D → Dense(256) → Dropout(0.5) → Dense(7)
  └─ Parametros: ~25.6M

Fase 1 - Warmup (15 epochs):
  ├─ Base Model: Frozen
  ├─ Treina: Apenas top classifier
  ├─ Learning Rate: 1e-3
  ├─ Otimizador: Adam
  ├─ Loss: Categorical Crossentropy
  ├─ Class Weights: Balanced (automatico)
  └─ Early Stopping: Paciencia=5 (monitora val_loss)

Fase 2 - Fine-tuning (30 epochs):
  ├─ Base Model: Descongela ultimas 30 camadas
  ├─ Treina: Toda a rede
  ├─ Learning Rate: 1e-5 (MUITO pequeno)
  ├─ ReduceLROnPlateau: Factor=0.5, Paciencia=3
  ├─ Early Stopping: Paciencia=7
  └─ Checkpoint: Salva melhor modelo

FASE 2: EFFICIENTNETB0 (OTIMIZADO)
────────────────────────────────────

Arquitetura:
  ├─ Base: EfficientNetB0 (ImageNet pre-training)
  ├─ Input: 224×224×3 RGB
  ├─ Top: GlobalAveragePooling2D → Dense(128) → Dropout(0.3) → Dense(7)
  └─ Parametros: ~5.3M (mais leve)

Mesma estrategia de 2 fases, mas:
  ├─ Descongela: Ultimas 20 camadas (ao invés de 30)
  ├─ Dropout: 0.3 (ao invés de 0.5)
  └─ Tempo: ~30% mais rápido que ResNet50V2

TECNICAS DE REGULARIZACAO
──────────────────────────

Data Augmentation (agressivo):
  [OK] Rotation: +/-20°
  [OK] Width/Height Shift: +/-20%
  [OK] Zoom: 80-120%
  [OK] Shear: +/-20%
  [OK] Brightness: 80-120%
  [OK] Horizontal Flip: Yes

Class Balancing:
  [OK] class_weight='balanced' em fit()
  [OK] Automatico baseado em frequência de classe
  [OK] Disgust (436 imagens) recebera peso ~26x
  [OK] Happy (7215 imagens) recebera peso ~1.6x

Learning Rate Schedule:
  [OK] ReduceLROnPlateau monitora val_loss
  [OK] Reduz por fator 0.5 quando plateau eh atingido
  [OK] Min learning rate: 1e-7 (protecao contra zero)

METRICAS DE AVALIACAO
─────────────────────

Fase de Validacao (por epoch):
  * Loss (categorical crossentropy)
  * Accuracy global
  * Val Loss (monitorado para early stopping)
  * Val Accuracy

Fase de Teste (ao final):
  [OK] Acuracia global
  [OK] Precision por classe
  [OK] Recall por classe
  [OK] F1-score por classe
  [OK] Matriz de Confusao (normalizada)
  [OK] Curva ROC-AUC (se aplicável)

Artefatos Salvos:
  ├─ models/resnet50v2_fase1.keras (checkpoint)
  ├─ models/resnet50v2_fase2.keras (melhor modelo)
  ├─ models/efficientnetb0_fase1.keras
  ├─ models/efficientnetb0_fase2.keras
  ├─ reports/metrics_resnet50v2.csv (classification report)
  ├─ reports/metrics_efficientnetb0.csv
  ├─ reports/figures/cm_resnet50v2.png (confusion matrix)
  ├─ reports/figures/cm_efficientnetb0.png
  ├─ reports/historico_resnet50v2.csv (training history)
  └─ reports/historico_efficientnetb0.csv

CRONOGRAMA E TEMPO ESTIMADO
────────────────────────────

Com GPU NVIDIA (recomendado):
  ├─ ResNet50V2 Setup: ~5 min
  ├─ ResNet50V2 Training (2 phases): ~2.5 horas
  ├─ ResNet50V2 Evaluation: ~10 min
  ├─ EfficientNetB0 Setup: ~5 min
  ├─ EfficientNetB0 Training: ~1.5 horas
  ├─ EfficientNetB0 Evaluation: ~10 min
  └─ TOTAL: ~4.5 horas

Sem GPU (CPU apenas):
  └─ TOTAL: ~20-30 horas (nao recomendado)

PLANO DE ANALISE POS-TREINAMENTO
─────────────────────────────────

1. Comparativa de Resultados:
   ├─ Qual modelo tem melhor acuracia geral?
   ├─ Qual tem melhor F1-score por classe?
   ├─ Qual eh mais rápido (FPS)?
   └─ Ha overfitting? (comparar train vs val)

2. Analise de Erros:
   ├─ Quais classes são confundidas?
   ├─ Disgust: conseguimos melhorar sua recall?
   ├─ Happy: está fazendo overfitting?
   └─ Classes balanceadas melhoraram com class_weight?

3. Visualizacoes Finais:
   ├─ Gráficos de loss/accuracy ao longo do treino
   ├─ Confusion matrices normalizadas
   ├─ Comparativa de métricas por classe
   └─ Tudo pronto para monografia

CHECKPOINTS E SALVAMENTOS INTERMEDIARIOS
──────────────────────────────────────────

Automatico:
  [OK] ModelCheckpoint: Salva melhor modelo em cada fase
  [OK] Early Stopping: Evita treinar além do necessário
  [OK] ReduceLROnPlateau: Ajusta taxa de aprendizado

Manual (você pode interromper):
  [OK] Se GPU ficar instavel, pode pausar e retomar
  [OK] Se achar que learning rate está ruim, pode ajustar
  [OK] Se suspeitar de overfitting, pode parar cedo

POTENCIAIS PROBLEMAS E SOLUCOES
────────────────────────────────

Problema: GPU fora de memória (OOM)
  → Solucao: Reduzir batch_size de 64 para 32

Problema: Learning rate muito alto (loss aumenta)
  → Solucao: Reduzir de 1e-3 para 5e-4

Problema: Modelo não está aprendendo (loss plano)
  → Solucao: Verificar se class_weights estão sendo aplicadas

Problema: Overfitting extremo (train 90%, val 55%)
  → Solucao: Aumentar dropout ou augmentation

PROXIMOS PASSOS APOS TREINO
────────────────────────────

1. Analise Comparativa:
   └─ Comparar ResNet vs EfficientNet
   └─ Qual escolher para producao?

2. Otimizacoes Opcionais:
   └─ Tentar técnicas adicionais (SMOTE, Focal Loss, etc)
   └─ Testar em outros datasets (RAF-DB)

3. Documentacao:
   └─ Escrever seção de Resultados para monografia
   └─ Incluir gráficos e tabelas

4. Apresentacao:
   └─ Preparar slides com descobertas
   └─ Praticar apresentação

================================================================================
INSTRUCOES DE EXECUCAO
================================================================================

OPCAO 1: Script Python (Recomendado - Automatizado)
────────────────────────────────────────────────────
cd /caminho/para/tcc-fer-deeplearning
./venv/Scripts/python.exe src/train/transfer_learning_fer2013.py

Tempo: ~4.5 horas (com GPU)
Output: Todos os gráficos, métricas e modelos salvos automaticamente

OPCAO 2: Jupyter Notebook (Interativo - Melhor para aprender)
────────────────────────────────────────────────────────────
jupyter notebook notebooks/03_modelagem_cnn/04_transfer_learning_fer2013.ipynb

Tempo: 4.5 horas (com quebras para análise)
Output: Executa célula por célula, pode visualizar cada etapa

OPCAO 3: Hibrida (Eu recomendo)
───────────────────────────────
1. Executar script para treino rápido
2. Usar notebook para análise e visualização dos resultados

================================================================================
RECOMENDACOES FINAIS
================================================================================

[OK] USE GPU: Diferenca de 5-6x no tempo de treinamento
[OK] MONITORE DURANTE EXECUCAO: Veja loss/accuracy em tempo real
[OK] SALVE CHECKPOINTS: Em caso de interrupcao, pode retomar
[OK] DOCUMENTE TUDO: Qualquer alteracao, anote no git
[OK] COMPARACAO FINAL: Qual modelo eh melhor? Por quê?

================================================================================
STATUS: PRONTO PARA COMECO PASSO 2
================================================================================

Você tem:
  [OK] Dataset validado (EDA completa)
  [OK] Cross-validation estratificada planejada
  [OK] Tecnicas de augmentation definidas
  [OK] Arquiteturas selecionadas (ResNet50V2 + EfficientNetB0)
  [OK] Metricas de avaliacao claras
  [OK] Plano de execucao estruturado

Próximo passo: Começar o treinamento!

"""
    
    return plan


def save_execution_plan(plan_text):
    """Salva plano de execucao em arquivo."""
    plan_path = os.path.join(REPORTS_DIR, 'EXECUTION_PLAN_PASSO2.txt')
    with open(plan_path, 'w', encoding='utf-8') as f:
        f.write(plan_text)
    print(f"\n[SALVO] Plano de execucao: {plan_path}")
    return plan_path


def main():
    """Executa validacao avancada completa."""
    print("\n" + "="*80)
    print("VALIDACAO AVANCADA - PREPARACAO PARA PASSO 2")
    print("="*80)
    
    # Analise K-Fold
    print("\n[1/3] Analisando K-Fold Estratificado...")
    df_folds, df_distributions, skf, image_paths, labels = analyze_kfold_stratification(n_splits=5)
    plot_kfold_distribution(df_distributions)
    
    print("\nResumo de Folds:")
    print(df_folds.to_string(index=False))
    
    # Analise de Augmentation
    print("\n[2/3] Analisando efeito de Data Augmentation...")
    analyze_augmentation()
    
    # Gerar plano de execucao
    print("\n[3/3] Gerando Plano de Execucao Estruturado...")
    plan = generate_execution_plan()
    plan_path = save_execution_plan(plan)
    
    # Salvar metadados
    metadata = {
        'Total_Imagens': len(image_paths),
        'Num_Classes': len(EMOTION_CLASSES),
        'Num_Folds': 5,
        'Random_State': 42,
        'Augmentation_Config': 'rotation=20, shifts=0.2, zoom=0.2, shear=0.2, brightness=0.2, flip=True'
    }
    
    metadata_df = pd.DataFrame([metadata])
    metadata_path = os.path.join(REPORTS_DIR, 'training_metadata.csv')
    metadata_df.to_csv(metadata_path, index=False)
    print(f"[SALVO] Metadados: {metadata_path}")
    
    print("\n" + "="*80)
    print("VALIDACAO AVANCADA CONCLUIDA")
    print("="*80)
    print("\nArtefatos gerados:")
    print(f"  [OK] Grafico K-Fold: {FIGURES_DIR}/advanced_01_kfold_stratification.png")
    print(f"  [OK] Grafico Augmentation: {FIGURES_DIR}/advanced_02_augmentation_effect.png")
    print(f"  [OK] Plano de Execucao: {plan_path}")
    print(f"  [OK] Metadados: {metadata_path}")
    print("\n" + "="*80 + "\n")
    
    # Printa o plano
    print(plan)


if __name__ == "__main__":
    main()
