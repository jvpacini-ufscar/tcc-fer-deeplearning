#!/usr/bin/env python3
"""
Exploratory Data Analysis (EDA) - RAF-DB Dataset
Gera grficos, estatsticas e relatrio para a monografia
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Configuracoes para qualidade monografia
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'sans-serif'

# Paths
BASE_DIR = "."
TRAIN_DIR = os.path.join(BASE_DIR, 'data/raw/raf-db/train')
VAL_DIR = os.path.join(BASE_DIR, 'data/raw/raf-db/test')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
FIGURES_DIR = os.path.join(REPORTS_DIR, 'figures')

os.makedirs(FIGURES_DIR, exist_ok=True)

EMOTION_CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


def explore_directory_structure(root_dir):
    """Conta arquivos por classe."""
    class_counts = {}
    for emotion in EMOTION_CLASSES:
        emotion_path = os.path.join(root_dir, emotion)
        if os.path.exists(emotion_path):
            files = [f for f in os.listdir(emotion_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
            class_counts[emotion] = len(files)
        else:
            class_counts[emotion] = 0
    return class_counts


def plot_distribution_rafdb(train_counts, test_counts, total_train, total_test):
    """Gráfico 1: Distribuição absoluta (Train vs Test)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(EMOTION_CLASSES))
    width = 0.35
    
    train_vals = [train_counts[e] for e in EMOTION_CLASSES]
    test_vals = [test_counts[e] for e in EMOTION_CLASSES]
    
    bars1 = ax.bar(x - width/2, train_vals, width, label='Train', alpha=0.8, color='#2E86AB')
    bars2 = ax.bar(x + width/2, test_vals, width, label='Test', alpha=0.8, color='#A23B72')
    
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Emoção', fontsize=12, fontweight='bold')
    ax.set_ylabel('Número de Imagens', fontsize=12, fontweight='bold')
    ax.set_title('Distribuição de Classes no RAF-DB (Train vs Test)', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([e.capitalize() for e in EMOTION_CLASSES], rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, 'eda_rafdb_01_distribuicao.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SALVO] {fig_path}")


def plot_imbalance_rafdb(train_counts, total_train):
    """Gráfico 2: Análise de desbalanceamento."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    df_train = pd.DataFrame({
        'Emocao': [e.capitalize() for e in EMOTION_CLASSES],
        'Quantidade': list(train_counts.values())
    })
    df_train_sorted = df_train.sort_values('Quantidade', ascending=False)
    
    colors_imbalance = ['#FF6B6B' if x < 500 else '#4ECDC4' for x in df_train_sorted['Quantidade']]
    bars = ax.barh(df_train_sorted['Emocao'], df_train_sorted['Quantidade'], color=colors_imbalance, alpha=0.8)
    
    for i, bar in enumerate(bars):
        width = bar.get_width()
        pct = (width / total_train * 100)
        ax.text(width, bar.get_y() + bar.get_height()/2., f' {int(width)} ({pct:.1f}%)', ha='left', va='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Número de Imagens', fontsize=12, fontweight='bold')
    ax.set_title('Desbalanceamento de Classes no RAF-DB (Train Set)', fontsize=13, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, 'eda_rafdb_02_desbalanceamento.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SALVO] {fig_path}")


def plot_sample_images_rafdb():
    """Gráfico 3: Amostras visuais por emoção."""
    fig, axes = plt.subplots(7, 4, figsize=(10, 14))
    fig.suptitle('Amostras do RAF-DB por Emoção (Aligned RGB)', fontsize=14, fontweight='bold', y=0.995)
    
    for row, emotion in enumerate(EMOTION_CLASSES):
        emotion_path = os.path.join(TRAIN_DIR, emotion)
        files = [f for f in os.listdir(emotion_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        sampled = np.random.choice(files, min(4, len(files)), replace=False)
        
        for col, fname in enumerate(sampled):
            img = Image.open(os.path.join(emotion_path, fname))
            ax = axes[row, col]
            ax.imshow(img)
            ax.axis('off')
            if col == 0:
                ax.set_ylabel(emotion.capitalize(), fontsize=11, fontweight='bold', labelpad=10)
    
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, 'eda_rafdb_03_amostras.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SALVO] {fig_path}")


def analyze_resolutions():
    """Analisa resoluções das imagens."""
    resolutions = []
    for emotion in EMOTION_CLASSES:
        emotion_path = os.path.join(TRAIN_DIR, emotion)
        files = [f for f in os.listdir(emotion_path) if f.endswith(('.png', '.jpg', '.jpeg'))][:100]
        for fname in files:
            img = Image.open(os.path.join(emotion_path, fname))
            resolutions.append(img.size) # (width, height)
    
    df_res = pd.DataFrame(resolutions, columns=['Width', 'Height'])
    print("\nEstatísticas de Resolução (Amostra 700 imagens):")
    print(df_res.describe())
    return df_res


def main():
    print("\n" + "="*80)
    print("EXPLORATORY DATA ANALYSIS - RAF-DB")
    print("="*80)
    
    train_counts = explore_directory_structure(TRAIN_DIR)
    test_counts = explore_directory_structure(VAL_DIR)
    
    total_train = sum(train_counts.values())
    total_test = sum(test_counts.values())
    
    print(f"  Train: {total_train} imagens")
    print(f"  Test: {total_test} imagens")
    
    plot_distribution_rafdb(train_counts, test_counts, total_train, total_test)
    plot_imbalance_rafdb(train_counts, total_train)
    plot_sample_images_rafdb()
    analyze_resolutions()
    
    # Salvar CSV de estatísticas
    stats = pd.DataFrame({
        'Emocao': [e.capitalize() for e in EMOTION_CLASSES],
        'Train': [train_counts[e] for e in EMOTION_CLASSES],
        'Test': [test_counts[e] for e in EMOTION_CLASSES]
    })
    stats.to_csv(os.path.join(REPORTS_DIR, 'eda_rafdb_statistics.csv'), index=False)
    
    print("\n" + "="*80)
    print("ANÁLISE EXPLORATÓRIA RAF-DB CONCLUÍDA")
    print("="*80)

if __name__ == "__main__":
    main()
