#!/usr/bin/env python3
"""
Exploratory Data Analysis (EDA) - FER2013 Dataset
Gera gráficos, estatísticas e relatório para a monografia
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
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
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAIN_DIR = os.path.join(BASE_DIR, 'data/raw/fer2013/train')
VAL_DIR = os.path.join(BASE_DIR, 'data/raw/fer2013/test')
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


def plot_distribution(train_counts, test_counts, total_train, total_test):
    """Gráfico 1: Distribuição absoluta (Train vs Test)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(EMOTION_CLASSES))
    width = 0.35
    
    train_vals = [train_counts[e] for e in EMOTION_CLASSES]
    test_vals = [test_counts[e] for e in EMOTION_CLASSES]
    
    bars1 = ax.bar(x - width/2, train_vals, width, label='Train', alpha=0.8, color='#2E86AB')
    bars2 = ax.bar(x + width/2, test_vals, width, label='Test', alpha=0.8, color='#A23B72')
    
    # Valores nas barras
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Emocao', fontsize=12, fontweight='bold')
    ax.set_ylabel('Numero de Imagens', fontsize=12, fontweight='bold')
    ax.set_title('Distribuicao de Classes no FER2013 (Train vs Test)', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([e.capitalize() for e in EMOTION_CLASSES], rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, 'eda_01_distribuicao_classes.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SALVO] {fig_path}")


def plot_pie_distribution(train_counts, test_counts, total_train, total_test):
    """Gráfico 2: Percentual (Pie charts)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    train_vals = [train_counts[e] for e in EMOTION_CLASSES]
    test_vals = [test_counts[e] for e in EMOTION_CLASSES]
    colors = sns.color_palette("husl", len(EMOTION_CLASSES))
    
    axes[0].pie(train_vals, labels=[e.capitalize() for e in EMOTION_CLASSES],
                autopct='%1.1f%%', colors=colors, startangle=90)
    axes[0].set_title(f'Train Set (n={total_train})', fontsize=12, fontweight='bold')
    
    axes[1].pie(test_vals, labels=[e.capitalize() for e in EMOTION_CLASSES],
                autopct='%1.1f%%', colors=colors, startangle=90)
    axes[1].set_title(f'Test Set (n={total_test})', fontsize=12, fontweight='bold')
    
    plt.suptitle('Percentual de Distribuicao por Emocao no FER2013', 
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    fig_path = os.path.join(FIGURES_DIR, 'eda_02_percentual_pie.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SALVO] {fig_path}")


def plot_imbalance(train_counts, total_train):
    """Gráfico 3: Análise de desbalanceamento."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    df_train = pd.DataFrame({
        'Emocao': [e.capitalize() for e in EMOTION_CLASSES],
        'Quantidade': list(train_counts.values())
    })
    df_train_sorted = df_train.sort_values('Quantidade', ascending=False)
    
    colors_imbalance = ['#FF6B6B' if x < 1000 else '#4ECDC4' for x in df_train_sorted['Quantidade']]
    bars = ax.barh(df_train_sorted['Emocao'], df_train_sorted['Quantidade'], 
                   color=colors_imbalance, alpha=0.8)
    
    for i, bar in enumerate(bars):
        width = bar.get_width()
        pct = (width / total_train * 100)
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                f' {int(width)} ({pct:.1f}%)', ha='left', va='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Numero de Imagens', fontsize=12, fontweight='bold')
    ax.set_title('Desbalanceamento de Classes no FER2013 (Train Set)\nVermelho: Classes minoritarias (<1000)',
                 fontsize=13, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='x')
    
    mean_val = df_train_sorted['Quantidade'].mean()
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Media ({mean_val:.0f})')
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, 'eda_03_desbalanceamento.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SALVO] {fig_path}")


def load_sample_images(class_path, n_samples=4):
    """Carrega amostras aleatórias de uma classe."""
    if not os.path.exists(class_path):
        return []
    
    files = [f for f in os.listdir(class_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    sampled = np.random.choice(files, min(n_samples, len(files)), replace=False)
    
    images = []
    for fname in sampled:
        img_path = os.path.join(class_path, fname)
        img = Image.open(img_path).convert('L')
        images.append(np.array(img))
    
    return images


def plot_sample_images():
    """Gráfico 4: Amostras visuais por emoção."""
    fig, axes = plt.subplots(7, 4, figsize=(10, 14))
    fig.suptitle('Amostras do FER2013 por Emocao (Escala de Cinza)', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    for row, emotion in enumerate(EMOTION_CLASSES):
        emotion_path = os.path.join(TRAIN_DIR, emotion)
        images = load_sample_images(emotion_path, n_samples=4)
        
        for col, img in enumerate(images):
            ax = axes[row, col]
            ax.imshow(img, cmap='gray')
            ax.axis('off')
            
            if col == 0:
                ax.set_ylabel(emotion.capitalize(), fontsize=11, fontweight='bold', labelpad=10)
    
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, 'eda_04_amostras_visuais.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SALVO] {fig_path}")


def analyze_image_properties():
    """Analisa propriedades das imagens."""
    properties = []
    
    for emotion in EMOTION_CLASSES:
        emotion_path = os.path.join(TRAIN_DIR, emotion)
        files = [f for f in os.listdir(emotion_path) if f.endswith(('.png', '.jpg', '.jpeg'))][:50]
        
        for fname in files:
            img_path = os.path.join(emotion_path, fname)
            img = Image.open(img_path).convert('L')
            img_array = np.array(img)
            
            properties.append({
                'Emocao': emotion.capitalize(),
                'Largura': img_array.shape[1],
                'Altura': img_array.shape[0],
                'Intensidade_Media': img_array.mean(),
                'Intensidade_Desvio': img_array.std(),
                'Intensidade_Min': img_array.min(),
                'Intensidade_Max': img_array.max()
            })
    
    return pd.DataFrame(properties)


def plot_intensity(df_properties):
    """Gráfico 5: Intensidade de pixels."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    intensity_by_emotion = df_properties.groupby('Emocao')['Intensidade_Media'].agg(['mean', 'std'])
    
    x = np.arange(len(intensity_by_emotion))
    axes[0].bar(x, intensity_by_emotion['mean'], yerr=intensity_by_emotion['std'],
                capsize=5, alpha=0.8, color=sns.color_palette("husl", len(EMOTION_CLASSES)))
    axes[0].set_xlabel('Emocao', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Intensidade Media', fontsize=11, fontweight='bold')
    axes[0].set_title('Intensidade Media de Pixel por Emocao', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(intensity_by_emotion.index, rotation=45, ha='right')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Boxplot
    intensity_std_by_emotion = df_properties.groupby('Emocao')['Intensidade_Desvio'].apply(list)
    axes[1].boxplot([intensity_std_by_emotion[e] for e in intensity_std_by_emotion.index],
                    labels=intensity_std_by_emotion.index)
    axes[1].set_xlabel('Emocao', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Desvio Padrao de Intensidade', fontsize=11, fontweight='bold')
    axes[1].set_title('Variacao de Intensidade por Emocao', fontsize=12, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, 'eda_05_intensidade_pixels.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SALVO] {fig_path}")


def check_data_leakage():
    """Verifica data leakage entre train e test."""
    train_files = set()
    test_files = set()
    
    for emotion in EMOTION_CLASSES:
        emotion_path = os.path.join(TRAIN_DIR, emotion)
        if os.path.exists(emotion_path):
            files = [f for f in os.listdir(emotion_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
            train_files.update(files)
    
    for emotion in EMOTION_CLASSES:
        emotion_path = os.path.join(VAL_DIR, emotion)
        if os.path.exists(emotion_path):
            files = [f for f in os.listdir(emotion_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
            test_files.update(files)
    
    overlap = train_files.intersection(test_files)
    return len(overlap), len(train_files), len(test_files)


def generate_report(train_counts, test_counts, total_train, total_test):
    """Gera relatório em texto."""
    ratio = max(train_counts.values()) / min(v for v in train_counts.values() if v > 0)
    
    report = f"""{'='*80}
RELATORIO EXPLORATÓRIO - FER2013 DATASET
{'='*80}

1. RESUMO EXECUTIVO
{'-'*80}
O dataset FER2013 eh composto por 35.887 imagens em escala de cinza (48x48 pixels)
divididas em 7 classes de emocoes e distribuidas em splits de treinamento ({total_train})
e validacao ({total_test}).

2. DISTRIBUICAO DE CLASSES
{'-'*80}

TRAIN SET:
"""
    
    for emotion in EMOTION_CLASSES:
        count = train_counts[emotion]
        pct = (count / total_train * 100)
        report += f"  {emotion.upper():12s}: {count:5d} imagens ({pct:5.2f}%)\n"
    
    report += f"\nTEST SET:\n"
    for emotion in EMOTION_CLASSES:
        count = test_counts[emotion]
        pct = (count / total_test * 100)
        report += f"  {emotion.upper():12s}: {count:5d} imagens ({pct:5.2f}%)\n"
    
    report += f"""
3. DESAFIOS IDENTIFICADOS
{'-'*80}

3.1 DESBALANCEAMENTO DE CLASSES (CRITICO)
  * Ratio maximo/minimo: {ratio:.2f}x
  * Classe super-representada: HAPPY ({train_counts['happy']} imagens)
  * Classe sub-representada: DISGUST ({train_counts['disgust']} imagens)
  
  Implicacao: Modelos treinados ingenuamente favorecero classes maiores.
  Mitigacao: Usar class_weight='balanced' ou oversampling.

3.2 VARIACAO DE QUALIDADE
  * Imagens capturadas in-the-wild (condicoes reais)
  * Variacoes de iluminacao, pose, ocluses
  
  Mitigacao: Data augmentation agressivo durante treinamento.

3.3 TAMANHO PEQUENO
  * Resolucao: 48x48 pixels (grayscale)
  
  Mitigacao: Transfer learning com models pre-treinados.

4. RECOMENDACOES PARA MODELAGEM
{'-'*80}

1. DATA AUGMENTATION
   - Rotacao +/-20°
   - Shifts +/-20%
   - Zoom 80-120%
   - Brightness +/-20%

2. BALANCEAMENTO
   - Usar class_weight='balanced'
   - Monitorar F1-score (nao apenas acuracia)

3. TRANSFER LEARNING
   - ResNet50V2 ou EfficientNetB0
   - Fine-tuning em 2 fases

4. VALIDACAO
   - Monitorar F1-score por classe
   - Analisar confusion matrix
   - Verificar leakage

{'='*80}
Fim do Relatorio
{'='*80}
"""
    
    return report


def save_statistics(train_counts, test_counts, total_train, total_test):
    """Salva estatísticas em CSV."""
    stats = pd.DataFrame({
        'Emocao': [e.capitalize() for e in EMOTION_CLASSES],
        'Train_Quantidade': [train_counts[e] for e in EMOTION_CLASSES],
        'Train_Percentual': [(train_counts[e]/total_train*100) for e in EMOTION_CLASSES],
        'Test_Quantidade': [test_counts[e] for e in EMOTION_CLASSES],
        'Test_Percentual': [(test_counts[e]/total_test*100) for e in EMOTION_CLASSES]
    })
    
    stats_path = os.path.join(REPORTS_DIR, 'eda_statistics_summary.csv')
    stats.to_csv(stats_path, index=False)
    print(f"[SALVO] {stats_path}")
    return stats


def main():
    """Executa análise exploratória completa."""
    print("\n" + "="*80)
    print("EXPLORATORY DATA ANALYSIS - FER2013")
    print("="*80)
    
    # Explore data
    print("\n[1/7] Explorando estrutura de diretórios...")
    train_counts = explore_directory_structure(TRAIN_DIR)
    test_counts = explore_directory_structure(VAL_DIR)
    
    total_train = sum(train_counts.values())
    total_test = sum(test_counts.values())
    
    print(f"  Train: {total_train} imagens")
    print(f"  Test: {total_test} imagens")
    
    # Check leakage
    print("\n[2/7] Verificando data leakage...")
    overlap, train_unique, test_unique = check_data_leakage()
    if overlap == 0:
        print(f"  OK: Nenhuma sobreposição detectada.")
    else:
        print(f"  AVISO: {overlap} arquivos em ambos os sets!")
    
    # Generate plots
    print("\n[3/7] Gerando gráfico de distribuição...")
    plot_distribution(train_counts, test_counts, total_train, total_test)
    
    print("[4/7] Gerando gráficos de pizza...")
    plot_pie_distribution(train_counts, test_counts, total_train, total_test)
    
    print("[5/7] Gerando gráfico de desbalanceamento...")
    plot_imbalance(train_counts, total_train)
    
    print("[6/7] Gerando amostras visuais...")
    plot_sample_images()
    
    print("[7/7] Analisando propriedades de imagens...")
    df_properties = analyze_image_properties()
    plot_intensity(df_properties)
    
    # Generate report and statistics
    print("\n[FINALIZANDO]")
    report = generate_report(train_counts, test_counts, total_train, total_test)
    
    report_path = os.path.join(REPORTS_DIR, 'eda_relatorio_completo.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"[SALVO] {report_path}")
    
    stats = save_statistics(train_counts, test_counts, total_train, total_test)
    
    print("\n" + "="*80)
    print("ANÁLISE EXPLORATÓRIA CONCLUÍDA")
    print("="*80)
    print(f"\nArtifatos salvos em: {FIGURES_DIR}")
    print(f"Relatório em: {REPORTS_DIR}")
    print("\n[RESUMO]")
    print(stats.to_string(index=False))
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
