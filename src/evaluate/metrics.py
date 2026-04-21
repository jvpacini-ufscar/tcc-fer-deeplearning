import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def save_classification_report(y_true, y_pred, classes, model_name, output_dir="../../reports"):
    """
    Gera e salva o classification report (Precision, Recall, F1-Score) em formato CSV.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Gera o dicionário do report
    report_dict = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    
    # Converte para DataFrame
    df_report = pd.DataFrame(report_dict).transpose()
    
    # Salva em CSV
    csv_path = os.path.join(output_dir, f"metrics_{model_name}.csv")
    df_report.to_csv(csv_path, index=True)
    print(f"[INFO] Métricas salvas em: {csv_path}")
    
    return df_report

def plot_and_save_confusion_matrix(y_true, y_pred, classes, model_name, output_dir="../../reports/figures"):
    """
    Gera, exibe e salva a Matriz de Confusão normalizada.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Calcula a matriz normalizada
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Matriz de Confusão Normalizada - {model_name}')
    plt.ylabel('Verdadeiro')
    plt.xlabel('Predito')
    
    # Salva a figura
    fig_path = os.path.join(output_dir, f"cm_{model_name}.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Matriz de confusão salva em: {fig_path}")

def evaluate_model(model, x_test, y_test, classes, model_name):
    """
    Pipeline completo de avaliação de um modelo Keras/TensorFlow.
    """
    print(f"[INFO] Avaliando modelo: {model_name}...")
    
    # Predições do modelo
    y_pred_prob = model.predict(x_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    # Se y_test for one-hot encoded, converte para rótulos numéricos
    if len(y_test.shape) > 1 and y_test.shape[1] > 1:
        y_true = np.argmax(y_test, axis=1)
    else:
        y_true = y_test
        
    # Salva as métricas rigorosas
    df_metrics = save_classification_report(y_true, y_pred, classes, model_name)
    
    # Salva a matriz de confusão
    plot_and_save_confusion_matrix(y_true, y_pred, classes, model_name)
    
    return df_metrics
