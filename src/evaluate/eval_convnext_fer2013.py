import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# Configurações
BASE_DIR = "."
DATA_DIR = os.path.join(BASE_DIR, "data/raw/fer2013")
VAL_DIR = os.path.join(DATA_DIR, "test")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_CLASSES = 7
EMOTION_CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def evaluate():
    print(f"--- Iniciando Avaliação do ConvNeXt Tiny no FER2013 ---")
    print(f"Device: {DEVICE}")

    # 1. Pipeline de Dados (Mesmo do treino)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_dataset = datasets.ImageFolder(VAL_DIR, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 2. Carregar Modelo
    model = timm.create_model('convnext_tiny', pretrained=False, num_classes=NUM_CLASSES)
    model_path = os.path.join(MODELS_DIR, "convnext_fer2013_best.pth")
    
    if not os.path.exists(model_path):
        print(f"Erro: Modelo não encontrado em {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # 3. Predições
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Avaliando"):
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.cpu().tolist())

    # 4. Gerar Relatórios
    print("\n--- Resultados ---")
    acc = (torch.tensor(y_true) == torch.tensor(y_pred)).float().mean().item()
    print(f"Acurácia Final: {acc:.4f}")

    report = classification_report(y_true, y_pred, target_names=EMOTION_CLASSES, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    
    report_path = os.path.join(REPORTS_DIR, "metrics_convnext_fer2013.csv")
    report_df.to_csv(report_path)
    print(f"Métricas salvas em: {report_path}")

    # 5. Matriz de Confusão
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Purples', xticklabels=EMOTION_CLASSES, yticklabels=EMOTION_CLASSES)
    plt.title(f'Matriz de Confusão ConvNeXt Tiny - FER2013 (Acc: {acc:.4f})')
    plt.ylabel('Real')
    plt.xlabel('Predito')
    
    os.makedirs(FIGURES_DIR, exist_ok=True)
    cm_path = os.path.join(FIGURES_DIR, "cm_convnext_fer2013.png")
    plt.savefig(cm_path)
    print(f"Matriz de confusão salva em: {cm_path}")

if __name__ == "__main__":
    evaluate()
