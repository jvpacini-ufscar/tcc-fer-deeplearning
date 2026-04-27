import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import timm
import os
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Configurações
BASE_DIR = "."
DATA_DIR = os.path.join(BASE_DIR, "data/raw/raf-db")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "test")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
MODELS_DIR = os.path.join(BASE_DIR, "models")
BEST_MODEL_RAFDB = os.path.join(MODELS_DIR, "efficientnetb0_rafdb_best.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EMOTION_CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# 1. Pipeline de Extração (Sem Augmentation para os embeddings base)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def extract_embeddings(model, loader):
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Extraindo Embeddings"):
            inputs = inputs.to(DEVICE)
            # Extrair penúltima camada (global pool)
            features = model.forward_features(inputs)
            features = model.global_pool(features)
            embeddings.append(features.cpu().numpy())
            labels.append(targets.numpy())
    return np.vstack(embeddings), np.concatenate(labels)

def main():
    print(f"\n[EXP 2] SMOTE nos Embeddings da EfficientNetB0")
    
    # 1. Carregar Extrator
    model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=7)
    if os.path.exists(BEST_MODEL_RAFDB):
        model.load_state_dict(torch.load(BEST_MODEL_RAFDB, map_location=DEVICE))
        print("[INFO] Usando pesos da EfficientNet afinada no RAF-DB.")
    else:
        print("[AVISO] Pesos do RAF-DB não encontrados. Usando ImageNet baseline.")
        model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=7)
    model.to(DEVICE)

    # 2. Carregar Dados
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 3. Extração
    X_train, y_train = extract_embeddings(model, train_loader)
    X_val, y_val = extract_embeddings(model, val_loader)

    # 4. Aplicar SMOTE
    print(f"[INFO] Aplicando SMOTE. Antes: {np.bincount(y_train)}")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print(f"[INFO] Depois do SMOTE: {np.bincount(y_train_smote)}")

    # 5. Treinar Classificador MLP sobre Embeddings Sintéticos
    X_train_t = torch.tensor(X_train_smote, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_smote, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    smote_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=128, shuffle=True)
    
    classifier = nn.Sequential(
        nn.Linear(1280, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 7)
    ).to(DEVICE)

    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print("[INFO] Treinando classificador MLP...")
    for epoch in range(50):
        classifier.train()
        for batch_x, batch_y in smote_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            outputs = classifier(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    # 6. Avaliação
    classifier.eval()
    with torch.no_grad():
        outputs = classifier(X_val_t.to(DEVICE))
        _, y_pred = outputs.max(1)
        y_pred = y_pred.cpu().numpy()

    report = classification_report(y_val, y_pred, target_names=EMOTION_CLASSES, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(os.path.join(REPORTS_DIR, "metrics_efficientnet_smote.csv"))
    
    cm = confusion_matrix(y_val, y_pred, normalize='true')
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Purples', xticklabels=EMOTION_CLASSES, yticklabels=EMOTION_CLASSES)
    plt.title('Matriz de Confusão: EfficientNet + SMOTE (Embeddings)')
    plt.savefig(os.path.join(FIGURES_DIR, "cm_efficientnet_smote.png"))
    plt.close()
    print("[FINALIZADO EXP 2] SMOTE concluído.")

if __name__ == "__main__":
    main()
