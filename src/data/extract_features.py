import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
import os
import numpy as np
from tqdm import tqdm

# Configurações
BASE_DIR = "."
CK_DIR = os.path.join(BASE_DIR, "data/raw/ckplus")
MODEL_PATH = os.path.join(BASE_DIR, "models/convnext_affectnet_best.pth")
FEATURES_DIR = os.path.join(BASE_DIR, "data/processed/features")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract():
    print("--- Iniciando Extração de Features (Backbone: ConvNeXt) ---")
    os.makedirs(FEATURES_DIR, exist_ok=True)

    # 1. Carregar Modelo sem a cabeça de classificação (Extrator)
    model = timm.create_model('convnext_tiny', pretrained=False, num_classes=8)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    
    # Removemos apenas a camada linear final, mantendo o pooling
    model.head.fc = nn.Identity() 
    model.to(DEVICE)
    model.eval()

    # 2. Pipeline de Dados
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(CK_DIR, transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False) # Batch 1 para manter ordem das sequências

    features = []
    labels = []
    filenames = []

    with torch.no_grad():
        for i, (inputs, target) in enumerate(tqdm(loader, desc="Processando Frames")):
            inputs = inputs.to(DEVICE)
            feat = model(inputs)
            
            features.append(feat.cpu().numpy().flatten())
            labels.append(target.item())
            filenames.append(dataset.samples[i][0])

    # 3. Salvar
    np.save(os.path.join(FEATURES_DIR, "ckplus_features.npy"), np.array(features))
    np.save(os.path.join(FEATURES_DIR, "ckplus_labels.npy"), np.array(labels))
    
    # Salvar filenames para reconstruir as sequências depois
    with open(os.path.join(FEATURES_DIR, "ckplus_filenames.txt"), "w") as f:
        for fname in filenames:
            f.write(f"{fname}\n")

    print(f"Extração concluída! {len(features)} frames processados.")

if __name__ == "__main__":
    extract()
