import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
import os
import numpy as np
from tqdm import tqdm

# Configurações
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAF_DIR = os.path.join(BASE_DIR, "data/raw/raf-db/test")
MODEL_PATH = os.path.join(BASE_DIR, "models/convnext_affectnet_best.pth")
FEATURES_DIR = os.path.join(BASE_DIR, "data/processed/features")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract():
    print("--- Iniciando Extração de Features RAF-DB (Backbone: ConvNeXt) ---")
    os.makedirs(FEATURES_DIR, exist_ok=True)

    if not os.path.exists(RAF_DIR):
        print(f"Erro: Diretório {RAF_DIR} não encontrado.")
        return

    # 1. Carregar Modelo sem a cabeça de classificação (Extrator)
    model = timm.create_model('convnext_tiny', pretrained=False, num_classes=8)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    
    model.head.fc = nn.Identity() 
    model.to(DEVICE)
    model.eval()

    # 2. Pipeline de Dados
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(RAF_DIR, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    features = []
    labels = []
    filenames = []

    with torch.no_grad():
        for i, (inputs, target) in enumerate(tqdm(loader, desc="Processando Frames")):
            inputs = inputs.to(DEVICE)
            feat = model(inputs)
            
            features.append(feat.cpu().numpy())
            labels.extend(target.tolist())
            
    # Concatenar features
    features = np.concatenate(features, axis=0)

    # 3. Salvar
    np.save(os.path.join(FEATURES_DIR, "rafdb_test_features.npy"), features)
    np.save(os.path.join(FEATURES_DIR, "rafdb_test_labels.npy"), np.array(labels))
    
    print(f"Extração concluída! {len(features)} imagens processadas.")

if __name__ == "__main__":
    extract()
