import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# Configurações de Caminhos
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data/raw")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Definição das classes por dataset
CLASSES_FER = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
CLASSES_RAF = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
CLASSES_CK  = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
# AffectNet (internamente no modelo): 0:anger, 1:contempt, 2:disgust, 3:fear, 4:happy, 5:neutral, 6:sad, 7:surprise

# Mapeamentos de Predição (Saída do Modelo -> Índice do Dataset)
def map_preds(preds, model_type, target_dataset):
    """
    Normaliza a saída do modelo para o índice da classe no dataset alvo.
    """
    # Se o modelo foi treinado no FER ou RAF, os índices (0-6) já batem com a maioria.
    # A exceção é o CK+ e o modelo AffectNet.
    
    mapped = []
    for p in preds:
        if model_type == 'affectnet': # 8 classes
            # Mapeamento para o padrão de 7 classes (removendo contempt)
            mapping = {0:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6}
            mapped.append(mapping.get(p, 4)) # default neutral if contempt
        elif model_type == 'ckplus': # 8 classes (incluindo neutral hack)
            # 0:anger, 1:contempt, 2:disgust, 3:fear, 4:happy, 5:sadness, 6:surprise, 7:neutral
            mapping = {0:0, 2:1, 3:2, 4:3, 7:4, 5:5, 6:6}
            mapped.append(mapping.get(p, 4))
        else: # fer2013 ou raf-db (7 classes padrão)
            mapped.append(p)
    return mapped

from torchvision import models

def get_model(model_name, model_path):
    state_dict = torch.load(model_path, map_location=DEVICE)
    
    if 'convnext' in model_name:
        num_classes = 8 if 'affectnet' in model_name or 'ckplus' in model_name else 7
        model = timm.create_model('convnext_tiny', pretrained=False, num_classes=num_classes)
        model.load_state_dict(state_dict)
    elif 'resnet' in model_name:
        # Detectar se é Sequential (tem "0.stem" ou "0.conv1") ou Timm padrão
        if any(k.startswith('0.') for k in state_dict.keys()):
            base_model = timm.create_model('resnetv2_50', pretrained=False, num_classes=0)
            model = nn.Sequential(
                base_model,
                nn.Flatten(),
                nn.Linear(2048, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(p=0.4),
                nn.Linear(512, 7)
            )
        else:
            model = timm.create_model('resnet50', pretrained=False, num_classes=7)
        model.load_state_dict(state_dict)
    elif 'efficientnet' in model_name:
        if 'features.0.0.weight' in state_dict: # Torchvision
            model = models.efficientnet_b0()
            num_ftrs = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.3, inplace=False),
                nn.Linear(num_ftrs, 128),
                nn.ReLU(),
                nn.Dropout(p=0.3, inplace=False),
                nn.Linear(128, 7)
            )
        else: # Timm
            model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=7)
        model.load_state_dict(state_dict)
    else:
        return None
    model.to(DEVICE)
    model.eval()
    return model

def evaluate_model_on_dataset(model, model_type, dataset_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    try:
        dataset = datasets.ImageFolder(dataset_path, transform=transform)
    except:
        return 0.0
        
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            
            # Normalizar predições
            normalized_preds = map_preds(preds.cpu().tolist(), model_type, None)
            all_preds.extend(normalized_preds)
            all_labels.extend(labels.tolist())
            
    return accuracy_score(all_labels, all_preds)

def main():
    # Lista de Modelos e seus pesos
    models_to_test = {
        'ConvNeXt (AffectNet)': ('affectnet', os.path.join(MODELS_DIR, 'convnext_affectnet_best.pth')),
        'ConvNeXt (RAF-DB)': ('raf-db', os.path.join(MODELS_DIR, 'convnext_rafdb_best.pth')),
        'ResNet50V2 (FER2013)': ('fer2013', os.path.join(MODELS_DIR, 'resnet50v2_pytorch_best.pth')),
        'EfficientNetB0 (FER2013)': ('fer2013', os.path.join(MODELS_DIR, 'efficientnetb0_pytorch_best.pth')),
        'ResNet50V2 (RAF-DB)': ('raf-db', os.path.join(MODELS_DIR, 'resnet50v2_rafdb_best.pth')),
        'EfficientNetB0 (RAF-DB)': ('raf-db', os.path.join(MODELS_DIR, 'efficientnetb0_rafdb_best.pth')),
    }
    
    datasets_to_test = {
        'FER2013': os.path.join(DATA_DIR, 'fer2013/test'),
        'RAF-DB': os.path.join(DATA_DIR, 'raf-db/test'),
        'CK+ (Peak)': os.path.join(DATA_DIR, 'ckplus')
    }
    
    results = []
    
    for m_name, (m_type, m_path) in models_to_test.items():
        if not os.path.exists(m_path):
            print(f"Aviso: Pesos para {m_name} não encontrados em {m_path}")
            continue
            
        print(f"Avaliando {m_name}...")
        model = get_model(m_name.lower(), m_path)
        
        row = {'Modelo': m_name}
        for d_name, d_path in datasets_to_test.items():
            acc = evaluate_model_on_dataset(model, m_type, d_path)
            row[d_name] = f"{acc*100:.2f}%"
            print(f"  - {d_name}: {acc*100:.2f}%")
        
        results.append(row)
        
    df = pd.DataFrame(results)
    output_path = os.path.join(REPORTS_DIR, "tabela_comparativa_final.csv")
    df.to_csv(output_path, index=False)
    print(f"\nMatriz de comparação salva em: {output_path}")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
