import os
import shutil
import pandas as pd
import kagglehub
from tqdm import tqdm

# Configurações do Projeto
BASE_DIR = "/mnt/c/Users/jvpac/Videos/tcc-fer-deeplearning"
TARGET_DIR = os.path.join(BASE_DIR, "data/raw/raf-db")
KAGGLE_DATASET = "shuvoalok/raf-db-dataset"

# Mapeamento RAF-DB (1-7) para o nosso padrão FER2013 (Pastas alfabéticas)
# RAF-DB: 1:Surprise, 2:Fear, 3:Disgust, 4:Happiness, 5:Sadness, 6:Anger, 7:Neutral
# Nosso Padrão (ImageFolder): 
# 0:angry, 1:disgust, 2:fear, 3:happy, 4:neutral, 5:sad, 6:surprise
RAF_TO_FER = {
    1: "surprise",
    2: "fear",
    3: "disgust",
    4: "happy",
    5: "sad",
    6: "angry",
    7: "neutral"
}

def setup_rafdb(username, key):
    # 1. Configura Credenciais
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key

    print(f"[INFO] Iniciando download do dataset {KAGGLE_DATASET} do Kaggle...")
    try:
        download_path = kagglehub.dataset_download(KAGGLE_DATASET)
        print(f"[INFO] Dataset baixado em: {download_path}")
    except Exception as e:
        print(f"[ERRO] Falha no download: {e}")
        return

    # 2. Estrutura de Pastas
    for split in ["train", "test"]:
        for emotion in RAF_TO_FER.values():
            os.makedirs(os.path.join(TARGET_DIR, split, emotion), exist_ok=True)

    # 3. Mapear e Copiar
    # O dataset do Kaggle (shuvoalok/raf-db-dataset) j vem organizado em 1, 2, 3...
    dataset_root = os.path.join(download_path, "DATASET")
    
    for split in ["train", "test"]:
        src_split_dir = os.path.join(dataset_root, split)
        if not os.path.exists(src_split_dir):
            print(f"[AVISO] Pasta {src_split_dir} no encontrada. Pulando...")
            continue
            
        for raf_id, emotion_name in RAF_TO_FER.items():
            src_emotion_dir = os.path.join(src_split_dir, str(raf_id))
            dst_emotion_dir = os.path.join(TARGET_DIR, split, emotion_name)
            
            if os.path.exists(src_emotion_dir):
                print(f"[INFO] Copiando {split}/{emotion_name}...")
                # Copia o contedo da pasta
                for img_name in os.listdir(src_emotion_dir):
                    shutil.copy2(
                        os.path.join(src_emotion_dir, img_name),
                        os.path.join(dst_emotion_dir, img_name)
                    )
            else:
                print(f"[AVISO] Pasta de origem {src_emotion_dir} no existe.")

    print(f"[SUCESSO] RAF-DB organizado em {TARGET_DIR}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Uso: python setup_rafdb.py <username> <key>")
    else:
        setup_rafdb(sys.argv[1], sys.argv[2])
