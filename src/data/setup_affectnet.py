import kagglehub
import os
import shutil

# Configurações
DATA_DIR = "./data/raw/affectnet-kaggle"

def download_affectnet():
    print("Iniciando download do AffectNet (Subset) via KaggleHub...")
    
    # Baixa o dataset
    path = kagglehub.dataset_download("mstjebashazida/affectnet")
    
    print(f"Dataset baixado para: {path}")
    
    # Criar diretório de destino se não existir
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    # Mover os arquivos para a estrutura do projeto
    # O KaggleHub baixa para um cache, vamos copiar para ./data/raw/
    print(f"Movendo arquivos para {DATA_DIR}...")
    
    # Se o dataset já estiver lá, removemos para evitar lixo
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)
    
    shutil.copytree(path, DATA_DIR)
    print("Download e organização concluídos!")

if __name__ == "__main__":
    download_affectnet()
