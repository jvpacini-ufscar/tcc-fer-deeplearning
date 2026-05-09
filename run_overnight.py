import subprocess
import os
import time
from datetime import datetime

def log_event(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {message}\n"
    print(log_line)
    with open("reports/overnight_sprint.log", "a") as f:
        f.write(log_line)

def run_script(script_path, desc):
    log_event(f"Iniciando: {desc} ({script_path})")
    try:
        # Usando o venv do projeto
        cmd = ["./venv/Scripts/python.exe", script_path]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        
        # Salva output em log em tempo real
        with open(f"reports/logs_{os.path.basename(script_path)}.log", "w") as f:
            for line in process.stdout:
                print(line, end="")
                f.write(line)
        
        process.wait()
        if process.returncode == 0:
            log_event(f"Sucesso: {desc}")
            return True
        else:
            log_event(f"Erro: {desc} terminou com código {process.returncode}")
            return False
    except Exception as e:
        log_event(f"Falha Crítica ao rodar {desc}: {str(e)}")
        return False

def main():
    log_event("=== INÍCIO DA SPRINT NOTURNA - TCC FER ===")
    
    # Task 1: Treinar no AffectNet (A parte mais pesada)
    success = run_script("src/train/train_convnext_affectnet.py", "Treinamento ConvNeXt no AffectNet")
    
    if success:
        # Task 2: Avaliação Cross-Dataset (Opcional, se o modelo existir)
        # Vou criar esse script de avaliação rápida logo em seguida
        run_script("src/evaluate/eval_cross_dataset.py", "Avaliação Cross-Dataset (AffectNet -> Others)")
    
    # Task 3: Atualizar Session Dump
    log_event("Sprint Finalizada. Atualizando status final...")
    with open("reports/session_dump.md", "a") as f:
        f.write(f"\n- **Sprint Noturna ({datetime.now().strftime('%d/%m')})**: Finalizada com sucesso.\n")

    log_event("=== SPRINT NOTURNA CONCLUÍDA ===")

if __name__ == "__main__":
    main()
