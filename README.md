# Reconhecimento de Emoções Humanas usando IA

Este projeto é um Trabalho de Conclusão de Curso desenvolvido no Departamento de Computação da UFSCar[cite: 1, 2].
**Orientador:** Prof. Alexandre Luís Magalhães Levada [cite: 12]

## Status Atual: Baseline Concluído

### 1. Análise Exploratória (EDA)
Identificamos um forte desbalanceamento no dataset FER2013, o que corrobora a literatura da área[cite: 85]:
* **Classe Majoritária:** Happy (~25% do dataset).
* **Classe Minoritária:** Disgust (<2% do dataset).

![Distribuição de Classes](distribuicao_classes_fer2013.png)

### 2. Modelo Baseline (CNN Simples)
Implementamos uma arquitetura convolucional básica para estabelecer um ponto de partida.
* **Arquitetura:** 2 camadas de Conv2D + MaxPooling, seguidas de uma camada densa de 64 neurônios.
* **Resultados (5 épocas):**
    * Acurácia de Treino: 55%
    * Acurácia de Validação: 50%

Embora a literatura aponte que modelos estado-da-arte no FER2013 atinjam entre 65-73%[cite: 85], nosso baseline de 50% é um excelente começo, superando significativamente o chute aleatório (14%).