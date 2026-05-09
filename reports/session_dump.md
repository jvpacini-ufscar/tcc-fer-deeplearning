# Session Dump - TCC Facial Expression Recognition
**Última Atualização:** 2026-05-08 23:15

## 1. Estado Atual
- **Dataset RAF-DB:** Recorde de **86.54%** com ConvNeXt Tiny.
- **Dataset FER2013:** Recorde de **65.06%** com ConvNeXt Tiny.
- **Dataset AffectNet (Subset):** Treinado com **70.32%** de acurácia. Usado como backbone para extração de features.
- **Fase Temporal (CK+):** Primeira pipeline completa concluída. Extração de features via ConvNeXt + Classificação via LSTM.

## 2. Resultados Recentes
- **LSTM no CK+:** Atingiu **95.45%** de acurácia em sequências temporais.
- **Cross-Dataset (AffectNet -> Others):** FER2013 (42.84%), RAF-DB (33.70%).

## 3. Credenciais
- Kaggle configurado e funcional.

## 4. Próximos Passos Imediatos
1. [ ] Implementar Attention Mechanism na LSTM para melhorar foco em frames-chave.
2. [ ] Testar a pipeline em um vídeo "in-the-wild" (ex: YouTube ou Webcam).
3. [ ] Iniciar a redação do capítulo de Metodologia e Resultados.

## 5. Pendências
- Avaliar se o desbalanceamento do CK+ impactou o F1-Score da LSTM (apesar da alta acurácia).

- **Sprint Noturna (08/05)**: Finalizada com sucesso.

- **Sprint Noturna (08/05)**: Finalizada com sucesso.
