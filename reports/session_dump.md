# Session Dump - TCC Facial Expression Recognition
**Última Atualização:** 2026-05-08 23:50

## 1. Estado Atual
- **Backbone (ConvNeXt Tiny):** Consolidado como o melhor extrator de features. Treinado no AffectNet (70.3% acc) e validado no FER2013 (65.06%) e RAF-DB (86.54%).
- **Fase Temporal (CK+):** Implementada e validada cientificamente.
- **Arquitetura Atual:** ConvNeXt (Backbone) + Bidirectional LSTM + Self-Attention Mechanism.

## 2. Resultados e Conquistas (Sessão de Hoje)
- **AffectNet (Subset):** Modelo treinado para servir como extrator de características genéricas.
- **Cross-Dataset:** Avaliação do backbone AffectNet em outros domínios concluída (Baseline para discussão de generalização).
- **Extração de Features:** Pipeline automatizada para converter sequências de vídeo em embeddings.
- **Validação Científica (Rigorosa):** Realizado **5-Fold Cross-Validation (Subject-Independent)** no CK+.
  - **Média de Acurácia:** **95.38% (+/- 2.36%)** 🚀
  - **Status:** Modelo temporal validado contra vazamento de identidade (Leakage).

## 3. Arquivos Chave (Checkpoint)
- `models/convnext_affectnet_best.pth`: Pesos do backbone.
- `models/fer_attention_lstm_ckplus.pth`: Pesos do modelo temporal.
- `reports/metrics_lstm_ckplus_rigorous.csv`: Relatório detalhado por classe.
- `reports/figures/cm_lstm_ckplus_rigorous.png`: Matriz de confusão para a monografia.

## 4. Próximos Passos (Retomada)
1. [ ] **Análise Temporal Visual:** Gerar gráficos de evolução de probabilidade (curva de confiança) em sequências do CK+.
2. [ ] **Inferência em Vídeos Externos:** Testar o modelo em clipes do YouTube/Webcam (Qualitativo).
3. [ ] **Redação:** Iniciar capítulo de Metodologia descrevendo o processo de extração de features e a LSTM com Atenção.

## 5. Pendências
- Nenhuma pendência técnica imediata. O ambiente está estável e o código modularizado em `src/`.

- **Sprint Noturna (08/05)**: Finalizada com sucesso.

- **Sprint Noturna (08/05)**: Finalizada com sucesso.
