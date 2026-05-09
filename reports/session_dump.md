# Session Dump - TCC Facial Expression Recognition
**Última Atualização:** 2026-05-08 10:15

## 1. Estado Atual
- **Dataset RAF-DB:** Concluída bateria de testes. ConvNeXt Tiny detém o recorde com **86.54%**.
- **Dataset FER2013:** ConvNeXt Tiny atingiu **65.06%** de acurácia (Novo Recorde para este dataset no projeto).
- **Dataset AffectNet:** Download concluído. Validada a estrutura em `data/raw/affectnet-kaggle/archive (3)/Train` e `Test`. Está no formato ImageFolder.

## 2. Resultados Recentes
- **ConvNeXt Tiny no FER2013:** Atingiu **65.06%** de acurácia (Recuperado em 08/05/2026).
- **ConvNeXt Tiny no RAF-DB:** Atingiu **86.54%**.

## 3. Credenciais
- Kaggle configurado em `~/.kaggle/kaggle.json`.

## 4. Próximos Passos Imediatos
1. [ ] Treinar ConvNeXt Tiny no AffectNet para consolidar o backbone estático.
2. [ ] Iniciar extração de features para a fase Temporal (LSTM/Attention).
3. [ ] Avaliação cross-dataset final (Train: AffectNet -> Test: RAF-DB/FER2013).

## 5. Pendências
- Organizar ou criar link simbólico para o AffectNet para facilitar o pathing (o caminho atual com 'archive (3)' é ruim).
