# Trained Models 📦

Esta pasta contém os modelos treinados para o projeto de análise de futebol.

## ⚠️ Modelos Não Inclusos

## 📥 Como Adicionar os Modelos

### Opção 1: Pasta `yolov11` (Principal)
```
Models/Trained/yolov11/
    └── weights/
        └── best.pt    ← Coloca aqui o arquivo
```

### Opção 2: Pasta `yolov11_keypoints_` (Alternativo)
```
Models/Trained/yolov11_keypoints_/
    └── weights/
        └── best.pt    ← Coloca aqui o arquivo
```

## 🔧 Alterar Qual Modelo Usar

Edita `constants.py` e descomenta a linha do modelo que quiseres:

```python
# Modelo principal 
model_path = PROJECT_DIR / "Models" / "Trained" / "yolov11" / "weights" / "best.pt"

# OU modelo alternativo (Keypoints)
# model_path = PROJECT_DIR / "Models" / "Trained" / "yolov11_keypoints_" / "weights" / "best.pt"
```

## 💾 Onde Obter os Modelos

- Se treinou localmente: procura em `Models/Trained/` no teu computador
- Se foram fornecidos: pede aos colegas
- Se precisas treinar: vê `keypoint_detection/training/` ou `player_detection/training/`

---

