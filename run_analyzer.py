#!/usr/bin/env python
"""
Quick Start Guide - Video Quality Analyzer
===========================================

Execute este script para analisar vídeos:

    python run_analyzer.py

Etapas:
1. Coloque seus vídeos .mp4 em 'videos/'
2. Execute este script
3. Verifique resultados em 'analysis_output/'
"""

import os
import sys
from pathlib import Path

# Verificar se as pastas existem
VIDEOS_DIR = "videos"
VARIANTS_DIR = "variants"
OUTPUT_DIR = "analysis_output"

print("=" * 70)
print("VIDEO QUALITY ANALYZER - QUICK START")
print("=" * 70)

# Criar estrutura de pastas
for directory in [VIDEOS_DIR, VARIANTS_DIR, OUTPUT_DIR]:
    os.makedirs(directory, exist_ok=True)
    print(f"✓ Pasta '{directory}' criada/verificada")

print("\n" + "=" * 70)
print("INSTRUÇÕES:")
print("=" * 70)

# Verificar se há vídeos
video_files = list(Path(VIDEOS_DIR).glob("*.mp4"))
if not video_files:
    print(f"\n⚠️  NENHUM VÍDEO ENCONTRADO em '{VIDEOS_DIR}/'")
    print("\nPor favor:")
    print(f"  1. Coloque seus vídeos .mp4 em: {os.path.abspath(VIDEOS_DIR)}/")
    print(f"  2. Execute novamente este script ou execute:")
    print(f"     python video_quality_analyzer.py")
    print("\nExemplo:")
    print(f"  cp ~/Downloads/seu_video.mp4 {VIDEOS_DIR}/")
    sys.exit(0)

print(f"\n✓ Encontrados {len(video_files)} vídeo(s):")
for vid in video_files:
    print(f"  - {vid.name}")

print("\n" + "=" * 70)
print("EXECUTANDO ANÁLISE...")
print("=" * 70)

# Importar e executar
try:
    from video_quality_analyzer import analyze_videos
    
    # CONFIGURAÇÃO - edite conforme necessário
    MODEL_PATH = r"Models\Trained\yolov11_sahi_1280\Model\weights\best.pt"
    FRAME_SKIP = 10  # Analisa cada 10º frame (mais rápido)
    BLUR_THRESHOLD = 100.0
    
    print(f"\nConfiguração:")
    print(f"  - Pasta de vídeos: {VIDEOS_DIR}/")
    print(f"  - Pasta de variantes: {VARIANTS_DIR}/")
    print(f"  - Pasta de saída: {OUTPUT_DIR}/")
    print(f"  - Modelo YOLO: {MODEL_PATH}")
    print(f"  - Frame skip: {FRAME_SKIP} (cada {FRAME_SKIP}º frame)")
    print(f"  - Blur threshold: {BLUR_THRESHOLD}")
    
    print("\nExecutando análise... (isto pode levar alguns minutos)")
    print("-" * 70)
    
    results = analyze_videos(
        videos_dir=VIDEOS_DIR,
        variants_dir=VARIANTS_DIR,
        output_dir=OUTPUT_DIR,
        model_path=MODEL_PATH,
        frame_skip=FRAME_SKIP,
        blur_threshold=BLUR_THRESHOLD
    )
    
    print("-" * 70)
    print("\n✓ ANÁLISE COMPLETA!")
    print("\n" + "=" * 70)
    print("RESULTADOS:")
    print("=" * 70)
    
    # Mostrar arquivos gerados
    json_files = list(Path(OUTPUT_DIR).glob("*_analysis.json"))
    csv_files = list(Path(OUTPUT_DIR).glob("*.csv"))
    
    print(f"\n📁 Arquivos JSON (por vídeo):")
    for json_file in json_files:
        print(f"  - {json_file.name}")
    
    print(f"\n📊 Resumo CSV:")
    for csv_file in csv_files:
        print(f"  - {csv_file.name}")
    
    print(f"\n📂 Variantes geradas:")
    for variant_dir in Path(VARIANTS_DIR).iterdir():
        if variant_dir.is_dir():
            variants = list(variant_dir.glob("*.mp4"))
            print(f"  {variant_dir.name}/: {len(variants)} variante(s)")
    
    print("\n" + "=" * 70)
    print("PRÓXIMOS PASSOS:")
    print("=" * 70)
    print(f"\n1. Abra o CSV para visualizar resumo:")
    print(f"   {os.path.abspath(OUTPUT_DIR)}/analysis_summary.csv")
    
    print(f"\n2. Verifique JSON detalhado:")
    print(f"   {os.path.abspath(OUTPUT_DIR)}/<video_name>_analysis.json")
    
    print(f"\n3. Visualize vídeos downscaled:")
    print(f"   {os.path.abspath(VARIANTS_DIR)}/")
    
    print("\n" + "=" * 70)
    
except ImportError as e:
    print(f"\n❌ Erro ao importar módulo: {e}")
    print("\nVerifique se tem as dependências instaladas:")
    print("  pip install opencv-python numpy pandas ultralytics")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Erro durante execução: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
