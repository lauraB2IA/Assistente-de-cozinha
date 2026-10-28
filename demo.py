"""Demo interativo do Assistente de Cozinha com interface Gradio.

Uso:
    python demo.py

Abre uma página no navegador com:
  - Gravação de voz pelo microfone
  - Gráfico de barras mostrando a confiança em cada classe
  - Histórico das últimas predições
"""
from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import librosa
import numpy as np
import tensorflow as tf

# ── Constantes (devem ser iguais às do treinar_modelo.py) ─────────────────────
SR          = 16_000
DURACAO     = 2.0
N_MFCC      = 40
N_FFT       = 512
HOP_LENGTH  = 256
PASTA_MODELO = Path("modelo")

# ── Emojis para cada classe ───────────────────────────────────────────────────
EMOJIS = {
    "iniciar_receita":    "🍳",
    "proximo_passo":      "⏭️",
    "repetir_instrucao":  "🔁",
    "finalizar_receita":  "✅",
    "desconhecido":       "❓",
}

# ── Carregamento do modelo ────────────────────────────────────────────────────
print("Carregando modelo...")
modelo      = tf.keras.models.load_model(PASTA_MODELO / "comandos.keras")
classes     = json.loads((PASTA_MODELO / "rotulos.json").read_text(encoding="utf-8"))
norm        = np.load(PASTA_MODELO / "normalizacao.npz")
# Garante shape (1,1,1,3) para broadcasting correto com x de shape (40, T, 3)
MEDIA       = norm["media"].reshape(1, 1, 1, -1)   # (1,1,1,3)
DESVIO      = norm["desvio"].reshape(1, 1, 1, -1)  # (1,1,1,3)
print(f"Modelo carregado. Classes: {classes}")


# ── Extração de características ───────────────────────────────────────────────
def extrair(audio: np.ndarray) -> np.ndarray:
    tamanho = int(SR * DURACAO)
    audio   = np.pad(audio, (0, max(0, tamanho - len(audio))))[:tamanho]
    mfcc    = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=N_MFCC,
                                   n_fft=N_FFT, hop_length=HOP_LENGTH)
    delta   = librosa.feature.delta(mfcc)
    delta2  = librosa.feature.delta(mfcc, order=2)
    return np.stack([mfcc, delta, delta2], axis=-1).astype(np.float32)


# ── Predição principal ────────────────────────────────────────────────────────
def prever(audio_tuple):
    """Recebe (sample_rate, array) do Gradio e retorna barras de confiança + texto."""
    if audio_tuple is None:
        return {c: 0.0 for c in classes}, "⏸️ Grave um áudio para começar."

    sr_entrada, dados = audio_tuple

    # Converte para mono float32 normalizado
    audio = dados.astype(np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if audio.max() > 1.0:
        audio /= 32768.0  # int16 → float32

    # Reamostra se necessário
    if sr_entrada != SR:
        audio = librosa.resample(audio, orig_sr=sr_entrada, target_sr=SR)

    # Features + normalização por canal (broadcast automático)
    x = extrair(audio)                          # shape (40, T, 3)
    x = (x - MEDIA[0]) / (DESVIO[0] + 1e-8)   # MEDIA[0] shape (1,1,3)

    # Inferência
    probs = modelo.predict(x[None, ...], verbose=0)[0]
    idx   = int(np.argmax(probs))
    classe_prevista = classes[idx]
    confianca = float(probs[idx])

    # Threshold: se o modelo não tiver certeza, diz "desconhecido"
    LIMIAR = 0.55
    if confianca < LIMIAR:
        classe_prevista = "desconhecido"
        idx = classes.index("desconhecido") if "desconhecido" in classes else -1
        confianca_aviso = True
    else:
        confianca_aviso = False

    # Monta dict para o Label do Gradio (chave → probabilidade 0-1)
    resultado = {c: float(p) for c, p in zip(classes, probs)}

    emoji = EMOJIS.get(classe_prevista, "🎙️")
    if confianca_aviso:
        nivel = f"Confiança baixa ({confianca:.0%}) — classificado como desconhecido ⚠️"
    elif confianca >= 0.80:
        nivel = "Alta confiança ✅"
    elif confianca >= 0.55:
        nivel = "Confiança média ⚠️"
    else:
        nivel = "Baixa confiança ❌"

    texto = (
        f"## {emoji} {classe_prevista.replace('_', ' ').title()}\n\n"
        f"**Confiança:** {confianca:.1%} — {nivel}\n\n"
        f"Fale com clareza a até ~30 cm do microfone para melhores resultados."
    )
    return resultado, texto


# ── Interface Gradio ──────────────────────────────────────────────────────────
css = """
body { font-family: 'Inter', sans-serif; }
.gradio-container { max-width: 860px !important; margin: 0 auto; }
#title { text-align: center; margin-bottom: 8px; }
#subtitle { text-align: center; color: #888; margin-bottom: 24px; }
.label-wrap { border-radius: 12px !important; }
"""

with gr.Blocks(title="Assistente de Cozinha · Demo") as demo:

    gr.HTML("<h1 id='title'>🎙️ Assistente de Cozinha — Demo</h1>")
    gr.HTML("<p id='subtitle'>Grave um comando de voz e veja o modelo classificar em tempo real.</p>")

    with gr.Row():
        with gr.Column(scale=1):
            audio_entrada = gr.Audio(
                sources=["microphone"],
                type="numpy",
                label="🎤 Grave seu comando",
            )
            btn = gr.Button("🔍 Classificar", variant="primary", size="lg")

            gr.Markdown("""
**Comandos disponíveis:**
- 🍳 `iniciar receita`
- ⏭️ `próximo passo`
- 🔁 `repetir instrução`
- ✅ `finalizar receita`
- ❓ `desconhecido` (qualquer outra fala)
            """)

        with gr.Column(scale=1):
            saida_label = gr.Label(
                label="📊 Confiança por classe",
                num_top_classes=5,
            )
            saida_texto = gr.Markdown(
                value="*Grave um áudio para ver a predição.*"
            )

    btn.click(fn=prever, inputs=audio_entrada, outputs=[saida_label, saida_texto])
    audio_entrada.change(fn=prever, inputs=audio_entrada, outputs=[saida_label, saida_texto])

    gr.HTML("""
    <hr style='margin:32px 0; opacity:.2'/>
    <div style='text-align:center; color:#666; font-size:.85rem'>
      Projeto AC2 · CNN treinada com TensorFlow/Keras · Features: MFCC + Δ + ΔΔ · Acurácia no teste: 93,55%
    </div>
    """)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
        theme=gr.themes.Base(
            primary_hue="violet",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        ),
        css=css,
    )
