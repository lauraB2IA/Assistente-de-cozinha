"""Reconhecimento de um comando de cozinha pelo microfone."""
from __future__ import annotations

import json
from pathlib import Path

import librosa
import numpy as np
import sounddevice as sd
import tensorflow as tf

SR = 16_000
DURACAO = 2.0
N_MFCC = 40
N_FFT = 512
HOP_LENGTH = 256
MODELO = Path("modelo/comandos.keras")


def caracteristicas(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    tamanho = int(SR * DURACAO)
    audio = np.pad(audio, (0, max(0, tamanho - len(audio))))[:tamanho]
    mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=N_MFCC,
                                n_fft=N_FFT, hop_length=HOP_LENGTH)
    return np.stack([mfcc, librosa.feature.delta(mfcc),
                     librosa.feature.delta(mfcc, order=2)], axis=-1).astype(np.float32)


def main() -> None:
    if not MODELO.exists():
        raise FileNotFoundError("Modelo não encontrado. Execute treinar_modelo.py primeiro.")
    modelo = tf.keras.models.load_model(MODELO)
    classes = json.loads(Path("modelo/rotulos.json").read_text(encoding="utf-8"))
    normalizacao = np.load("modelo/normalizacao.npz")
    print("Fale um comando após o aviso. Pressione Ctrl+C para sair.")
    while True:
        input("Pressione Enter para gravar...")
        print("Ouvindo...")
        audio = sd.rec(int(SR * DURACAO), samplerate=SR, channels=1, dtype="float32")
        sd.wait()
        x = caracteristicas(audio)
        x = (x - normalizacao["media"]) / (normalizacao["desvio"] + 1e-8)
        probabilidades = modelo.predict(x[None, ...], verbose=0)[0]
        indice = int(np.argmax(probabilidades))
        print(f"Comando: {classes[indice].replace('_', ' ')} | confiança: {probabilidades[indice]:.1%}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPrograma encerrado.")
