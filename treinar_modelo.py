"""Treina uma CNN para classificar comandos de um assistente de cozinha.

Características: MFCC + delta + delta-delta extraídos com Librosa.
Saídas: modelo/comandos.keras, modelo/rotulos.json e resultados/.

Melhorias aplicadas:
- Data Augmentation: ruído gaussiano, time stretch, pitch shift,
  time shift e variação de volume. Cada áudio original gera N_AUMENTOS
  cópias sintéticas, multiplicando o dataset sem nova coleta.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# ── Hiperparâmetros ────────────────────────────────────────────────────────────
SEMENTE = 42
SR = 16_000
DURACAO = 2.0
N_MFCC = 40
N_FFT = 512
HOP_LENGTH = 256

# Quantos áudios aumentados gerar por arquivo original.
# 5 → o dataset fica ~6× maior sem gravar nada novo.
N_AUMENTOS = 5

PASTA_DADOS = Path("dados")
PASTA_MODELO = Path("modelo")
PASTA_RESULTADOS = Path("resultados")


# ── Data Augmentation ─────────────────────────────────────────────────────────

def adicionar_ruido(audio: np.ndarray, fator: float = 0.005) -> np.ndarray:
    """Adiciona ruído gaussiano branco ao áudio."""
    ruido = np.random.randn(len(audio)).astype(np.float32)
    return audio + fator * ruido


def time_stretch(audio: np.ndarray) -> np.ndarray:
    """Altera a velocidade do áudio entre 0.85× e 1.15×."""
    taxa = np.random.uniform(0.85, 1.15)
    return librosa.effects.time_stretch(audio, rate=taxa)


def pitch_shift(audio: np.ndarray) -> np.ndarray:
    """Desloca o pitch em até ±3 semitons."""
    semitons = np.random.uniform(-3, 3)
    return librosa.effects.pitch_shift(audio, sr=SR, n_steps=semitons)


def time_shift(audio: np.ndarray) -> np.ndarray:
    """Desloca o áudio no tempo em até ±20% da duração."""
    max_desl = int(0.20 * len(audio))
    desl = np.random.randint(-max_desl, max_desl)
    return np.roll(audio, desl)


def variar_volume(audio: np.ndarray) -> np.ndarray:
    """Multiplica o volume por um fator entre 0.7 e 1.3."""
    fator = np.random.uniform(0.70, 1.30)
    return audio * fator


# Lista de técnicas de augmentation disponíveis.
TECNICAS = [adicionar_ruido, time_stretch, pitch_shift, time_shift, variar_volume]


def aumentar(audio: np.ndarray) -> np.ndarray:
    """Aplica 1 ou 2 técnicas aleatórias ao áudio e devolve resultado fixo."""
    n_tecnicas = np.random.randint(1, 3)
    tecnicas_escolhidas = random.sample(TECNICAS, k=n_tecnicas)
    audio_aug = audio.copy()
    for tecnica in tecnicas_escolhidas:
        try:
            audio_aug = tecnica(audio_aug)
        except Exception:
            pass  # ignora falhas pontuais (ex.: time_stretch em áudios muito curtos)
    return audio_aug


# ── Extração de características ───────────────────────────────────────────────

def _paddar(audio: np.ndarray) -> np.ndarray:
    """Garante que o áudio tenha exatamente SR * DURACAO amostras."""
    tamanho = int(SR * DURACAO)
    if len(audio) < tamanho:
        audio = np.pad(audio, (0, tamanho - len(audio)))
    else:
        audio = audio[:tamanho]
    return audio


def extrair_caracteristicas(audio: np.ndarray) -> np.ndarray:
    """Converte array de áudio em matriz (tempo, MFCC, 3 canais)."""
    audio = _paddar(audio)
    mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=N_MFCC,
                                 n_fft=N_FFT, hop_length=HOP_LENGTH)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    return np.stack([mfcc, delta, delta2], axis=-1).astype(np.float32)


# ── Carregamento do dataset (apenas originais) ───────────────────────────────

def carregar_dataset() -> tuple[list[np.ndarray], np.ndarray, list[str]]:
    """Retorna os áudios originais brutos (sem augmentation) para split seguro."""
    classes = sorted(p.name for p in PASTA_DADOS.iterdir() if p.is_dir())
    if len(classes) < 2:
        raise RuntimeError("Crie pelo menos duas pastas de classes em dados/.")

    audios_brutos, rotulos = [], []
    for indice, classe in enumerate(classes):
        wavs = sorted((PASTA_DADOS / classe).glob("*.wav"))
        if not wavs:
            print(f"Aviso: a classe '{classe}' não possui WAV e será ignorada.")
            continue
        for wav in wavs:
            try:
                audio, _ = librosa.load(wav, sr=SR, mono=True, duration=DURACAO)
                audios_brutos.append(audio)
                rotulos.append(indice)
            except Exception as erro:
                print(f"Ignorando {wav}: {erro}")

    if len(audios_brutos) < len(classes) * 2:
        raise RuntimeError("Poucos áudios. Grave pelo menos 10 exemplos por classe.")

    print(f"{len(audios_brutos)} áudios originais carregados ({len(classes)} classes)")
    return audios_brutos, np.array(rotulos), classes


def construir_split(
    audios: list[np.ndarray],
    rotulos: np.ndarray,
    indices: np.ndarray,
    aumentar_dados: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Extrai features de um subconjunto e opcionalmente aplica augmentation."""
    X, y = [], []
    for i in indices:
        audio = audios[i]
        X.append(extrair_caracteristicas(audio))
        y.append(rotulos[i])
        if aumentar_dados:
            for _ in range(N_AUMENTOS):
                X.append(extrair_caracteristicas(aumentar(audio)))
                y.append(rotulos[i])
    return np.array(X, dtype=np.float32), np.array(y)


# ── Modelo ────────────────────────────────────────────────────────────────────

def criar_modelo(formato: tuple[int, int, int], quantidade_classes: int) -> tf.keras.Model:
    modelo = tf.keras.Sequential([
        tf.keras.layers.Input(shape=formato),

        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.20),

        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.35),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.20),
        tf.keras.layers.Dense(quantidade_classes, activation="softmax"),
    ])
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return modelo


# ── Treinamento principal ─────────────────────────────────────────────────────

def main() -> None:
    random.seed(SEMENTE)
    np.random.seed(SEMENTE)
    tf.random.set_seed(SEMENTE)
    PASTA_MODELO.mkdir(exist_ok=True)
    PASTA_RESULTADOS.mkdir(exist_ok=True)

    # 1) Carrega apenas os áudios originais
    print("Carregando áudios originais...")
    audios, rotulos, classes = carregar_dataset()
    indices = np.arange(len(audios))

    # 2) Separa índices treino/teste ANTES de qualquer augmentation
    idx_treino, idx_teste = train_test_split(
        indices, test_size=0.20, random_state=SEMENTE, stratify=rotulos)

    # 3) Augmentation SOMENTE no treino; teste usa apenas originais
    print(f"Aplicando augmentation no treino (×{N_AUMENTOS + 1})...")
    X_treino, y_treino = construir_split(audios, rotulos, idx_treino, aumentar_dados=True)
    X_teste,  y_teste  = construir_split(audios, rotulos, idx_teste,  aumentar_dados=False)
    print(f"Treino: {len(X_treino)} amostras | Teste: {len(X_teste)} amostras")

    # 4) Normalização por canal (média e desvio sobre amostras, linhas MFCC E tempo)
    # axis=(0,1,2) → resultado shape (1,1,1,3) → um valor por canal
    # Isso torna o modelo insensível a QUANDO no clip de 2s a fala aparece.
    media = X_treino.mean(axis=(0, 1, 2), keepdims=True)
    desvio = X_treino.std(axis=(0, 1, 2), keepdims=True)
    X_treino = (X_treino - media) / (desvio + 1e-8)
    X_teste  = (X_teste  - media) / (desvio + 1e-8)

    modelo = criar_modelo(X_treino.shape[1:], len(classes))
    modelo.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=6, min_lr=1e-6),
    ]

    historico = modelo.fit(
        X_treino, y_treino,
        validation_split=0.20,
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Avaliação ──────────────────────────────────────────────────────────────
    perda, acuracia = modelo.evaluate(X_teste, y_teste, verbose=0)
    probabilidades = modelo.predict(X_teste, verbose=0)
    previsoes = np.argmax(probabilidades, axis=1)
    print(f"\nAcurácia no teste: {acuracia:.2%}")
    print(classification_report(y_teste, previsoes, target_names=classes, zero_division=0))

    # ── Salvar modelo e normalização ───────────────────────────────────────────
    modelo.save(PASTA_MODELO / "comandos.keras")
    (PASTA_MODELO / "rotulos.json").write_text(
        json.dumps(classes, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez(PASTA_MODELO / "normalizacao.npz",
             media=media.squeeze(), desvio=desvio.squeeze())

    # ── Matriz de confusão ─────────────────────────────────────────────────────
    matriz = confusion_matrix(y_teste, previsoes, labels=range(len(classes)))
    plt.figure(figsize=(8, 6))
    sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.title(f"Matriz de confusão — acurácia: {acuracia:.2%}")
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(PASTA_RESULTADOS / "matriz_confusao.png", dpi=150)
    plt.close()

    # ── Curvas de treinamento ──────────────────────────────────────────────────
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(historico.history["accuracy"], label="treino")
    plt.plot(historico.history["val_accuracy"], label="validação")
    plt.title("Acurácia")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(historico.history["loss"], label="treino")
    plt.plot(historico.history["val_loss"], label="validação")
    plt.title("Perda")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PASTA_RESULTADOS / "curvas_treinamento.png", dpi=150)
    plt.close()

    print("Arquivos salvos em modelo/ e resultados/.")


if __name__ == "__main__":
    main()
