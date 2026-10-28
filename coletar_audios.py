"""Coleta áudios para o dataset do projeto.

Uso:
    python coletar_audios.py --por-classe 30

Os arquivos serão salvos em dados/iniciar_receita, dados/proximo_passo,
 dados/repetir_instrucao, dados/finalizar_receita e dados/desconhecido.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import sounddevice as sd
from scipy.io.wavfile import write

CLASSES = ["iniciar_receita", "proximo_passo", "repetir_instrucao",
           "finalizar_receita", "desconhecido"]
SAMPLE_RATE = 16_000
DURACAO = 2.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Coleta comandos de voz em WAV.")
    parser.add_argument("--por-classe", type=int, default=30,
                        help="quantidade de gravações por comando")
    parser.add_argument("--duracao", type=float, default=DURACAO,
                        help="duração de cada gravação em segundos")
    args = parser.parse_args()

    print("Dê preferência a um local silencioso e mantenha distância semelhante do microfone.")
    print(f"Cada gravação terá {args.duracao:.1f}s. Pressione Enter para iniciar cada uma.")

    for classe in CLASSES:
        pasta = Path("dados") / classe
        pasta.mkdir(parents=True, exist_ok=True)
        existentes = len(list(pasta.glob("*.wav")))
        print(f"\nClasse: {classe} | já existem {existentes} arquivo(s)")
        for indice in range(existentes, args.por_classe):
            input(f"[{indice + 1}/{args.por_classe}] Fale '{classe.replace('_', ' ')}' e pressione Enter...")
            print("Gravando...")
            audio = sd.rec(int(args.duracao * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                          channels=1, dtype="float32")
            sd.wait()
            caminho = pasta / f"{classe}_{indice + 1:03d}.wav"
            write(caminho, SAMPLE_RATE, audio)
            print(f"Salvo: {caminho}")
            time.sleep(0.3)

    print("\nColeta concluída. Confira a quantidade de arquivos em cada pasta.")


if __name__ == "__main__":
    main()
