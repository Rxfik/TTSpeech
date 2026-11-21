"""
Interface web TTS : Tacotron2-DDC (+ HiFi-GAN) vs VITS  (LJSpeech, anglais)
Lancement :  python app.py   puis ouvrir http://127.0.0.1:7860
"""
import os
import shutil
import threading
import time
from pathlib import Path

# Cache des modèles dans le dossier du projet (à définir AVANT d'importer TTS)
os.environ.setdefault("TTS_HOME", str(Path(__file__).parent.joinpath("tts_cache").resolve()))
Path(os.environ["TTS_HOME"]).mkdir(parents=True, exist_ok=True)

import gradio as gr
import numpy as np
import pandas as pd
import torch
from TTS.api import TTS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_CHARS = 400

MODELS = {
    "Tacotron2-DDC + HiFi-GAN": "tts_models/en/ljspeech/tacotron2-DDC",
    "VITS": "tts_models/en/ljspeech/vits",
}
CHOICES = list(MODELS) + ["Les deux (comparaison)"]

_cache: dict[str, TTS] = {}
_lock = threading.Lock()  # un seul calcul à la fois (évite de saturer RAM/GPU)


def get_model(label: str) -> TTS:
    """Charge un modèle à la première utilisation, puis le garde en mémoire."""
    if label in _cache:
        return _cache[label]
    if label == "VITS" and not (shutil.which("espeak-ng") or shutil.which("espeak")):
        raise gr.Error(
            "espeak-ng est introuvable (nécessaire pour VITS). "
            "Windows : installer le .msi espeak-ng et l'ajouter au PATH, "
            "macOS : brew install espeak-ng, Linux : sudo apt install espeak-ng. "
            "Puis relancer l'application."
        )
    gr.Info(f"Chargement de {label} (première utilisation, quelques secondes)...")
    model = TTS(model_name=MODELS[label], progress_bar=False).to(DEVICE)
    model.tts("Warm up.")  # échauffement non mesuré
    _cache[label] = model
    return model


def synthesize(text: str, label: str):
    model = get_model(label)
    t0 = time.perf_counter()
    wav = np.asarray(model.tts(text), dtype=np.float32)
    gen_time = time.perf_counter() - t0
    sr = model.synthesizer.output_sample_rate
    duration = len(wav) / sr
    row = {
        "Modèle": label,
        "Temps génération (s)": round(gen_time, 2),
        "Durée audio (s)": round(duration, 2),
        "RTF": round(gen_time / duration, 3),
    }
    return (sr, wav), row


def generate(text: str, choice: str):
    text = " ".join((text or "").split())
    if not text:
        raise gr.Error("Écris d'abord une phrase en anglais.")
    if len(text) > MAX_CHARS:
        raise gr.Error(f"Texte trop long ({len(text)} caractères). Maximum : {MAX_CHARS}.")
    if sum(not c.isascii() for c in text) > 0.2 * len(text):
        gr.Warning("Les modèles sont entraînés sur de l'anglais : le résultat risque d'être mauvais.")

    labels = list(MODELS) if choice == CHOICES[-1] else [choice]
    audios = {l: None for l in MODELS}
    rows = []
    with _lock:
        for label in labels:
            audio, row = synthesize(text, label)
            audios[label] = audio
            rows.append(row)

    table = pd.DataFrame(rows)
    return audios["Tacotron2-DDC + HiFi-GAN"], audios["VITS"], table


with gr.Blocks(title="TTS : Tacotron2-DDC vs VITS") as demo:
    gr.Markdown(
        "# Synthèse vocale : Tacotron2-DDC + HiFi-GAN vs VITS\n"
        f"Voix LJSpeech (anglais). Appareil : `{DEVICE}`. "
        "RTF = temps de génération ÷ durée audio (< 1 : plus rapide que la parole)."
    )
    with gr.Row():
        with gr.Column(scale=3):
            text_in = gr.Textbox(
                label="Texte (en anglais)", lines=4, max_lines=8,
                placeholder="Type an English sentence here...",
            )
            model_in = gr.Radio(CHOICES, value=CHOICES[-1], label="Modèle")
            btn = gr.Button("Générer la voix", variant="primary")
            gr.Examples(
                examples=[
                    ["Artificial intelligence is changing the way we interact with technology."],
                    ["Text to speech converts written words into natural spoken audio."],
                    ["In 2025, the project used more than 13000 recordings of a single speaker."],
                ],
                inputs=text_in,
            )
        with gr.Column(scale=3):
            out_taco = gr.Audio(label="Tacotron2-DDC + HiFi-GAN", type="numpy")
            out_vits = gr.Audio(label="VITS", type="numpy")
            out_table = gr.Dataframe(label="Mesures", interactive=False)

    btn.click(generate, inputs=[text_in, model_in], outputs=[out_taco, out_vits, out_table])
    text_in.submit(generate, inputs=[text_in, model_in], outputs=[out_taco, out_vits, out_table])

if __name__ == "__main__":
    demo.queue(max_size=8).launch(server_name="127.0.0.1")