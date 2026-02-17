import os
import math
import wave
import numpy as np
import subprocess
import tflite_runtime.interpreter as tflite

# --- Constants (paths relative to this script so they work in Docker / any cwd) ---
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_SCRIPT_DIR, "audio-model-int8.tflite")
LABELS_PATH = os.path.join(_SCRIPT_DIR, "labels", "en_us.txt")
SAMPLE_WAV = os.path.join(_SCRIPT_DIR, "samples", "sample.wav")

# Model expects 48 kHz, 3-second windows (144000 samples)
SAMPLE_RATE = 48000
MODEL_INPUT_SEC = 3.0

STEP_SEC = 1.5          # slide step (matches your analysis step)
THRESHOLD = 0.6

# Keep your same preprocessing behavior
mic_gain = 3.0


def load_labels(path: str) -> list[str]:
    with open(path, "r") as f:
        # matches your original: take last underscore chunk
        return [line.strip().split("_")[-1] for line in f.readlines()]


def load_model(model_path: str):
    interpreter = tflite.Interpreter(model_path=model_path, num_threads=4)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details


def _resample_stereo(audio: np.ndarray, orig_rate: int, target_rate: int) -> np.ndarray:
    """Resample stereo (N, 2) from orig_rate to target_rate via linear interpolation."""
    n = audio.shape[0]
    new_n = int(n * target_rate / orig_rate)
    if new_n == n:
        return audio
    x_old = np.arange(n, dtype=np.float64)
    x_new = np.linspace(0, n - 1, new_n, dtype=np.float64)
    out = np.column_stack([
        np.interp(x_new, x_old, audio[:, 0]),
        np.interp(x_new, x_old, audio[:, 1]),
    ]).astype(np.float32)
    return out


def read_wav_stereo(path: str, expected_rate: int = 48000) -> np.ndarray:
    """
    Reads a WAV file using Python stdlib only (no ffmpeg).
    Supports PCM int16 and PCM float32. If file rate != expected_rate, resamples to expected_rate.
    Returns float32 numpy array shape (N, 2).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing WAV file: {path}")

    with wave.open(path, "rb") as wf:
        channels = wf.getnchannels()
        rate = wf.getframerate()
        sampwidth = wf.getsampwidth()
        nframes = wf.getnframes()

        if channels != 2:
            raise ValueError(f"WAV must be stereo (2ch). Got {channels}ch")
        if sampwidth not in (2, 4):
            raise ValueError(f"Unsupported sample width {sampwidth} bytes. Use PCM16 or PCM_F32.")

        frames = wf.readframes(nframes)

    if sampwidth == 2:
        audio_i16 = np.frombuffer(frames, dtype=np.int16).reshape(-1, 2)
        audio = audio_i16.astype(np.float32) / 32768.0
    else:
        audio = np.frombuffer(frames, dtype=np.float32).reshape(-1, 2).astype(np.float32)

    if rate != expected_rate:
        audio = _resample_stereo(audio, rate, expected_rate)
    return audio


def run_inference(
    interpreter,
    input_details,
    output_details,
    chunk_stereo: np.ndarray,
    threshold: float = THRESHOLD,
    mic_gain_val: float = mic_gain,
):
    # 1) Stereo -> mono
    chunk_mono = np.mean(chunk_stereo, axis=1)

    # 2) "Safe high-pass": subtract rolling mean
    rolling_mean = np.convolve(chunk_mono, np.ones(10) / 10, mode="same")
    chunk_mono = chunk_mono - rolling_mean

    # 3) Gain
    chunk_mono = chunk_mono * mic_gain_val

    # 4) Normalize
    max_val = np.max(np.abs(chunk_mono))
    if max_val > 0:
        chunk_mono = chunk_mono / max_val

    # 5) Model
    input_data = np.expand_dims(chunk_mono, axis=0).astype(np.float32)
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]["index"])[0]
    scale, zero_point = output_details[0].get("quantization", (0.0, 0))

    results = []
    for i, raw_score in enumerate(output_data):
        logit = (raw_score - zero_point) * scale if scale else raw_score
        confidence = 1.0 / (1.0 + np.exp(-logit))
        if confidence > threshold:
            results.append((i, float(confidence)))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def main():
    print("Loading model...")
    interpreter, input_details, output_details = load_model(MODEL_PATH)
    labels = load_labels(LABELS_PATH)

    print(f"Reading WAV: {SAMPLE_WAV}")
    audio_stereo = read_wav_stereo(SAMPLE_WAV, expected_rate=SAMPLE_RATE)

    total_samples = audio_stereo.shape[0]
    window_samples = int(SAMPLE_RATE * MODEL_INPUT_SEC)
    step_samples = int(SAMPLE_RATE * STEP_SEC)

    if total_samples < window_samples:
        pad = window_samples - total_samples
        audio_stereo = np.pad(audio_stereo, ((0, pad), (0, 0)), mode="constant")
        total_samples = audio_stereo.shape[0]

    duration_sec = total_samples / SAMPLE_RATE
    print(
        f"Running inference over audio: {duration_sec:.1f}s "
        f"(window={MODEL_INPUT_SEC:.1f}s, step={STEP_SEC:.1f}s, thr={THRESHOLD})"
    )

    # species -> best confidence observed
    detections: dict[str, float] = {}

    for start in range(0, max(1, total_samples - window_samples + 1), step_samples):
        window = audio_stereo[start : start + window_samples]
        results = run_inference(interpreter, input_details, output_details, window)

        if not results:
            continue

        # keep top-3 per window (like your look-back loop)
        for idx, conf in results[:3]:
            name = labels[idx] if idx < len(labels) else f"label_{idx}"
            if name not in detections or conf > detections[name]:
                detections[name] = conf

    if not detections:
        print("No birds detected.")
        return

    print("\nDetections (best confidence per species):")
    for name, conf in sorted(detections.items(), key=lambda x: x[1], reverse=True):
        print(f"- {name}: {conf*100:.1f}%")


if __name__ == "__main__":
    main()
