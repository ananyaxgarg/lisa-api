import os
import uuid
import tempfile
from typing import Dict, Any, Optional

import numpy as np
import librosa
from scipy.signal import butter, filtfilt
from fastapi import FastAPI, File, UploadFile, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

# ---- model / feature config -------------------------------------------------
try:
    import keras
    load_model = keras.saving.load_model
except Exception:
    from tensorflow.keras.models import load_model  # fallback

ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", "./PILOT")

MODEL_MURMUR = os.path.join(ARTIFACTS_DIR, "model_murmur_presence.keras")
MODEL_HEART_HEALTH = os.path.join(ARTIFACTS_DIR, "model_normal_abnormal.keras")
MU_SIGMA_V1_PATH = os.path.join(ARTIFACTS_DIR, "mu_sigma_v1.npy")
MU_SIGMA_V2_PATH = os.path.join(ARTIFACTS_DIR, "mu_sigma_v2.npy")

SR = 22050
DUR = 5
FIX_SAMPLES = SR * DUR
N_MELS = 128
N_FFT = 1024
HOP = 256

app = FastAPI(title="LISA API", version="1.0.0")

# ---- CORS -------------------------------------------------------------------
allowed_origins = [
    "https://lisa-yb7v.onrender.com",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
extra = os.environ.get("ALLOW_ORIGINS", "")
if extra:
    allowed_origins.extend([o.strip() for o in extra.split(",") if o.strip()])

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(set(allowed_origins)),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- DSP helpers -------------------------------------------------------------
def bandpass_20_400(y: np.ndarray, sr: int = SR, lo: float = 20.0, hi: float = 400.0, order: int = 4) -> np.ndarray:
    ny = 0.5 * sr
    b, a = butter(order, [lo / ny, hi / ny], btype="band")
    try:
        return filtfilt(b, a, y).astype(np.float32)
    except Exception:
        return y.astype(np.float32)

def rms_normalize(y: np.ndarray, target: float = -20.0) -> np.ndarray:
    rms = np.sqrt(np.mean(y ** 2) + 1e-12)
    if rms < 1e-6:
        return y
    gain = 10 ** (target / 20.0) / (rms + 1e-12)
    y = np.clip(y * gain, -1.0, 1.0)
    return y.astype(np.float32)

def load_fixed(y: np.ndarray) -> np.ndarray:
    if len(y) >= FIX_SAMPLES:
        return y[:FIX_SAMPLES]
    return np.pad(y, (0, FIX_SAMPLES - len(y)))

def extract_mel_v1(file_path: str) -> np.ndarray:
    y, _ = librosa.load(file_path, sr=SR)
    y = load_fixed(y)
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP, power=2.0)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.astype(np.float32)

def extract_mel_v2(file_path: str) -> np.ndarray:
    y, _ = librosa.load(file_path, sr=SR)
    y = y - np.mean(y)
    y = bandpass_20_400(y, sr=SR)
    y = rms_normalize(y)
    y = load_fixed(y)
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP, power=2.0)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.astype(np.float32)

def to4_dup(mel_2d: np.ndarray, mu_vec: np.ndarray, sg_vec: np.ndarray) -> np.ndarray:
    x4 = np.repeat(mel_2d[..., np.newaxis], 4, axis=-1)
    return (x4 - mu_vec.reshape(1, 1, 4)) / (sg_vec.reshape(1, 1, 4) + 1e-6)

def to4_single(mel_2d: np.ndarray, mu_vec: np.ndarray, sg_vec: np.ndarray) -> np.ndarray:
    F, T = mel_2d.shape
    x4 = np.zeros((F, T, 4), dtype=np.float32)
    x4[..., 0] = mel_2d
    return (x4 - mu_vec.reshape(1, 1, 4)) / (sg_vec.reshape(1, 1, 4) + 1e-6)

def predict_avg(model, x_dup: np.ndarray, x_single: np.ndarray) -> float:
    p1 = float(model.predict(x_dup, verbose=0)[0][0])
    p2 = float(model.predict(x_single, verbose=0)[0][0])
    return 0.5 * (p1 + p2)

def _load_threshold(thr_path: str, default_val: float) -> float:
    try:
        return float(np.load(thr_path)[0])
    except Exception:
        return default_val

(mu1, sg1) = np.load(MU_SIGMA_V1_PATH, allow_pickle=True)
(mu2, sg2) = np.load(MU_SIGMA_V2_PATH, allow_pickle=True)
mu1, sg1, mu2, sg2 = mu1.astype(np.float32), sg1.astype(np.float32), mu2.astype(np.float32), sg2.astype(np.float32)

murmur_model = load_model(MODEL_MURMUR, compile=False, safe_mode=False)
health_model = load_model(MODEL_HEART_HEALTH, compile=False, safe_mode=False)

T_MURMUR = _load_threshold(MODEL_MURMUR.replace(".keras", "_thr.npy"), 0.5)
T_HEALTH = _load_threshold(MODEL_HEART_HEALTH.replace(".keras", "_thr.npy"), 0.064)

def run_inference_on_file(path: str) -> Dict[str, Any]:
    mel1 = extract_mel_v1(path)
    x1_dup = to4_dup(mel1, mu1, sg1)[np.newaxis, ...]
    x1_single = to4_single(mel1, mu1, sg1)[np.newaxis, ...]
    m_score = predict_avg(murmur_model, x1_dup, x1_single)
    m_label = "Present" if m_score >= T_MURMUR else "Absent"

    mel2 = extract_mel_v2(path)
    x2_dup = to4_dup(mel2, mu2, sg2)[np.newaxis, ...]
    h_score = float(health_model.predict(x2_dup, verbose=0)[0][0])
    h_label = "Abnormal" if h_score >= T_HEALTH else "Normal"

    return {
        "murmur_score": round(m_score, 3),
        "murmur_label": m_label,
        "heart_health_score": round(h_score, 3),
        "heart_health_label": h_label,
        "thr_murmur": round(T_MURMUR, 3),
        "thr_heart": round(T_HEALTH, 3),
    }

# ---- endpoints --------------------------------------------------------------
@app.get("/")
def root():
    return {"ok": True, "message": "LISA API is up", "endpoints": ["/health", "/recording"]}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "models": {
            "murmur": os.path.basename(MODEL_MURMUR),
            "heart": os.path.basename(MODEL_HEART_HEALTH),
        },
    }

# allow preflight explicitly (some proxies are picky)
@app.options("/recording")
def options_recording():
    return Response(status_code=204)

# accept 'file' (preferred) or 'audio' (fallback) for robustness
@app.post("/recording")
async def analyse_recording(
    file: Optional[UploadFile] = File(default=None),
    audio: Optional[UploadFile] = File(default=None),
    origin: Optional[str] = Header(default=None),
):
    upload: UploadFile = file or audio
    if upload is None:
        return JSONResponse(status_code=400, content={"ok": False, "error": "file required (field name 'file' or 'audio')"})

    suffix = os.path.splitext(upload.filename or "")[-1].lower()
    if suffix not in [".wav", ".mp3", ".webm", ".m4a", ".ogg", ".flac", ".aac"]:
        suffix = ".wav"

    tmp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}{suffix}")
    try:
        with open(tmp_path, "wb") as f:
            f.write(await upload.read())
        result = run_inference_on_file(tmp_path)
        return JSONResponse({"ok": True, "filename": upload.filename, "result": result})
    except Exception as e:
        return JSONResponse(status_code=400, content={"ok": False, "error": str(e)})
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
