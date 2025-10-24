import os, uuid, tempfile, subprocess
import numpy as np
import librosa
from scipy.signal import butter, filtfilt
from fastapi import FastAPI, File, UploadFile, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model

# ======================
# Paths
# ======================
ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", "./PILOT")
MODEL_MURMUR       = os.path.join(ARTIFACTS_DIR, "model_murmur_presence.keras")
MODEL_HEART_HEALTH = os.path.join(ARTIFACTS_DIR, "model_normal_abnormal.keras")
MU_SIGMA_V1_PATH   = os.path.join(ARTIFACTS_DIR, "mu_sigma_v1.npy")
MU_SIGMA_V2_PATH   = os.path.join(ARTIFACTS_DIR, "mu_sigma_v2.npy")

# ======================
# App + CORS setup
# ======================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # later: ["https://lisa-yb7v.onrender.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================
# Constants (from your training)
# ======================
SR = 22050
DUR = 5
FIX_SAMPLES = SR * DUR
N_MELS = 128
N_FFT = 1024
HOP = 256

# ======================
# DSP + feature helpers
# ======================
def bandpass_20_400(y, sr=SR, lo=20.0, hi=400.0, order=4):
    ny = 0.5 * sr
    b, a = butter(order, [lo/ny, hi/ny], btype='band')
    try:
        return filtfilt(b, a, y).astype(np.float32)
    except Exception:
        return y.astype(np.float32)

def rms_normalize(y, target=-20.0):
    rms = np.sqrt(np.mean(y**2) + 1e-12)
    if rms < 1e-6: return y
    gain = 10 ** (target / 20.0) / (rms + 1e-12)
    return np.clip(y * gain, -1.0, 1.0).astype(np.float32)

def load_fixed(y):
    if len(y) >= FIX_SAMPLES: return y[:FIX_SAMPLES]
    return np.pad(y, (0, FIX_SAMPLES - len(y)))

def extract_mel_v1(file):
    y, _ = librosa.load(file, sr=SR)
    y = load_fixed(y)
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP, power=2.0)
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)

def extract_mel_v2(file):
    y, _ = librosa.load(file, sr=SR)
    y = y - np.mean(y)
    y = bandpass_20_400(y, sr=SR)
    y = rms_normalize(y)
    y = load_fixed(y)
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP, power=2.0)
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)

def to4_dup(mel_2d, mu_vec, sg_vec):
    x4 = np.repeat(mel_2d[..., np.newaxis], 4, axis=-1)
    return (x4 - mu_vec.reshape(1,1,4)) / (sg_vec.reshape(1,1,4) + 1e-6)

def to4_single(mel_2d, mu_vec, sg_vec):
    F, T = mel_2d.shape
    x4 = np.zeros((F, T, 4), dtype=np.float32)
    x4[..., 0] = mel_2d
    return (x4 - mu_vec.reshape(1,1,4)) / (sg_vec.reshape(1,1,4) + 1e-6)

def predict_avg(model, x_dup, x_single):
    p1 = float(model.predict(x_dup,    verbose=0)[0][0])
    p2 = float(model.predict(x_single, verbose=0)[0][0])
    return 0.5 * (p1 + p2)

def to_wav(in_path, out_path, sr=SR):
    subprocess.run(
        ["ffmpeg","-y","-i",in_path,"-ac","1","-ar",str(sr),out_path],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

# ======================
# Load models + μ/σ once
# ======================
(mu1, sg1) = np.load(MU_SIGMA_V1_PATH, allow_pickle=True)
(mu2, sg2) = np.load(MU_SIGMA_V2_PATH, allow_pickle=True)
mu1, sg1, mu2, sg2 = mu1.astype(np.float32), sg1.astype(np.float32), mu2.astype(np.float32), sg2.astype(np.float32)

murmur_model = load_model(MODEL_MURMUR, compile=False, safe_mode=False)
health_model = load_model(MODEL_HEART_HEALTH, compile=False, safe_mode=False)

T_MURMUR = float(os.environ.get("T_MURMUR", "0.5"))
T_HEALTH = float(os.environ.get("T_HEALTH", "0.064"))  # forced threshold

# ======================
# Routes
# ======================
@app.get("/health")
def health():
    return {"ok": True}

@app.post("/recording")
async def recording(file: UploadFile = File(...), x_api_key: str | None = Header(default=None)):
    api_key = os.environ.get("API_KEY")
    if api_key and x_api_key != api_key:
        return JSONResponse({"ok": False, "error": "Unauthorized"}, status_code=401)

    raw_suffix = os.path.splitext(file.filename or "")[-1].lower() or ".webm"
    raw = os.path.join(tempfile.gettempdir(), f"up_{uuid.uuid4().hex}{raw_suffix}")
    wav = os.path.join(tempfile.gettempdir(), f"cv_{uuid.uuid4().hex}.wav")

    try:
        with open(raw, "wb") as f:
            f.write(await file.read())

        to_wav(raw, wav, sr=SR)

        # --- murmur presence
        mel1 = extract_mel_v1(wav)
        x1_dup, x1_single = to4_dup(mel1, mu1, sg1)[np.newaxis, ...], to4_single(mel1, mu1, sg1)[np.newaxis, ...]
        m_score = predict_avg(murmur_model, x1_dup, x1_single)
        m_label = "Present" if m_score >= T_MURMUR else "Absent"

        # --- heart health
        mel2 = extract_mel_v2(wav)
        x2_dup = to4_dup(mel2, mu2, sg2)[np.newaxis, ...]
        h_score = float(health_model.predict(x2_dup, verbose=0)[0][0])
        h_label = "Abnormal" if h_score >= T_HEALTH else "Normal"

        return JSONResponse({
            "ok": True,
            "scores": {
                "murmur_presence": round(m_score, 4),
                "heart_health": round(h_score, 4)
            },
            "labels": {
                "murmur": m_label,
                "heart_health": h_label
            },
            "thresholds": {
                "murmur": T_MURMUR,
                "heart_health": T_HEALTH
            }
        })
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
    finally:
        for p in (raw, wav):
            try: os.remove(p)
            except: pass
