# -*- coding: utf-8 -*-
# PARCA 01: giris, sabitler, yol yardimcilari
import hashlib
import io
import mimetypes
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.parse
from pathlib import Path

BASE = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(BASE / "hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

PORT = 5102
APP_VER = "sade-20261007c"
MODEL_ID = "google/embeddinggemma-2"
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".avif",
           ".tif", ".tiff", ".heic", ".heif"}
VID_EXT = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".mpg", ".mpeg"}
AUD_EXT = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".opus", ".aac"}
TXT_EXT = {".txt", ".md"}
URL_EXT = {".url"}
MEDIA_EXT = IMG_EXT | VID_EXT | AUD_EXT | TXT_EXT
MAX_CANDIDATES = 1500
MAX_QUERY_FILES = 25
MAX_FRAMES_PER_VIDEO = 12
MAX_AUDIO_SEGMENTS = 6
AUDIO_SEG_SEC = 30
FRAME_FPS = 1.0
CHUNK = 8
THUMB_SIZE = 320
RESULT_THUMBS = 160
BROWSE_FILE_CAP = 500
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"}

from flask import Flask, request, jsonify, send_file  # noqa: E402

try:  # iPhone HEIC/HEIF fotograflari PIL ile acilabilsin (yoksa zarif atlanir)
    from pillow_heif import register_heif_opener  # noqa: E402
    register_heif_opener()
except Exception:
    pass

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 300 * 1024 * 1024

_MODEL = None
_FORMS = {}
_MODEL_LOCK = threading.Lock()
JOB = {"state": "idle", "phase": "-", "done": 0, "total": 0, "found": 0,
       "results": [], "error": None, "out_dir": "", "refs": [], "src": [],
       "query": "", "units": 0}
JOB_LOCK = threading.Lock()
STOP = threading.Event()  # durdurma bayragi — her asama kontrol eder


class ScanStopped(Exception):
    """Kullanici 'Durdur'a basti."""


def _stop():
    if STOP.is_set():
        raise ScanStopped()


def try_mount_drive(letter):
    mnt = Path("/mnt") / letter.lower()
    if mnt.exists():
        return True
    mnt.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["wsl.exe", "-u", "root", "--", "mount", "-t", "drvfs",
                        f"{letter}:", str(mnt)], timeout=45, capture_output=True)
    except Exception:
        pass
    return mnt.exists()


def to_wsl_path(raw):
    s = str(raw or "").strip().strip('"').strip("'")
    if not s:
        return s
    s = s.replace("\\", "/")
    if os.name == "nt":
        if re.match(r"^[A-Za-z]:", s) or s.startswith("/"):
            return s
        if s == "~" or s.startswith("~/"):
            return str(Path(s).expanduser())
        return str((BASE / s).resolve())
    m = re.match(r"^([A-Za-z]):(/.*|)$", s)
    if m:
        letter, rest = m.group(1), (m.group(2) or "/")
        if not try_mount_drive(letter):
            return str(raw)
        base = Path("/mnt") / letter.lower()
        return str(base) if rest == "/" else str(base) + rest
    if s == "~" or s.startswith("~/"):
        return str(Path(s).expanduser())
    if s.startswith("/"):
        return s
    return str((BASE / s).resolve())


def win_of(p):
    if os.name == "nt":
        return p or ""
    m = re.match(r"^/mnt/([A-Za-z])(/.*)?$", p or "")
    if not m:
        return p or ""
    rest = m.group(2) or "/"
    return m.group(1).upper() + ":" + ("" if rest == "/" else rest.replace("/", "\\"))


def setj(**kw):
    with JOB_LOCK:
        JOB.update(kw)


def media_kind(name):
    suf = Path(str(name)).suffix.lower()
    if suf in IMG_EXT:
        return "image"
    if suf in VID_EXT:
        return "video"
    if suf in AUD_EXT:
        return "audio"
    if suf in TXT_EXT:
        return "text"
    if suf in URL_EXT:
        return "url"
    return "other"


def fmt_time(sec):
    try:
        s = int(float(sec))
    except Exception:
        return "00:00"
    return f"{s // 60:02d}:{s % 60:02d}"


def read_url_file(path):
    """Windows .url kisayolu veya icinde URL olan duz metin -> URL."""
    txt = Path(str(path)).read_text(encoding="utf-8", errors="replace")
    for line in txt.splitlines():
        line = line.strip()
        if line.lower().startswith("url="):
            line = line[4:].strip()
        if line.startswith(("http://", "https://")):
            return line
    raise RuntimeError(f"URL bulunamadi: {Path(str(path)).name}")
# -*- coding: utf-8 -*-
# PARCA 02a: model yukleme + prob yardimcilari
def _diag_processor_error():
    """EmbeddingGemma2Processor sarmalayici hatasini koke iner:
    asil ModuleNotFoundError'i (orn. torchvision eksik) dogrudan uretir."""
    try:
        from transformers.models.embedding_gemma2.processing_embedding_gemma2 import (  # noqa: F401
            EmbeddingGemma2Processor)
        return ""
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def get_model():
    """Modeli bir kez yukle — TAM surum (ses+video encoder acik)."""
    global _MODEL
    with _MODEL_LOCK:
        if _MODEL is None:
            import torch
            from sentence_transformers import SentenceTransformer
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            # sarmalayici hatanin altindaki asil import hatasini gorunur kil
            try:
                import transformers
                transformers.utils.logging.set_verbosity_debug()
            except Exception:
                pass
            try:
                _MODEL = SentenceTransformer(
                    MODEL_ID, device=dev,
                    model_kwargs={"torch_dtype": dtype}, trust_remote_code=True)
            except TypeError:
                _MODEL = SentenceTransformer(
                    MODEL_ID, device=dev, model_kwargs={"torch_dtype": dtype})
            except Exception as e:
                cause = _diag_processor_error()
                if cause:
                    missing = ""
                    if "No module named" in cause:
                        pkg = cause.split("No module named")[-1].strip().strip("'\"")
                        missing = (f" | EKSIK PAKET: '{pkg}' — cozum: "
                                   f"winvenv\\Scripts\\pip install {pkg.split('.')[0]}")
                    raise RuntimeError(
                        f"{e} | ASIL HATA: {cause}{missing}") from e
                raise
    return _MODEL


def _norm_ok(vec, dim):
    import numpy as np
    a = np.asarray(vec, dtype="float32")
    return (a.ndim == 1 and a.shape[0] == dim
            and bool(np.isfinite(a).all())
            and 0.9 < float((a ** 2).sum() ** 0.5) < 1.1)


def _encode_batch(model, batch):
    import numpy as np
    out = model.encode(batch, normalize_embeddings=True)
    arr = np.asarray(out, dtype="float32")
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr
# -*- coding: utf-8 -*-
# PARCA 02b: gorsel/ses giris bicimi problari
def probe_image_form(model):
    if _FORMS.get("image"):
        return _FORMS["image"]
    import numpy as np
    from PIL import Image
    dim = model.get_sentence_embedding_dimension()
    td = BASE / "cache" / "_probe"
    td.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (160, 120), (200, 40, 40))
    pa, pb = td / "aaa_zzz_identical.jpg", td / "b_different_name.jpg"
    img.save(pa)
    img.save(pb)
    cands = [
        ("dict", lambda ps: [{"image": [str(p)]} for p in ps]),
        ("path", lambda ps: [str(p) for p in ps]),
        ("pil", lambda ps: [Image.open(p).convert("RGB") for p in ps]),
    ]
    for name, builder in cands:
        try:
            arr = _encode_batch(model, builder([pa, pb]))
            if arr.shape[0] == 2 and _norm_ok(arr[0], dim) and _norm_ok(arr[1], dim):
                if float(arr[0] @ arr[1]) > 0.995:
                    _FORMS["image"] = name
                    return name
        except Exception:
            continue
    raise RuntimeError("Model gorsel girdisi desteklemiyor")


def probe_audio_form(model):
    if "audio" in _FORMS:
        return _FORMS["audio"]
    import numpy as np
    import soundfile as sf
    dim = model.get_sentence_embedding_dimension()
    td = BASE / "cache" / "_probe"
    td.mkdir(parents=True, exist_ok=True)
    wav = td / "probe_sine.wav"
    if not wav.exists():
        sr = 16000
        t = np.arange(sr, dtype="float32") / sr
        sf.write(str(wav), 0.3 * np.sin(2 * np.pi * 440 * t).astype("float32"), sr)
    cands = [
        ("adict", lambda p: [{"audio": str(p)}]),
        ("alist", lambda p: [{"audio": [str(p)]}]),
        ("apath", lambda p: [str(p)]),
    ]
    for name, builder in cands:
        try:
            arr = _encode_batch(model, builder(wav))
            if arr.shape[0] == 1 and _norm_ok(arr[0], dim):
                _FORMS["audio"] = name
                return name
        except Exception:
            continue
    _FORMS["audio"] = None
    return None
# -*- coding: utf-8 -*-
# PARCA 02c: birlesik metin+gorsel+ses encode
def encode_units(model, units, track=True):
    """units: [(kind, payload)] kind=text|image|audio -> (n, dim).
    track=False: ilerleme yazma (akisli tarama disaridan yonetir)."""
    import numpy as np
    from PIL import Image
    if not units:
        return np.zeros((0, model.get_sentence_embedding_dimension()), dtype="float32")
    groups = {}
    for i, (k, _p) in enumerate(units):
        groups.setdefault(k, []).append(i)
    res = [None] * len(units)
    done = 0
    total = len(units)
    if track:
        with JOB_LOCK:
            JOB["total"] = max(JOB.get("total", 0), total)
    if "text" in groups:
        idxs = groups["text"]
        for s in range(0, len(idxs), CHUNK):
            _stop()
            part = idxs[s:s + CHUNK]
            batch = [str(units[i][1])[:4000] for i in part]
            arr = _encode_batch(model, batch)
            for r, i in enumerate(part):
                res[i] = arr[r]
            done += len(part)
            if track:
                setj(done=min(done, JOB.get("total", done)))
    if "image" in groups:
        form = probe_image_form(model)
        idxs = groups["image"]
        for s in range(0, len(idxs), CHUNK):
            _stop()
            part = idxs[s:s + CHUNK]
            if form == "dict":
                batch = [{"image": [str(units[i][1])]} for i in part]
            elif form == "path":
                batch = [str(units[i][1]) for i in part]
            else:
                batch = [Image.open(str(units[i][1])).convert("RGB") for i in part]
            arr = _encode_batch(model, batch)
            for r, i in enumerate(part):
                res[i] = arr[r]
            done += len(part)
            if track:
                setj(done=min(done, JOB.get("total", done)))
    if "audio" in groups:
        form = probe_audio_form(model)
        if not form:
            bad = [str(units[i][1]) for i in groups["audio"]][:3]
            raise RuntimeError("Model ses girdisini kabul etmedi: "
                               + ", ".join(Path(b).name for b in bad))
        idxs = groups["audio"]
        for s in range(0, len(idxs), CHUNK):
            _stop()
            part = idxs[s:s + CHUNK]
            if form == "adict":
                batch = [{"audio": str(units[i][1])} for i in part]
            elif form == "alist":
                batch = [{"audio": [str(units[i][1])]} for i in part]
            else:
                batch = [str(units[i][1]) for i in part]
            arr = _encode_batch(model, batch)
            for r, i in enumerate(part):
                res[i] = arr[r]
            done += len(part)
            if track:
                setj(done=min(done, JOB.get("total", done)))
    return np.vstack([np.asarray(r, dtype="float32") for r in res])
# -*- coding: utf-8 -*-
# PARCA 03a: video kareleri + ses segmentleri
def video_duration(path):
    try:
        import av
        with av.open(str(path)) as c:
            if c.duration:
                return float(c.duration) / 1000000.0
    except Exception:
        pass
    return 0.0


def extract_video_frames(path, out_dir, fps=FRAME_FPS, cap=MAX_FRAMES_PER_VIDEO):
    """Videodan 1 fps kare cikarir -> [(t, jpg_yolu)]."""
    import av
    from PIL import Image
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    try:
        with av.open(str(path)) as c:
            vs = c.streams.video[0]
            avg = float(vs.average_rate) if vs.average_rate else 25.0
            step = max(1, int(round(avg / fps)))
            idx = 0
            for fi, fr in enumerate(c.decode(video=0)):
                if fi % step != 0:
                    continue
                t = float(fr.time) if fr.time is not None else fi / avg
                p = out_dir / f"f{len(frames):03d}_t{int(t)}s.jpg"
                fr.to_image().convert("RGB").save(p, "JPEG", quality=85)
                frames.append((t, str(p)))
                idx += 1
                if len(frames) >= cap:
                    break
    except Exception:
        pass
    return frames


def audio_segments(path, out_dir, seg_sec=AUDIO_SEG_SEC, cap=MAX_AUDIO_SEGMENTS):
    """Sesi 16kHz wav'a cevirir, seg_sec'lik parcalara boler."""
    import numpy as np
    import soundfile as sf
    out_dir.mkdir(parents=True, exist_ok=True)
    segs = []
    try:
        data, sr = sf.read(str(path), always_2d=True)
        if sr != 16000:
            import librosa
            mono = data.mean(axis=1)
            mono = librosa.resample(mono, orig_sr=sr, target_sr=16000)
            data = mono[:, None]
            sr = 16000
        else:
            data = data.mean(axis=1, keepdims=True) if data.shape[1] > 1 else data
        n = int(seg_sec * sr)
        total = data.shape[0]
        i = 0
        while i * n < total and len(segs) < cap:
            chunk = data[i * n:(i + 1) * n, 0]
            p = out_dir / f"seg{i:02d}_t{int(i * seg_sec)}s.wav"
            sf.write(str(p), chunk.astype("float32"), sr)
            segs.append((float(i * seg_sec), str(p)))
            i += 1
    except Exception:
        pass
    return segs


def thumb_of(path, out_path, size=THUMB_SIZE):
    from PIL import Image
    try:
        im = Image.open(path)
        im.load()
        im = im.convert("RGB")
        im.thumbnail((size, size))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        im.save(out_path, "JPEG", quality=82)
        return True
    except Exception:
        return False


def file_thumb(path, out_path):
    """Gorsel -> kucuk onizleme; video -> orta kare; ses/metin -> rozet."""
    from PIL import Image, ImageDraw
    kind = media_kind(path)
    if kind == "image":
        return thumb_of(path, out_path, RESULT_THUMBS)
    if kind == "video":
        dur = video_duration(path)
        mid = dur / 2 if dur > 1 else 0
        fr = extract_video_frames(path, out_path.parent / (out_path.stem + "_vf"),
                                  fps=0.2, cap=3)
        pick = None
        if fr:
            pick = min(fr, key=lambda x: abs(x[0] - mid))[1]
        if pick:
            return thumb_of(pick, out_path, RESULT_THUMBS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGB", (RESULT_THUMBS, RESULT_THUMBS), (24, 26, 38))
    d = ImageDraw.Draw(im)
    label = {"audio": "SES", "text": "METIN"}.get(kind, "DOSYA")
    d.text((18, 66), label, fill=(126, 231, 135))
    try:
        d.text((18, 88), Path(path).suffix.upper()[:6], fill=(140, 148, 165))
    except Exception:
        pass
    im.save(out_path, "JPEG", quality=82)
    return True


def _try_image_unit(p):
    """Bilinmeyen uzanti once gorsel mi? (PIL + HEIC dahil) — gommeye uygunsa True."""
    from PIL import Image
    try:
        im = Image.open(str(p))
        im.load()
        return True
    except Exception:
        return False


def _try_text_snippet(p):
    """Bilinmeyen uzanti duz metin mi? (pdf/docx binary ise reddet) -> metin ya da ""."""
    try:
        t = Path(str(p)).read_text(encoding="utf-8", errors="replace")[:4000]
    except Exception:
        return ""
    if not t or "\x00" in t:
        return ""
    printable = sum(1 for c in t if c.isprintable() or c in "\n\r\t")
    if printable / len(t) < 0.85:
        return ""
    return t
# -*- coding: utf-8 -*-
# PARCA 03b: gezgin (dizin+dosya listeleme) + kaynak toplama
SKIP_DIRS = {"hf_cache", "matched", "cache", "__pycache__", ".venv", ".git"}


def _isdir(p: Path):
    """Windows korumali dizinlerde (System Volume Information vb.) stat
    EACCES firlatir — tum listeyi cokertmemek icin yutar."""
    try:
        return p.is_dir()
    except OSError:
        return False


def _isfile(p: Path):
    try:
        return p.is_file()
    except OSError:
        return False


def _exists(p: Path):
    try:
        return p.exists()
    except OSError:
        return False


def _stat_size(p: Path):
    try:
        return p.stat().st_size
    except OSError:
        return 0


def _drive_not_ready_msg(s: str):
    t = s.replace("\\", "/")
    if re.match(r"^[A-Za-z]:/?$", t):
        return (f"{t} su an erisilemiyor — surucude disk yok, "
                "baglanti kesik veya izin yok.")
    m = re.match(r"^/mnt/([A-Za-z])/?$", t)
    if m:
        return (f"{m.group(1).upper()}:/ su an WSL icinde erisilemiyor — "
                "surucunun Windows'ta takili oldugunu dogrulayin, "
                "gerekirse Windows cmd'de 'wsl --shutdown' ile WSL'yi "
                "yeniden baslatip tekrar deneyin.")
    return ""


def _safe_list_dir(pd: Path):
    try:
        return sorted(pd.iterdir(), key=lambda x: x.name.lower())
    except (PermissionError, OSError):
        return []


def _nt_drive_list():
    """Windows suruculeri: bos DVD / hazir olmayan surucu bile listede
    gorunur (isaretli), tiklaninca anlasilir hata doner."""
    import string
    letters = []
    try:
        import ctypes
        buf = ctypes.create_unicode_buffer(256)
        n = ctypes.windll.kernel32.GetLogicalDriveStringsW(256, buf)
        if n:
            for part in "".join(buf[:n]).split("\x00"):
                if (len(part) >= 2 and part[1] == ":"
                        and part[0].upper() in string.ascii_uppercase):
                    letters.append(part[0].upper())
    except Exception:
        letters = []
    if not letters:
        letters = list(string.ascii_uppercase)
    out = []
    for letter in dict.fromkeys(letters):
        try:
            d = Path(f"{letter}:/")
            ready = _isdir(d)
            pstr = str(d)
        except Exception:
            pstr = f"{letter}:/"
            ready = False
        out.append({"name": f"{letter}:/" + ("" if ready else " (hazir degil)"),
                    "path": pstr, "type": "drive", "ready": ready})
    return out


def _list_drives():
    out = []
    if os.name == "nt":
        return _nt_drive_list()
    try:
        items = sorted(Path("/mnt").iterdir())
    except OSError:
        items = []
    for d in items:
        if len(d.name) != 1 or not d.name.isalpha():
            continue
        try:
            ok = _isdir(d)
        except OSError:
            ok = False
        out.append({"name": d.name.upper() + ":/" + ("" if ok else " (hazir degil)"),
                    "path": str(d), "type": "drive", "ready": bool(ok)})
    try:
        if Path("/").is_dir():
            out.append({"name": "/ (kok)", "path": "/", "type": "drive", "ready": True})
    except OSError:
        pass
    return out


def browse_entries(path_str, need="any"):
    """need: any|media|dir — gezgin satirlari: dirs + files (+ ust)."""
    p = (path_str or "").strip()
    if not p:
        return {"path": "", "parent": "", "dirs": [], "files": [],
                "drives": _list_drives(), "need": need}
    pd = Path(to_wsl_path(p))
    if not _isdir(pd):
        return {"error": _drive_not_ready_msg(str(pd)) or f"Dizin degil: {pd}"}
    dirs, files = [], []
    for e in _safe_list_dir(pd):
        if e.name.startswith("."):
            continue
        if _isdir(e):
            if e.name in SKIP_DIRS:
                continue
            n = 0
            try:
                for j, c in enumerate(e.iterdir()):
                    if j >= 400:
                        n = 400
                        break
                    if _isfile(c) and c.suffix.lower() in MEDIA_EXT:
                        n += 1
            except (PermissionError, OSError):
                pass
            dirs.append({"name": e.name, "path": str(e), "type": "dir", "n": n})
        elif _isfile(e):
            suf = e.suffix.lower()
            if need == "dir":
                continue
            # need==media dahil TUM dosyalar listelenir; modelin gomemedigi
            # bicimler tarama asamasinda acik mesajla atlanir.
            try:
                st = e.stat()
                sz = st.st_size
            except OSError:
                sz = 0
            files.append({"name": e.name, "path": str(e), "type": "file",
                          "kind": media_kind(e.name), "size": sz})
            if len(files) >= BROWSE_FILE_CAP:
                files.append({"name": f"... ilk {BROWSE_FILE_CAP} dosya", "path": "",
                              "type": "note", "kind": "other", "size": 0})
                break
    return {"path": str(pd), "win": win_of(str(pd)),
            "parent": "" if str(pd) == "/" else str(pd.parent),
            "win_parent": "" if str(pd) == "/" else win_of(str(pd.parent)),
            "dirs": dirs, "files": files, "need": need}


def collect_media(root: Path, recursive=True):
    out = []
    try:
        it = root.rglob("*") if recursive else root.iterdir()
        items = sorted(it, key=lambda x: str(x).lower())
    except OSError:
        return out
    for p in items:
        try:
            is_f = p.is_file()
        except OSError:
            continue
        if not is_f or p.suffix.lower() not in MEDIA_EXT:
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if any(part.startswith(".") for part in p.parts):
            continue
        out.append(str(p))
        if len(out) >= MAX_CANDIDATES:
            break
    return out


def collect_url_media(url: str):
    import requests
    from bs4 import BeautifulSoup
    r = requests.get(url, headers=UA, timeout=30)
    r.raise_for_status()
    ctype = (r.headers.get("content-type") or "").lower()
    if ctype.startswith(("image/", "video/", "audio/")):
        return [url]
    soup = BeautifulSoup(r.text, "html.parser")
    found, seen = [], set()

    def add(u):
        if not u:
            return
        u = urllib.parse.urljoin(url, u.strip())
        if not u.startswith(("http://", "https://")) or u in seen:
            return
        seen.add(u)
        found.append(u)

    for tag in soup.find_all(["img", "source", "video", "audio", "meta", "a", "link"]):
        for attr in ("src", "data-src", "data-original", "href", "content"):
            try:
                v = tag.get(attr)
            except Exception:
                v = None
            if not v:
                continue
            if attr == "content" and tag.get("property") not in (
                    "og:image", "twitter:image", "og:video", "og:audio"):
                continue
            if attr == "href" and tag.name == "a":
                pl = urllib.parse.urlparse(urllib.parse.urljoin(url, v)).path.lower()
                if Path(pl).suffix not in MEDIA_EXT:
                    continue
            add(v)
        for ssattr in ("srcset", "data-srcset"):
            ss = tag.get(ssattr)
            if ss:
                for part in ss.split(","):
                    add(part.strip().split(" ")[0])
    return found[:MAX_CANDIDATES]


def load_image_bytes(data: bytes):
    from PIL import Image
    im = Image.open(io.BytesIO(data))
    im.load()
    if im.mode != "RGB":
        im = im.convert("RGB")
    return im
# -*- coding: utf-8 -*-
# PARCA 04a: sorgu ve aday birimleri (video kare / ses segment acilimi)
def build_query_units(cfg, job_tmp):
    """Sorgu -> units[(kind,payload)] + gosterim listesi."""
    texts = [t for t in (cfg.get("texts") or []) if str(t).strip()]
    paths = [p for p in (cfg.get("query_paths") or []) if str(p).strip()]
    ups = cfg.get("uploads") or []
    units, shown = [], []
    if texts:
        blob = "\n".join(str(t).strip()[:500] for t in texts[:5])
        units.append(("text", blob))
        shown.append({"kind": "text", "label": blob[:120], "thumb": ""})
    qdir = (cfg.get("query_dir") or "").strip()
    if qdir:
        rd = Path(to_wsl_path(qdir))
        if not _isdir(rd):
            raise RuntimeError(
                _drive_not_ready_msg(str(rd))
                or f"Sorgu dizini bulunamadi: '{qdir}' -> '{rd}'")
        paths += collect_media(rd)[:MAX_QUERY_FILES]
    for u in ups[:MAX_QUERY_FILES]:
        p = job_tmp / ("up_" + Path(u["name"]).name.replace("/", "_"))
        p.write_bytes(u["bytes"])
        paths.append(str(p))
    paths = paths[:MAX_QUERY_FILES]
    audio_ok = probe_audio_form(get_model()) is not None
    skipped = []
    for p in paths:
        _stop()
        k = media_kind(p)
        if k == "url":
            try:
                u = read_url_file(to_wsl_path(p))
            except Exception as e:
                skipped.append(Path(p).name + " (url okunamadi)")
                continue
            units.append(("text", u))
            shown.append({"kind": "url", "label": u[:120], "thumb": ""})
        elif k == "text":
            try:
                t = Path(p).read_text(encoding="utf-8", errors="replace")[:4000]
            except Exception:
                t = Path(p).name
            units.append(("text", t))
            shown.append({"kind": "text", "label": Path(p).name + ": " + t[:100], "thumb": ""})
        elif k == "image":
            units.append(("image", str(p)))
            shown.append({"kind": "image", "label": Path(p).name, "thumb": "",
                          "fs": "query", "path": str(p)})
        elif k == "video":
            fr = extract_video_frames(p, job_tmp / ("q_" + Path(p).stem[:24]),
                                      fps=FRAME_FPS, cap=MAX_FRAMES_PER_VIDEO)
            if not fr:
                skipped.append(Path(p).name)
                continue
            for t, fp in fr:
                units.append(("image", fp))
            shown.append({"kind": "video", "label": f"{Path(p).name} ({len(fr)} kare)",
                          "thumb": "", "fs": "query", "path": str(p)})
        elif k == "audio":
            if not audio_ok:
                skipped.append(Path(p).name + " (ses destegi kapali)")
                continue
            sg = audio_segments(p, job_tmp / ("qa_" + Path(p).stem[:24]))
            if not sg:
                skipped.append(Path(p).name)
                continue
            for t, fp in sg:
                units.append(("audio", fp))
            shown.append({"kind": "audio", "label": f"{Path(p).name} ({len(sg)} parcax{AUDIO_SEG_SEC}sn)",
                          "thumb": "", "fs": "query", "path": str(p)})
        else:
            # bilinmeyen uzanti: once gorsel dene (PIL/HEIC dahil), sonra duz metin
            if _try_image_unit(str(p)):
                units.append(("image", str(p)))
                shown.append({"kind": "image", "label": Path(p).name, "thumb": "",
                              "fs": "query", "path": str(p)})
            else:
                t = _try_text_snippet(str(p))
                if t:
                    units.append(("text", t))
                    shown.append({"kind": "text", "label": Path(p).name + ": " + t[:100],
                                  "thumb": ""})
                else:
                    skipped.append(Path(p).name + " (desteklenmeyen bicim)")
    return units, shown, skipped
# -*- coding: utf-8 -*-
# PARCA 04b: aday toplama + genisletme + tarama
def gather_candidates(cfg, job_tmp):
    cands = []  # (gosterim adi, yerel yol/None, url?, tur)
    stype = cfg.get("source_type") or "dir"
    if stype == "dir":
        sd = Path(to_wsl_path(cfg.get("source") or ""))
        if not _isdir(sd):
            raise RuntimeError(
                _drive_not_ready_msg(str(sd))
                or f"Kaynak dizin bulunamadi: '{cfg.get('source')}' -> '{sd}'")
        for p in collect_media(sd):
            cands.append((Path(p).name, p, "", media_kind(p)))
    elif stype == "files":
        for p in (cfg.get("source_paths") or [])[:MAX_CANDIDATES]:
            q = Path(to_wsl_path(p))
            if _isfile(q):  # uzanti kisiti yok: gomulebilen bicimler asagida ayiklanir
                cands.append((q.name, str(q), "", media_kind(q.name)))
        updir = job_tmp / "srcup"
        for i, u in enumerate((cfg.get("source_uploads") or [])[:MAX_CANDIDATES]):
            p = updir / ("su_{:03d}_".format(i) + Path(u["name"]).name.replace("/", "_"))
            try:
                updir.mkdir(parents=True, exist_ok=True)
                p.write_bytes(u["bytes"])
            except OSError:
                continue
            cands.append((u["name"], str(p), "", media_kind(u["name"])))
    elif stype == "url":
        src_u = (cfg.get("source") or "").strip()
        if src_u.lower().endswith(".url"):
            src_u = read_url_file(to_wsl_path(src_u))
        urls = collect_url_media(src_u)
        setj(phase=f"Baglantida {len(urls)} medya bulundu, indiriliyor...")
        import requests
        dl = job_tmp / "dl"
        dl.mkdir(parents=True, exist_ok=True)
        for i, u in enumerate(urls):
            _stop()
            try:
                rr = requests.get(u, headers=UA, timeout=20)
                if rr.status_code != 200 or len(rr.content) < 512:
                    continue
                suf = Path(urllib.parse.urlparse(u).path).suffix.lower() or ".jpg"
                if suf not in MEDIA_EXT:
                    ct = (rr.headers.get("content-type") or "").lower()
                    suf = (".jpg" if "image" in ct else
                           ".mp4" if "video" in ct else
                           ".mp3" if "audio" in ct else ".jpg")
                p = dl / f"{i:04d}{suf}"
                p.write_bytes(rr.content)
                if media_kind(p.name) == "other":
                    try:
                        load_image_bytes(rr.content)
                    except Exception:
                        continue
                cands.append((u, str(p), u, media_kind(p.name)))
            except Exception:
                continue
            if len(cands) >= MAX_CANDIDATES:
                break
    return cands


def expand_candidate(c, job_tmp, idx, audio_ok):
    """Aday -> alt birimler [(kind,payload,t,etiket)]."""
    name, path, url, kind = c
    if kind == "image":
        return [("image", str(path), 0.0, "")]
    if kind == "text":
        try:
            t = Path(str(path)).read_text(encoding="utf-8", errors="replace")[:4000]
        except Exception:
            t = name
        return [("text", t, 0.0, "")]
    if kind == "video":
        fr = extract_video_frames(str(path), job_tmp / f"c{idx:04d}",
                                  fps=FRAME_FPS, cap=MAX_FRAMES_PER_VIDEO)
        if not fr:
            return []
        step = max(1, len(fr) // MAX_FRAMES_PER_VIDEO)
        return [("image", fp, t, f"t={fmt_time(t)}") for t, fp in fr[::step]]
    if kind == "audio" and audio_ok:
        sg = audio_segments(str(path), job_tmp / f"ca{idx:04d}")
        return [("audio", fp, t, f"t={fmt_time(t)}") for t, fp in sg]
    if kind in ("other", "url"):
        # bilinmeyen bicim: gorsel dene, olmazsa duz metin dene
        if _try_image_unit(str(path)):
            return [("image", str(path), 0.0, "")]
        t = _try_text_snippet(str(path))
        if t:
            return [("text", t, 0.0, "")]
    return []
# -*- coding: utf-8 -*-
# PARCA 04c: run_scan — skor = aday_max(sorgu_max(cosine))
def run_scan(cfg):
    """Tarama — durdurulabilir + sonuclar akiskan yayinlanir.

    Adaylar 16'lik partiler halinde gomulur; her partiden sonra skoru
    esigi gecen adaylar aninda kaydedilip /api/status uzerinden yayinlanir.
    """
    import numpy as np
    job = cfg["job"]
    job_tmp = BASE / "cache" / f"job_{job}"
    job_tmp.mkdir(parents=True, exist_ok=True)
    out_dir = BASE / "matched" / job
    out_dir.mkdir(parents=True, exist_ok=True)
    results, shown, units = [], [], []
    saved_seq = 0
    thr = float(cfg.get("threshold") or 0.5)
    top_k = int(cfg.get("top_k") or 50)
    try:
        setj(state="running", phase="Model yukleniyor (ilk seferde ~1,5 GB iner)...",
             done=0, total=1, found=0, error=None, results=[], refs=[],
             src=[], query="", units=0, out_dir=str(out_dir))
        _stop()
        model = get_model()
        _stop()
        setj(phase="Sorgu hazirlaniyor...")
        units, shown, skipped = build_query_units(cfg, job_tmp)
        if not units:
            raise RuntimeError("Sorgu bos — metin yazin veya resim/video/ses secin"
                               + (f" (atlanan: {', '.join(skipped[:3])})" if skipped else ""))
        setj(phase=f"Sorgu gomuluyor ({len(units)} birim)...", total=len(units), done=0)
        q_embs = encode_units(model, units)
        _stop()
        setj(phase="Adaylar toplaniyor...")
        cands = gather_candidates(cfg, job_tmp)
        if not cands:
            raise RuntimeError("Kaynakta hic medya bulunamadi")
        audio_ok = probe_audio_form(model) is not None

        def _publish(ci, score, t, lab):
            """Skoru esigi gecen adayi aninda kaydet + yayinla (top_k, skor sirali)."""
            nonlocal saved_seq
            if score < thr:
                return
            if len(results) >= top_k and score <= results[-1]["score"]:
                return
            name, path, url, kind = cands[ci]
            saved_seq += 1
            safe = re.sub(r"[^\w\-. ]+", "_", Path(name).name)[:80]
            if not url:
                keep = out_dir / f"{saved_seq:03d}_sim{score:.3f}_{safe}"
            else:
                keep = out_dir / f"{saved_seq:03d}_sim{score:.3f}_{Path(str(path)).suffix or '.jpg'}"
            try:
                shutil.copyfile(path, keep)
            except Exception:
                keep = None
            th = BASE / "cache" / "thumbs" / f"{job}_{saved_seq}.jpg"
            try:
                file_thumb(path if path else job_tmp / "dl" / f"{ci:04d}.jpg", th)
            except Exception:
                pass
            detail = lab
            if kind == "video" and not detail:
                detail = f"t={fmt_time(t)}"
            entry = {"i": saved_seq, "score": round(score, 4),
                     "name": (Path(name).name if not url else name[:120]),
                     "kind": kind, "t": round(float(t or 0), 1),
                     "detail": detail,
                     "saved": keep.name if keep else "",
                     "thumb": f"/api/thumb/{job}/{saved_seq}",
                     "full": (f"/api/matchfile/{job}/{keep.name}" if keep else ""),
                     "src": path or "", "url": url or ""}
            pos = 0
            while pos < len(results) and results[pos]["score"] >= entry["score"]:
                pos += 1
            results.insert(pos, entry)
            if len(results) > top_k:
                ev = results.pop()  # en zayif: diskten de sil
                if ev.get("saved"):
                    try:
                        (out_dir / ev["saved"]).unlink()
                    except OSError:
                        pass
                try:
                    (BASE / "cache" / "thumbs" / f"{job}_{ev['i']}.jpg").unlink()
                except OSError:
                    pass
            # akiskan yayin: durum uzerinden aninda gorunur
            setj(results=list(results), found=len(results),
                 phase=f"Adaylar taraniyor — {len(results)} eslesme")

        BATCH = 16  # aday basina parti (gomme + yayin araligi)
        exp_done = 0
        setj(phase=f"Adaylar taraniyor (0/{len(cands)}) — 0 eslesme",
             total=len(cands), done=0)
        for b0 in range(0, len(cands), BATCH):
            _stop()
            batch = cands[b0:b0 + BATCH]
            flat, owner = [], []
            for j, c in enumerate(batch):
                _stop()
                ci = b0 + j
                for (k, pay, t, lab) in expand_candidate(c, job_tmp, ci, audio_ok):
                    flat.append((k, pay))
                    owner.append((ci, t, lab))
                exp_done += 1
                setj(done=exp_done, total=len(cands),
                     phase=f"Adaylar taraniyor ({exp_done}/{len(cands)}) — {len(results)} eslesme")
            if not flat:
                continue
            _stop()
            arr = encode_units(model, flat, track=False)
            sims = arr @ q_embs.T
            best = sims.max(axis=1)
            per = {}
            for r, (ci, t, lab) in enumerate(owner):
                s = float(best[r])
                if ci not in per or s > per[ci][0]:
                    per[ci] = (s, t, lab)
            for ci in sorted(per, key=lambda i: -per[i][0]):
                _stop()
                s, t, lab = per[ci]
                _publish(ci, s, t, lab)
            setj(done=exp_done, total=len(cands),
                 phase=f"Adaylar taraniyor ({exp_done}/{len(cands)}) — {len(results)} eslesme")
        qlabel = "; ".join([s.get("label", "") for s in shown][:3])[:200]
        setj(state="done", phase=f"Tamam — {len(results)} eslesme (esik {thr})",
             found=len(results), results=list(results), out_dir=str(out_dir),
             refs=[], query=qlabel, src=[s.get("label", "") for s in shown],
             units=len(units))
    except ScanStopped:
        qlabel = "; ".join([s.get("label", "") for s in shown][:3])[:200]
        setj(state="stopped", phase=f"Durduruldu — {len(results)} eslesme bulundu",
             found=len(results), results=list(results), out_dir=str(out_dir),
             refs=[], query=qlabel, src=[s.get("label", "") for s in shown],
             units=len(units), error=None)
    except Exception as e:
        msg = str(e)
        if ("EmbeddingGemma2Processor" in msg or "trust_remote_code" in msg.lower()) \
                and "ASIL HATA" not in msg:
            # teshis edilemediyse bilinen kok nedeni soyle (torchvision)
            msg += (" | COZUM: buyuk olasilikla torchvision eksik — "
                    "winvenv\\Scripts\\pip install torchvision==0.29.1 "
                    "sonra sayfayi yenileyin.")
        setj(state="error", phase="Hata", error=msg[:600],
             results=list(results), found=len(results), out_dir=str(out_dir))
# -*- coding: utf-8 -*-
# PARCA 05a: API — gezgin + onizleme + durum
@app.get("/health")
def health():
    return jsonify({"ok": True, "v": APP_VER})


@app.get("/api/pickdir")
def api_pickdir():
    """Windows Explorer klasor secici — modern diyalog (OpenFileDialog hilesi).

    Klasor secicisi tam bir Explorer penceresi acar; secim Tamam'a basinca
    doner. ?dry=1 testte dialog acmadan dogrular.
    """
    if request.args.get("dry"):
        return jsonify({"ok": True, "dry": True})
    import shutil as _sh
    ps = _sh.which("powershell.exe") or _sh.which("powershell")
    if not ps:
        return jsonify({"error": "PowerShell bulunamadi — klasor yolunu elle yazabilirsiniz"}), 501
    script = (
        "Add-Type -AssemblyName System.Windows.Forms;"
        "$d=New-Object System.Windows.Forms.OpenFileDialog;"
        "$d.ValidateNames=$false;$d.CheckFileExists=$false;"
        "$d.Title='Klasor secin - icine girip Tamama basin';"
        "$d.FileName='Klasor secin';"
        "if($d.ShowDialog() -eq 'OK'){Split-Path $d.FileName}"
    )
    kwargs = {"capture_output": True, "text": True, "timeout": 300}
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    try:
        r = subprocess.run([ps, "-NoProfile", "-Command", script], **kwargs)
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Secim zaman asimi (5 dk) — tekrar deneyin"}), 504
    except Exception as e:
        return jsonify({"error": "Secici calistirilamadi: " + str(e)[:200]
                        + " — klasor yolunu elle yazabilirsiniz"}), 500
    lines = (r.stdout or "").strip().splitlines()
    path = lines[-1].strip() if lines else ""
    if not path:
        if r.returncode != 0:
            return jsonify({"error": "Secici acilamadi: " + (r.stderr or "").strip()[:200]}), 502
        return jsonify({"cancelled": True})  # kullanici Vazgec'e basti
    return jsonify({"path": path, "win": path})


@app.post("/api/scan")
def api_scan():
    with JOB_LOCK:
        if JOB["state"] == "running":
            return jsonify({"error": "Devam eden tarama var — bitmesini bekleyin"}), 409
    def texts(key):
        v = request.form.get(key) or ""
        return [x for x in [s.strip() for s in v.split("\n")] if x]
    cfg = {
        "source_type": (request.form.get("source_type") or "dir"),
        "source": (request.form.get("source") or "").strip(),
        "source_paths": [s for s in (request.form.get("source_paths") or "").split("\n") if s.strip()],
        "query_paths": [s for s in (request.form.get("query_paths") or "").split("\n") if s.strip()],
        "query_dir": (request.form.get("query_dir") or "").strip(),
        "texts": texts("query_text"),
        "threshold": request.form.get("threshold") or 0.5,
        "top_k": request.form.get("top_k") or 50,
        "job": time.strftime("%Y%m%d_%H%M%S") + "_" + str(os.getpid())[-4:],
        "uploads": [],
        # eski arayuz uyumu:
        "ref_dir": (request.form.get("ref_dir") or "").strip(),
    }
    if cfg["ref_dir"] and not cfg["query_dir"] and not cfg["query_paths"]:
        cfg["query_dir"] = cfg["ref_dir"]
    if cfg["source_type"] not in ("dir", "files", "url"):
        cfg["source_type"] = "dir"
    if not cfg["source"] and cfg["source_type"] in ("dir", "url"):
        return jsonify({"error": "Kaynak bos — gezginden dizin/dosya secin veya baglanti yazin"}), 400
    for f in request.files.getlist("ref_files"):
        if f and f.filename:
            cfg["uploads"].append({"name": f.filename, "bytes": f.read()})
    for f in request.files.getlist("query_files"):
        if f and f.filename:
            cfg["uploads"].append({"name": f.filename, "bytes": f.read()})
    # Windows Explorer ile secilen KAYNAK dosyalari (yol gerekmez, bayt gelir)
    src_uploads = []
    for f in request.files.getlist("source_files"):
        if f and f.filename:
            src_uploads.append({"name": f.filename, "bytes": f.read()})
    cfg["source_uploads"] = src_uploads
    if cfg["source_type"] == "files" and not cfg["source_paths"] and not src_uploads:
        return jsonify({"error": "Kaynak dosya secilmedi — Windows Explorer'dan secin veya gezginden isaretleyin"}), 400
    if not cfg["texts"] and not cfg["query_paths"] and not cfg["query_dir"] and not cfg["uploads"]:
        return jsonify({"error": "Sorgu bos — metin yazin veya dosya/dizin secin"}), 400
    STOP.clear()  # onceki durdurma bayragini sifirla
    # onceki taramanin sonuclarini aninda temizle (eski sonuc gorunmesin)
    setj(state="running", phase="Kuyruga alindi...", error=None,
         results=[], found=0, out_dir="", query="", units=0)
    threading.Thread(target=run_scan, args=(cfg,), daemon=True).start()
    return jsonify({"ok": True, "job": cfg["job"]})


@app.post("/api/stop")
def api_stop():
    """Aktif taramayi derhal durdurur — calisan surec ilk kontrolde durur."""
    with JOB_LOCK:
        st = JOB["state"]
    if st != "running":
        return jsonify({"error": "Devam eden tarama yok"}), 409
    STOP.set()
    with JOB_LOCK:
        JOB["phase"] = "Durduruluyor..."
    return jsonify({"ok": True})


@app.get("/api/status")
def api_status():
    with JOB_LOCK:
        return jsonify({k: JOB[k] for k in
                        ("state", "phase", "done", "total", "found",
                         "results", "error", "out_dir", "refs", "src",
                         "query", "units")})


@app.get("/api/browse")
def api_browse():
    p = (request.args.get("path") or "").strip()
    need = (request.args.get("need") or "any").strip()
    if need not in ("any", "media", "dir"):
        need = "any"
    try:
        r = browse_entries(p, need)
    except Exception as e:
        return jsonify({"error": str(e)[:300]}), 500
    if "error" in r:
        return jsonify(r), 404
    return jsonify(r)


@app.get("/api/dirs")
def api_dirs_compat():
    """Eski gezgin: sadece dizinler (sonuc browse ile ayni sekilde)."""
    p = (request.args.get("path") or "").strip()
    try:
        r = browse_entries(p, "dir" if p else "any")
    except Exception as e:
        return jsonify({"error": str(e)[:300]}), 500
    if "error" in r:
        return jsonify(r), 404
    if not p:
        return jsonify({"path": "", "parent": "",
                        "drives": [{"name": d["name"], "path": d["path"]} for d in r["drives"]]})
    dirs = [{"name": d["name"], "path": d["path"], "n": d["n"]} for d in r["dirs"]]
    return jsonify({"path": r["path"], "parent": r["parent"], "dirs": dirs})
# -*- coding: utf-8 -*-
# PARCA 05b: dosya onizleme + indirme + thumb
def _resolve_fs(path_str):
    p = Path(to_wsl_path(path_str or ""))
    if not _isfile(p):  # uzanti kisiti yok: onizleme/indirme her dosyaya acik
        return None
    return p


@app.get("/api/file")
def api_file():
    """Gezginde secili dosyanin dogrudan onizlemesi (resim/ses/video/metin)."""
    p = _resolve_fs(request.args.get("path") or "")
    if not p:
        return ("bulunamadi", 404)
    if p.suffix.lower() in URL_EXT:
        try:
            u = read_url_file(p)
        except Exception as e:
            return (str(e)[:200], 400)
        return jsonify({"url": u})
    mt, _e = mimetypes.guess_type(p.name)
    if p.suffix.lower() in TXT_EXT:
        mt = "text/plain; charset=utf-8"
    try:
        return send_file(p, mimetype=mt or "application/octet-stream",
                         as_attachment=False, download_name=p.name)
    except Exception:
        return ("okunamadi", 500)


@app.get("/api/filemeta")
def api_filemeta():
    p = _resolve_fs(request.args.get("path") or "")
    if not p:
        return jsonify({"error": "dosya yok"}), 404
    if p.suffix.lower() in URL_EXT:
        try:
            u = read_url_file(p)
        except Exception as e:
            return jsonify({"error": str(e)[:200]}), 400
        return jsonify({"name": p.name, "path": str(p), "win": win_of(str(p)),
                        "kind": "url", "size": _stat_size(p),
                        "dur": 0.0, "url": u})
    kind = media_kind(p.name)
    info = {"name": p.name, "path": str(p), "win": win_of(str(p)),
            "kind": kind, "size": 0, "dur": 0.0}
    try:
        info["size"] = p.stat().st_size
    except OSError:
        pass
    if kind in ("video", "audio"):
        info["dur"] = round(video_duration(p) or 0.0, 1)
        if kind == "audio" and not info["dur"]:
            try:
                import soundfile as sf
                d, sr = sf.read(str(p), always_2d=True)
                info["dur"] = round(len(d) / float(sr or 1), 1)
            except Exception:
                pass
    if kind == "text":
        try:
            info["preview"] = p.read_text(encoding="utf-8", errors="replace")[:600]
        except Exception:
            info["preview"] = ""
    return jsonify(info)


@app.get("/api/thumb/<job>/<int:i>")
def api_thumb(job, i):
    p = BASE / "cache" / "thumbs" / f"{job}_{i}.jpg"
    if p.exists():
        return send_file(p, mimetype="image/jpeg")
    return ("yok", 404)


@app.get("/api/matchfile/<job>/<name>")
def api_matchfile(job, name):
    """Eslesen dosyanin tam boyu — sonuc kartindaki buyuk onizleme icin.
    Path traversal'a karsi sadece duz dosya adi kabul edilir."""
    if not re.fullmatch(r"[\w\-. ]{1,160}", name) or not re.fullmatch(r"[\w\-]{1,80}", job):
        return ("yok", 404)
    p = BASE / "matched" / job / name
    if not p.is_file():
        return ("yok", 404)
    mt, _e = mimetypes.guess_type(p.name)
    return send_file(p, mimetype=mt or "application/octet-stream",
                     as_attachment=False, download_name=p.name)


@app.get("/api/refthumb/<job>/<int:i>")
def api_refthumb(job, i):
    p = BASE / "cache" / "thumbs" / f"{job}_ref{i}.jpg"
    if p.exists():
        return send_file(p, mimetype="image/jpeg")
    p2 = BASE / "cache" / "thumbs" / f"{job}_{i}.jpg"
    if p2.exists():
        return send_file(p2, mimetype="image/jpeg")
    return ("yok", 404)
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# PARCA 06: arayuz — index.html dosyasindan servis edilir (bozuk inline-JS duzeltmesi)
# Tum tuslar gercek <button> + addEventListener ile baglidir; inline onclick ile
# yol interpolasyonu YOKTUR (eski eq()/winOf() kacis hatalari tarihe karisti).
def _load_index():
    fp = BASE / "index.html"
    try:
        return fp.read_text(encoding="utf-8")
    except Exception:
        return "<h1>index.html bulunamadi</h1>"


@app.get("/")
def index():
    return _load_index()


HTML = ""  # geriye uyumluluk: disaridan import eden kod kirilmasin


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    print(f"EmbeddingGemma 2 cok-modlu arama -> http://localhost:{PORT}  (HF_HOME={os.environ['HF_HOME']})")
    # acilis self-check: modelin ozel islemcisi + torchvision hazir mi?
    diag = _diag_processor_error()
    if diag:
        print(f"[DIAG] EmbeddingGemma2Processor HAZIR DEGIL: {diag}")
        if "No module named" in diag:
            pkg = diag.split("No module named")[-1].strip().strip("'\"").split(".")[0]
            print(f"[DIAG] EKSIK PAKET '{pkg}' — cozum: winvenv\\Scripts\\pip install {pkg}")
        else:
            print("[DIAG] cozum: winvenv\\Scripts\\pip install -r requirements.txt")
    else:
        try:
            import torchvision
            print(f"[DIAG] hazir: EmbeddingGemma2Processor OK, torchvision {torchvision.__version__}")
        except Exception as e:
            print(f"[DIAG] torchvision import hatasi: {e}")
    app.run(host="0.0.0.0", port=PORT, threaded=True, use_reloader=False)
