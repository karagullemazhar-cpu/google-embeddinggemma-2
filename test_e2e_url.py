# -*- coding: utf-8 -*-
"""URL kipi testi: HF sayfasindaki banner gorselini buldurmali + arayuz HTML."""
import io
import json
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import requests

APP = Path(__file__).resolve().parent
BASE = "http://127.0.0.1:5102"
PAGE = "https://huggingface.co/google/embeddinggemma-2"
BANNER = "https://ai.google.dev/gemma/images/embeddinggemma2_banner.png"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"}


def get(u, t=20):
    with urllib.request.urlopen(u, timeout=t) as r:
        return r.status, r.read()


def scan(fields, files):
    b = "----b" + uuid.uuid4().hex
    body = b""
    for k, v in fields.items():
        body += f"--{b}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode()
    for name, (fn, data, ct) in files:
        body += (f"--{b}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"{fn}\"\r\n"
                 f"Content-Type: {ct}\r\n\r\n").encode() + data + b"\r\n"
    body += f"--{b}--\r\n".encode()
    req = urllib.request.Request(BASE + "/api/scan", data=body,
                                 headers={"Content-Type": f"multipart/form-data; boundary={b}"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def wait_done(limit=420):
    t0, last = time.time(), ""
    while time.time() - t0 < limit:
        s = json.loads(get(BASE + "/api/status")[1])
        line = f"{s['state']:8s} {s['phase']} ({s['done']}/{s['total']})"
        if line != last:
            print("%5.0fs %s" % (time.time() - t0, line), flush=True)
            last = line
        if s["state"] in ("done", "error"):
            return s
        time.sleep(3)
    return {"state": "timeout"}


def main():
    fails = []
    st, html = get(BASE + "/")
    if st != 200 or "Görsel Eşleştirici" not in html.decode("utf-8", "replace"):
        fails.append(f"arayuz HTML sorunu: {st}")

    ref = requests.get(BANNER, headers=UA, timeout=30).content
    print("referans banner:", len(ref), "B")
    st, resp = scan({"source_type": "url", "source": PAGE, "ref_dir": "",
                     "threshold": "0.8", "top_k": "20"},
                    [("ref_files", ("banner.png", ref, "image/png"))])
    print("SCAN ->", st, resp)
    if st != 200:
        raise SystemExit("SCAN basarisiz")

    s = wait_done()
    if s["state"] != "done":
        fails.append("durum=" + s["state"] + " hata=" + str(s.get("error"))[:400])
    else:
        res = s.get("results", [])
        print("SONUCLAR:", json.dumps([(r["score"], r["name"][:90]) for r in res[:6]], ensure_ascii=False, indent=1))
        if not res:
            fails.append("hic eslesme yok")
        best = res[0] if res else {}
        name = best.get("name", "")
        if "embeddinggemma2_banner" not in name:
            fails.append("en iyi eslesme banner degil: " + name[:120])
        elif best.get("score", 0) < 0.98:
            fails.append(f"banner skoru beklenenik altinda: {best.get('score')}")
        # sonuc dosyasi kaydedilmis mi
        if s.get("out_dir") and not any(Path(s["out_dir"]).glob("*.png")) and not any(Path(s["out_dir"]).glob("*.jpg")):
            fails.append("out_dir bos gorunuyor: " + s["out_dir"])

    print()
    if fails:
        print("E2E_URL_TEST_FAIL:", " | ".join(fails))
        raise SystemExit(1)
    print("E2E_URL_TEST_OK: sayfadan gorseller indirildi, banner ~1.0 ile eslesti, arayuz 200")


if __name__ == "__main__":
    main()
