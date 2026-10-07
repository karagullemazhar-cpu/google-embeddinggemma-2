# -*- coding: utf-8 -*-
"""Uc-to-uc tarama testi: dizin kipi + esik + thumb. (Flask sunucusu 5102'de acik varsayilir)"""
import json
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

APP = Path(__file__).resolve().parent
BASE = "http://127.0.0.1:5102"


def get(u, t=15):
    with urllib.request.urlopen(u, timeout=t) as r:
        return r.status, r.read()


def wait_server():
    err = "yok"
    for _ in range(30):
        try:
            st, _b = get(BASE + "/", t=5)
            if st == 200:
                return True
        except Exception as e:
            err = str(e)[:120]
        time.sleep(1)
    raise SystemExit("SUNUCU YOK (5102): " + err)


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
    t0 = time.time()
    last = ""
    while time.time() - t0 < limit:
        st, body = get(BASE + "/api/status")
        s = json.loads(body)
        line = f"{s['state']:8s} {s['phase']} ({s['done']}/{s['total']})"
        if line != last:
            print("%5.0fs %s" % (time.time() - t0, line), flush=True)
            last = line
        if s["state"] in ("done", "error"):
            return s
        time.sleep(3)
    return {"state": "timeout"}


def main():
    wait_server()
    print("SUNUCU HAZIR")
    ref = (APP / "test_fixtures/refs/referans.jpg").read_bytes()
    st, resp = scan(
        {"source_type": "dir", "source": str(APP / "test_fixtures/scan"),
         "ref_dir": "", "threshold": "0.8", "top_k": "10"},
        [("ref_files", ("referans.jpg", ref, "image/jpeg"))])
    print("SCAN ->", st, resp)
    if st != 200:
        raise SystemExit("SCAN basarisiz")

    s = wait_done()
    fails = []
    if s["state"] != "done":
        fails.append("durum=" + s["state"] + " hata=" + str(s.get("error"))[:300])
    else:
        names = {r["name"]: r["score"] for r in s.get("results", [])}
        print("SONUCLAR:", json.dumps(names, ensure_ascii=False, indent=1))
        copy_score = next((v for k, v in names.items() if "birebir" in k), 0)
        if not copy_score or copy_score < 0.99:
            fails.append(f"birebir kopya skoru dusuk: {copy_score}")
        if any("alakasiz" in k for k in names):
            fails.append("alakasiz mavi ESLESI olarak cikti (esik calismiyor)")
        if not any("varyant" in k for k in names):
            print("not: varyant esigi gecmedi (bilgi)")
        if s.get("found", 0) < 1:
            fails.append("found < 1")
        # thumb
        try:
            st_t, data = get(BASE + s["results"][0]["thumb"])
            if st_t != 200 or len(data) < 200:
                fails.append(f"thumb sorunu: {st_t} {len(data)}B")
            else:
                print("THUMB OK", len(data), "B")
        except Exception as e:
            fails.append("thumb hata: " + str(e))

    print()
    if fails:
        print("E2E_DIR_TEST_FAIL:", " | ".join(fails))
        raise SystemExit(1)
    print("E2E_DIR_TEST_OK: kopya ~1.0 skorla yakalandi, alakasiz elendi, thumb calisiyor")


if __name__ == "__main__":
    main()
