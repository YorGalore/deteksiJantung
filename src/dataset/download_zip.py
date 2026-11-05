"""Unduh ZIP resmi MIMIC-IV-ECG per potongan paralel, dapat dilanjutkan (resume).

Jalankan dari root repo:
    python -u -m src.dataset.download_zip --workers 4

- Server mendukung Range dan berkas statis (ETag tetap), jadi tiap potongan ditulis ke posisinya.
- Potongan yang selesai dicatat di <zip>.done; menjalankan ulang melanjutkan, tidak mengulang dari nol.
- Potongan diminta dengan If-Range (ETag); jika berkas di server berubah, pengunduh berhenti.
- Maksimal 8 koneksi (batas PhysioNet). Ctrl+C menghentikan dengan cepat; hasil yang sudah ada aman.
- Tidak ada checksum resmi untuk ZIP; keutuhan dicek lewat ukuran, struktur ZIP, dan CRC saat ekstraksi.
Default keluaran: ../mimic_zip/mimic-iv-ecg-1.0.zip (di luar repo supaya tidak ikut Git).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

URL = "https://physionet.org/content/mimic-iv-ecg/get-zip/1.0/"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT.parent / "mimic_zip" / "mimic-iv-ecg-1.0.zip"
UA = {"User-Agent": "Mozilla/5.0 (zip-chunk-downloader)"}
MAX_WORKERS = 8


def probe(url: str) -> tuple[int, str]:
    """Ukuran total dan ETag dari permintaan Range 0-0."""
    req = urllib.request.Request(url, headers={**UA, "Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        if r.status != 206:
            raise RuntimeError(f"server tidak mendukung Range (status {r.status})")
        total = int(r.headers["Content-Range"].rsplit("/", 1)[1])
        etag = r.headers.get("ETag", "")
    return total, etag


def fetch_chunk(url: str, out: Path, start: int, end: int, total: int, etag: str,
                stop: threading.Event, retries: int = 5, on_bytes=None) -> None:
    want = end - start + 1
    last = None
    for attempt in range(retries):
        if stop.is_set():
            raise InterruptedError
        headers = {**UA, "Range": f"bytes={start}-{end}"}
        if etag:
            headers["If-Range"] = etag
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as r:
                if r.status != 206:
                    raise RuntimeError(f"status {r.status}: berkas di server berubah atau Range tidak dihormati")
                if r.headers["Content-Range"] != f"bytes {start}-{end}/{total}":
                    raise RuntimeError("Content-Range tidak cocok")
                got = 0
                with open(out, "r+b") as f:
                    f.seek(start)
                    while True:
                        if stop.is_set():
                            raise InterruptedError
                        block = r.read(256 * 1024)
                        if not block:
                            break
                        f.write(block)
                        got += len(block)
                        if on_bytes:
                            on_bytes(len(block))
                    f.flush()
                    os.fsync(f.fileno())
                if got != want:
                    raise OSError(f"potongan tidak lengkap: {got}/{want}")
                return
        except InterruptedError:
            raise
        except RuntimeError:
            raise
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            last = e
            time.sleep(min(30, 2 ** attempt))
    raise OSError(f"gagal setelah {retries} percobaan: {type(last).__name__}")


def run(url: str, out: Path, workers: int, chunk: int, restart: bool = False, quiet: bool = False) -> bool:
    workers = max(1, min(workers, MAX_WORKERS))
    out.parent.mkdir(parents=True, exist_ok=True)
    meta_p, done_p = Path(str(out) + ".meta"), Path(str(out) + ".done")
    total, etag = probe(url)
    meta = {"total": total, "etag": etag, "chunk": chunk}
    if restart:
        for p in (out, meta_p, done_p):
            p.unlink(missing_ok=True)
    if meta_p.exists():
        old = json.loads(meta_p.read_text())
        if old != meta:
            sys.exit(f"BERHENTI: berkas di server/pengaturan berbeda dari unduhan sebelumnya {old} vs {meta}. "
                     "Gunakan --restart hanya jika ingin mulai dari nol.")
    else:
        meta_p.write_text(json.dumps(meta))
    if not out.exists():
        with open(out, "wb") as f:
            f.truncate(total)
    elif out.stat().st_size != total:
        sys.exit("BERHENTI: ukuran berkas lokal tidak sama dengan ukuran di server.")

    n = (total + chunk - 1) // chunk
    done = set()
    if done_p.exists():
        done = {int(x) for x in done_p.read_text().split() if x.strip().isdigit()}
    todo = [i for i in range(n) if i not in done]
    if not quiet:
        print(f"total {total:,} byte, {n} potongan, sudah {len(done)}, sisa {len(todo)}, {workers} koneksi")

    stop = threading.Event()
    lock = threading.Lock()
    counter = {"bytes": 0}
    t0 = time.time()

    def on_bytes(k):
        with lock:
            counter["bytes"] += k

    def work(i):
        s = i * chunk
        e = min(total, s + chunk) - 1
        fetch_chunk(url, out, s, e, total, etag, stop, on_bytes=on_bytes)
        with lock:
            with open(done_p, "a") as f:
                f.write(f"{i}\n")
        return i

    pool = ThreadPoolExecutor(max_workers=workers)
    failed = None
    try:
        futures = [pool.submit(work, i) for i in todo]
        last_print = time.time()
        finished = 0
        for fut in as_completed(futures):
            try:
                fut.result()
                finished += 1
            except InterruptedError:
                pass
            except BaseException as e:  # kegagalan satu potongan: hentikan semua, simpan kemajuan
                failed = e
                stop.set()
                break
            if not quiet and time.time() - last_print >= 10:
                last_print = time.time()
                el = max(1e-6, time.time() - t0)
                rate = counter["bytes"] / el
                left = max(0, sum(min(chunk, total - i * chunk) for i in todo) - counter["bytes"])
                eta = left / rate / 3600 if rate else float("inf")
                print(f"  {len(done) + finished}/{n} potongan ({(len(done) + finished) / n * 100:.1f}%) | "
                      f"{rate / 1e6:.2f} MB/detik | sisa ~{eta:.1f} jam", flush=True)
    except BaseException:
        stop.set()
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown(wait=True, cancel_futures=True)
    if failed:
        print(f"BERHENTI karena kegagalan: {failed}. Jalankan ulang perintah yang sama untuk melanjutkan.")
        return False

    done = {int(x) for x in done_p.read_text().split() if x.strip().isdigit()}
    if len(done) != n:
        print(f"Belum lengkap: {len(done)}/{n}")
        return False
    ok = out.stat().st_size == total and zipfile.is_zipfile(out)
    if not quiet:
        print("selesai. ukuran cocok dan struktur ZIP terbaca:", ok)
        if ok:
            with zipfile.ZipFile(out) as z:
                names = z.namelist()
            print("jumlah anggota ZIP:", len(names), "| contoh awal jalur:", "/".join(names[0].split("/")[:2]))
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=URL)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--workers", type=int, default=4, help=f"koneksi paralel (maks {MAX_WORKERS})")
    ap.add_argument("--chunk-mib", type=int, default=16)
    ap.add_argument("--restart", action="store_true", help="hapus unduhan lama dan mulai dari nol")
    ap.add_argument("--max-restarts", type=int, default=100,
                    help="jika terputus, tunggu 60 detik lalu lanjutkan otomatis sebanyak ini")
    a = ap.parse_args(argv)
    chunk = a.chunk_mib * 1024 * 1024
    for k in range(a.max_restarts + 1):
        if run(a.url, a.out, a.workers, chunk, restart=a.restart and k == 0):
            return 0
        if k < a.max_restarts:
            print(f"menunggu 60 detik lalu melanjutkan otomatis (percobaan {k + 1}/{a.max_restarts})", flush=True)
            time.sleep(60)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())