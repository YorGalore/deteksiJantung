"""Unduh rekaman WFDB (.hea + .dat) dengan validasi ukuran, retry, dan tulis atomik.

Masalah yang diatasi:
- Server PhysioNet kadang memberi 502 / timeout -> dicoba ulang dengan jeda.
- Unduhan bisa terpotong tanpa error (mis. .dat 97.945 byte, seharusnya 120.000)
  -> ukuran .dat dibandingkan dengan nilai yang dihitung dari header (.hea).
- File yang sudah lengkap tidak diunduh ulang (tidak redundan).
"""
from __future__ import annotations

import time
import urllib.request
from pathlib import Path

import wfdb

BYTES_PER_SAMPLE_FMT16 = 2


def _path(base: Path, ext: str) -> Path:
    return base.parent / (base.name + ext)


def expected_dat_bytes(base: Path) -> int:
    """Ukuran .dat yang diharapkan, dihitung dari header. Hanya format 16-bit."""
    header = wfdb.rdheader(str(base))
    if any(f != "16" for f in header.fmt):
        raise ValueError(f"unsupported_fmt:{sorted(set(header.fmt))}")
    return int(header.sig_len) * int(header.n_sig) * BYTES_PER_SAMPLE_FMT16


def is_valid_record(base: Path) -> bool:
    """True jika .hea terbaca dan ukuran .dat sama dengan yang diharapkan."""
    hea, dat = _path(base, ".hea"), _path(base, ".dat")
    if not (hea.exists() and dat.exists()):
        return False
    try:
        return dat.stat().st_size == expected_dat_bytes(base)
    except Exception:
        return False


def _download(url: str, dest: Path, timeout: int = 60) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.parent / (dest.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "ECG-Research/1.0"})
    try:
        written = 0
        with urllib.request.urlopen(request, timeout=timeout) as response:
            declared = response.headers.get("Content-Length")
            with tmp.open("wb") as f:
                while True:
                    chunk = response.read(64 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    written += len(chunk)
        if written == 0:
            raise OSError("empty_response")
        if declared is not None and written != int(declared):
            raise OSError(f"truncated:{written}/{declared}")
        tmp.replace(dest)
    finally:
        if tmp.exists():
            tmp.unlink()


def fetch_record(base_url: str, rel: str, cache_dir: Path,
                 retries: int = 4, backoff: float = 3.0) -> tuple[bool, str]:
    """Pastikan rekaman `rel` (tanpa ekstensi) ada di cache dan lengkap.

    Mengembalikan (berhasil, pesan). Aman dipanggil dari banyak thread untuk
    rekaman yang berbeda.
    """
    base = cache_dir / rel
    if is_valid_record(base):
        return True, "cached"

    hea, dat = _path(base, ".hea"), _path(base, ".dat")
    last = ""
    for attempt in range(retries):
        try:
            if not hea.exists() or hea.stat().st_size == 0:
                _download(base_url + rel + ".hea", hea)
            try:
                expected = expected_dat_bytes(base)
            except Exception:
                hea.unlink(missing_ok=True)  # header rusak: unduh ulang
                raise
            dat.unlink(missing_ok=True)
            _download(base_url + rel + ".dat", dat)
            size = dat.stat().st_size
            if size != expected:
                dat.unlink(missing_ok=True)
                raise OSError(f"size_mismatch:{size}/{expected}")
            return True, "downloaded"
        except Exception as exc:
            last = f"{type(exc).__name__}: {str(exc)[:100]}"
            if attempt < retries - 1:
                time.sleep(backoff * (2 ** attempt))
    return False, last
