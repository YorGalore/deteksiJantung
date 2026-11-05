import hashlib
import http.server
import threading
import zipfile

import pytest

from src.dataset import download_zip as dz


def _server(payload, etag='"abc"', fail_every=0, fail_after=None):
    state = {"n": 0}

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            state["n"] += 1
            rng = self.headers.get("Range")
            s, e = (int(x) for x in rng.split("=")[1].split("-"))
            is_probe = (s, e) == (0, 0)
            bad = (fail_every and state["n"] % fail_every == 0) or (
                fail_after is not None and state["n"] > fail_after)
            if bad and not is_probe:
                self.send_response(500)
                self.end_headers()
                return
            e = min(e, len(payload) - 1)
            body = payload[s:e + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {s}-{e}/{len(payload)}")
            self.send_header("ETag", etag)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    srv.state = state
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _url(srv):
    return f"http://127.0.0.1:{srv.server_address[1]}/z"


@pytest.fixture
def zip_payload(tmp_path):
    p = tmp_path / "src.zip"
    with zipfile.ZipFile(p, "w", zipfile.ZIP_STORED) as z:
        z.writestr("root/files/a.bin", bytes(range(256)) * 20000)
        z.writestr("root/files/b.bin", b"x" * 300000)
    return p.read_bytes()


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def test_download_with_transient_failures_matches_source(tmp_path, zip_payload, monkeypatch):
    monkeypatch.setattr(dz.time, "sleep", lambda s: None)
    srv = _server(zip_payload, fail_every=5)
    out = tmp_path / "o" / "m.zip"
    assert dz.run(_url(srv), out, workers=4, chunk=64 * 1024, quiet=True)
    assert _sha(out.read_bytes()) == _sha(zip_payload)
    srv.shutdown()


def test_resume_continues_and_skips_done_chunks(tmp_path, zip_payload, monkeypatch):
    monkeypatch.setattr(dz.time, "sleep", lambda s: None)
    chunk = 64 * 1024
    n_chunks = (len(zip_payload) + chunk - 1) // chunk
    out = tmp_path / "m.zip"
    bad = _server(zip_payload, fail_after=4)          # server rusak setelah 3 potongan
    assert dz.run(_url(bad), out, 1, chunk, quiet=True) is False
    done_before = len(set((tmp_path / "m.zip.done").read_text().split()))
    assert 0 < done_before < n_chunks
    bad.shutdown()
    good = _server(zip_payload)
    assert dz.run(_url(good), out, 2, chunk, quiet=True)
    assert _sha(out.read_bytes()) == _sha(zip_payload)
    assert good.state["n"] == 1 + (n_chunks - done_before)   # 1 probe + hanya potongan yang belum ada
    good.shutdown()


def test_changed_etag_stops(tmp_path, zip_payload):
    out = tmp_path / "m.zip"
    srv = _server(zip_payload, etag='"v1"')
    dz.run(_url(srv), out, 1, 64 * 1024, quiet=True)
    srv.shutdown()
    srv2 = _server(zip_payload, etag='"v2"')
    with pytest.raises(SystemExit):
        dz.run(_url(srv2), out, 1, 64 * 1024, quiet=True)
    srv2.shutdown()


def test_workers_capped_at_eight():
    assert max(1, min(99, dz.MAX_WORKERS)) == 8


def test_main_resumes_automatically_until_success(monkeypatch):
    results = iter([False, False, True])
    calls = []
    monkeypatch.setattr(dz, "run", lambda *a, **k: (calls.append(k.get("restart")), next(results))[1])
    monkeypatch.setattr(dz.time, "sleep", lambda s: None)
    assert dz.main(["--restart", "--max-restarts", "5"]) == 0
    assert calls == [True, False, False]          # --restart hanya berlaku pada percobaan pertama


def test_main_gives_up_after_max_restarts(monkeypatch):
    monkeypatch.setattr(dz, "run", lambda *a, **k: False)
    monkeypatch.setattr(dz.time, "sleep", lambda s: None)
    assert dz.main(["--max-restarts", "2"]) == 1