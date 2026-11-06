#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SAM-Audio 分离工作台 —— 本地图形界面。

启动后在浏览器打开 http://127.0.0.1:8765
模型常驻内存，只需加载一次；任务串行执行（GPU 独占）。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
import traceback
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse, parse_qs

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from batch import SR, Separator, chunk_plan, decode, lowpass, slug  # noqa: E402


def load_audio(p: Path) -> np.ndarray:
    """已是 48k 单声道 wav 就直接读，省一次 ffmpeg 往返。"""
    if p.suffix.lower() == ".wav":
        import soundfile as sf
        x, sr = sf.read(str(p), dtype="float32", always_2d=False)
        if sr == SR:
            return x
    return decode(p)

MODEL_DIR = ROOT / "models" / "sam-audio-small"
WORK = ROOT / "ui_work"
UPLOADS = WORK / "uploads"
OUTPUTS = WORK / "outputs"
INDEX_HTML = ROOT / "web" / "index.html"
PORT = 8899
OVERLAP_SEC = 1.0

LOWPASS_CHOICES = {0: None, 3000: 3000.0, 2000: 2000.0, 1500: 1500.0}


# ---------------------------------------------------------------- 状态


class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.engine = "loading"          # loading | ready | error
        self.engine_note = "正在加载模型…"
        self.job = None                  # 当前或最近一次任务
        self.sep = None
        self.last = None                 # 最近一次上传，供页面刷新后恢复

    def snapshot(self):
        with self.lock:
            j = None
            if self.job:
                j = {
                    "id": self.job.id,
                    "status": self.job.status,
                    "progress": round(self.job.progress, 4),
                    "stage": self.job.stage,
                    "filename": self.job.filename,
                    "duration": round(self.job.duration, 2),
                    "prompts": self.job.prompts,
                    "lowpass": self.job.lowpass_fc,
                    "results": list(self.job.results),
                    "error": self.job.error,
                    "elapsed": round(time.time() - self.job.t0, 1) if self.job.t0 else 0,
                }
            return {
                "engine": self.engine,
                "engine_note": self.engine_note,
                "job": j,
                "last": self.last,
                "lowpass_choices": list(LOWPASS_CHOICES.keys()),
            }


STATE = State()


class Job:
    def __init__(self, filename: str, src: Path, prompts: list[str],
                 lowpass_fc: float | None, chunk_sec: float, duration: float):
        self.id = uuid.uuid4().hex[:12]
        self.filename = filename
        self.src = src
        self.prompts = prompts
        self.lowpass_fc = lowpass_fc
        self.chunk = chunk_sec
        self.duration = duration
        self.status = "queued"       # queued | running | done | error | cancelled
        self.progress = 0.0
        self.stage = "排队中"
        self.results: list[dict] = []
        self.error = None
        self.cancel = False
        self.t0 = 0.0


# ---------------------------------------------------------------- 引擎


def load_engine():
    try:
        t0 = time.time()
        sep = Separator(str(MODEL_DIR), "cuda", use_spans=False)
        with STATE.lock:
            STATE.sep = sep
            STATE.engine = "ready"
            STATE.engine_note = f"引擎就绪（{time.time() - t0:.0f}s）"
    except Exception as e:  # noqa: BLE001
        with STATE.lock:
            STATE.engine = "error"
            STATE.engine_note = f"模型加载失败: {e}"
        traceback.print_exc()


def spectrogram_png(target: np.ndarray, residual: np.ndarray, src: np.ndarray, out: Path):
    """三联频谱图：原音 / 目标 / 剩余。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def cut(x):
        return x[: SR * 20]

    panels = [("Original (mixed)", src), ("Target", target), ("Residual", residual)]
    fig, axes = plt.subplots(3, 1, figsize=(9, 5.4), sharex=True)
    for ax, (title, sig) in zip(axes, panels):
        if len(sig) < 1024:
            sig = np.zeros(4096, dtype=np.float32)
        ax.specgram(cut(sig), NFFT=1024, Fs=SR, noverlap=768, cmap="magma",
                    vmin=-120, vmax=-20)
        ax.set_title(title, fontsize=9, loc="left", color="#333")
        ax.set_ylabel("Hz", fontsize=7)
        ax.set_ylim(0, 12000)
        ax.tick_params(labelsize=6)
    axes[-1].set_xlabel("seconds", fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=110, facecolor="white")
    plt.close(fig)


def run_job(job: Job):
    job.status = "running"
    job.t0 = time.time()
    try:
        if STATE.sep is None:
            raise RuntimeError("模型尚未就绪")

        with STATE.lock:
            job.stage = "解码音频"

        audio = load_audio(job.src)
        n = len(audio)
        chunk = int(job.chunk * SR)
        overlap = int(OVERLAP_SEC * SR)
        plan = chunk_plan(n, chunk, overlap)

        win = np.ones(chunk, dtype=np.float64)
        if overlap > 0:
            fade = np.hanning(2 * overlap)[:overlap]
            win[:overlap] = fade
            win[-overlap:] = fade[::-1]

        total = len(plan) * len(job.prompts)
        done = 0
        outdir = OUTPUTS / job.id
        outdir.mkdir(parents=True, exist_ok=True)

        for pi, prompt in enumerate(job.prompts):
            if job.cancel:
                break
            acc_t = np.zeros(n, dtype=np.float64)
            acc_r = np.zeros(n, dtype=np.float64)
            wsum = np.zeros(n, dtype=np.float64)

            for i, (s, e) in enumerate(plan, 1):
                if job.cancel:
                    break
                with STATE.lock:
                    job.stage = f"{prompt} · 段 {i}/{len(plan)}"
                seg = audio[s:e]
                w = win[: len(seg)]
                t, r = STATE.sep.separate(
                    torch.from_numpy(seg).unsqueeze(0), prompt, False)
                m = min(len(t), len(seg))
                acc_t[s:s + m] += t[:m] * w[:m]
                acc_r[s:s + m] += r[:m] * w[:m]
                wsum[s:s + m] += w[:m]
                done += 1
                with STATE.lock:
                    job.progress = done / total

            if job.cancel:            # 中途取消：不写半成品
                break

            wsum[wsum < 1e-6] = 1.0
            t_out = (acc_t / wsum).astype(np.float32)
            r_out = (acc_r / wsum).astype(np.float32)
            if job.lowpass_fc:
                t_out = lowpass(t_out, job.lowpass_fc)

            name = slug(prompt)
            tp = outdir / f"{name}__target.wav"
            rp = outdir / f"{name}__residual.wav"
            short = slice(0, min(n, SR * 20))
            import soundfile as sf
            sf.write(tp, t_out, SR, subtype="PCM_24")
            sf.write(rp, r_out, SR, subtype="PCM_24")

            with STATE.lock:
                job.stage = f"{prompt} · 生成频谱图"
            img = outdir / f"{name}__spec.png"
            try:
                spectrogram_png(t_out[short], r_out[short], audio[short], img)
                img_ok = True
            except Exception:  # noqa: BLE001
                traceback.print_exc()
                img_ok = False

            energy = 20 * np.log10(
                (np.sqrt((t_out ** 2).mean()) + 1e-9) / (np.sqrt((r_out ** 2).mean()) + 1e-9))
            with STATE.lock:
                job.results.append({
                    "prompt": prompt,
                    "target": f"/api/file/{job.id}/{name}__target.wav",
                    "residual": f"/api/file/{job.id}/{name}__residual.wav",
                    "spec": f"/api/file/{job.id}/{name}__spec.png" if img_ok else None,
                    "duration": round(len(t_out) / SR, 2),
                    "energy": round(float(energy), 1),
                })

        with STATE.lock:
            if job.cancel:
                job.status = "cancelled"
                job.stage = "已取消"
            else:
                job.status = "done"
                job.stage = f"完成，用时 {time.time() - job.t0:.0f}s"
                job.progress = 1.0
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        with STATE.lock:
            job.status = "error"
            job.error = f"{type(e).__name__}: {e}"
            job.stage = "出错"


# ---------------------------------------------------------------- HTTP


def probe_duration(p: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(p)],
        capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # 静音
        pass

    # ---------- 工具

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _bytes(self, data: bytes, ctype: str):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _file(self, path: Path, ctype: str | None = None, download: str | None = None):
        """支持 Range，HTML5 audio 拖动进度条需要。"""
        if not path.is_file():
            self._json({"error": "not found"}, 404)
            return
        size = path.stat().st_size
        ctype = ctype or "application/octet-stream"
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        partial = False
        if rng and rng.startswith("bytes="):
            try:
                a, _, b = rng[6:].partition("-")
                if a:
                    start = int(a)
                if b:
                    end = int(b)
                elif not a:
                    start = size - int(b)
                end = min(end, size - 1)
                partial = True
            except ValueError:
                pass
        length = end - start + 1
        self.send_response(206 if partial else 200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        if download:
            self.send_header(
                "Content-Disposition",
                f"attachment; filename*=UTF-8''{path.name}")
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                buf = f.read(min(262144, remaining))
                if not buf:
                    break
                self.wfile.write(buf)
                remaining -= len(buf)

    # ---------- GET

    def do_GET(self):
        u = urlparse(self.path)
        p = u.path

        if p in ("/", "/index.html"):
            if not INDEX_HTML.is_file():
                self._json({"error": "index.html 缺失"}, 500)
                return
            self._bytes(INDEX_HTML.read_bytes(), "text/html; charset=utf-8")
            return

        if p == "/api/state":
            self._json(STATE.snapshot())
            return

        if p.startswith("/api/file/"):
            parts = p.split("/")
            if len(parts) >= 5:
                jid = parts[3]
                name = unquote("/".join(parts[4:]))
                target = (OUTPUTS / jid / name).resolve()
                try:
                    target.relative_to(OUTPUTS.resolve())
                except ValueError:
                    self._json({"error": "forbidden"}, 403)
                    return
                ctype = "audio/wav" if name.endswith(".wav") else "image/png"
                dl = "1" if parse_qs(u.query).get("dl") else None
                self._file(target, ctype, download=dl)
                return
            self._json({"error": "bad path"}, 400)
            return

        self._json({"error": "not found"}, 404)

    # ---------- POST

    def do_POST(self):
        u = urlparse(self.path)
        p = u.path
        q = parse_qs(u.query)

        if p == "/api/upload":
            name = unquote(q.get("name", ["input"])[0])[:120]
            UPLOADS.mkdir(parents=True, exist_ok=True)
            uid = uuid.uuid4().hex[:8]
            raw = UPLOADS / f"{uid}__src.tmp"
            length = int(self.headers.get("Content-Length", 0))
            remaining = length
            try:
                with open(raw, "wb") as f:
                    while remaining > 0:
                        buf = self.rfile.read(min(1048576, remaining))
                        if not buf:
                            break
                        f.write(buf)
                        remaining -= len(buf)
                # 立刻转成 48k 单声道 wav，原文件丢弃：312MB 视频 → 18MB 音频
                audio = decode(raw)
                if len(audio) < SR // 2:
                    raise RuntimeError("音频太短（不足 0.5 秒）")
                import soundfile as sf
                dest = UPLOADS / f"{uid}.wav"
                sf.write(dest, audio, SR, subtype="PCM_24")
                dur = len(audio) / SR
            except Exception as e:  # noqa: BLE001
                raw.unlink(missing_ok=True)
                for f in UPLOADS.glob(f"{uid}.wav"):
                    f.unlink(missing_ok=True)
                self._json({"error": f"无法解码此文件：{e}"}, 400)
                return
            finally:
                raw.unlink(missing_ok=True)
            with STATE.lock:
                STATE.last = {"upload": dest.name, "filename": name,
                              "duration": round(dur, 2)}
            self._json({"ok": True, "upload": dest.name, "filename": name,
                        "duration": round(dur, 2), "size": dest.stat().st_size,
                        "src_size": length})
            return

        if p == "/api/run":
            n = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                self._json({"error": "bad json"}, 400)
                return

            with STATE.lock:
                if STATE.job and STATE.job.status in ("queued", "running"):
                    self._json({"error": "已有任务在跑，等它结束或先取消"}, 409)
                    return

            up = body.get("upload")
            prompts = [s.strip() for s in body.get("prompts", []) if s.strip()]
            if not up or not prompts:
                self._json({"error": "缺少文件或提示词"}, 400)
                return

            src = UPLOADS / up
            if not src.is_file():
                self._json({"error": "上传文件已失效，请重新选择"}, 400)
                return

            lp_key = int(body.get("lowpass", 0) or 0)
            lowpass_fc = LOWPASS_CHOICES.get(lp_key, None)
            chunk = float(body.get("chunk", 15) or 15)
            dur = probe_duration(src)

            job = Job(body.get("filename") or up, src, prompts,
                      lowpass_fc, chunk, dur)
            with STATE.lock:
                STATE.job = job
            threading.Thread(target=run_job, args=(job,), daemon=True).start()
            self._json({"ok": True, "job": job.id})
            return

        if p == "/api/cancel":
            with STATE.lock:
                if STATE.job and STATE.job.status == "running":
                    STATE.job.cancel = True
                    STATE.job.stage = "正在取消…"
            self._json({"ok": True})
            return

        if p == "/api/reveal":
            n = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                body = {}
            jid = body.get("job")
            target = OUTPUTS / jid if jid else OUTPUTS
            if not target.is_dir():
                target = OUTPUTS
            try:
                subprocess.Popen(["explorer", str(target)])
                self._json({"ok": True, "path": str(target)})
            except Exception as e:  # noqa: BLE001
                self._json({"error": str(e)}, 500)
            return

        self._json({"error": "not found"}, 404)


def sweep_old(days: float = 3.0):
    """清掉几天前的上传与输出，避免长期使用把磁盘塞满。"""
    cutoff = time.time() - days * 86400
    for d in (UPLOADS, OUTPUTS):
        if not d.is_dir():
            continue
        for f in d.iterdir():
            try:
                if f.stat().st_mtime < cutoff:
                    shutil.rmtree(f, ignore_errors=True) if f.is_dir() else f.unlink(True)
            except OSError:
                pass


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    UPLOADS.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    sweep_old()

    threading.Thread(target=load_engine, daemon=True).start()

    srv = None
    port = PORT
    for p in range(PORT, PORT + 40):
        try:
            srv = ThreadingHTTPServer(("127.0.0.1", p), Handler)
            port = p
            break
        except OSError:
            continue
    if srv is None:
        raise RuntimeError(f"{PORT}-{PORT + 40} 端口全部被占用")

    url = f"http://127.0.0.1:{port}"
    print(f"\n  分离工作台已启动: {url}")
    print("  模型在后台加载，界面会显示状态。Ctrl+C 退出。\n")
    try:
        webbrowser.open(url)
    except Exception:  # noqa: BLE001
        pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n  已退出")
        srv.shutdown()


if __name__ == "__main__":
    main()
