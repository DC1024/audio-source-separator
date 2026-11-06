"""批量音源分离：一次加载模型，处理整个文件夹里的音视频。

用法：
    python batch.py 输入目录或文件 -p "car engine"
    python batch.py D:\\素材\\车载 -p "car engine" "engine revving" --spans
    python batch.py D:\\a.mp4 -p "music" -o D:\\输出

输出结构：
    <out>/<原文件名>/<提示词>__target.wav     目标声音
    <out>/<原文件名>/<提示词>__residual.wav   剩余声音

长音频自动切段处理（默认 15s 一段、1s 重叠交叉淡化），避免显存爆掉。
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

# 本机已备齐全部权重，默认离线 + 固定缓存目录，避免走失效代理卡在连接超时
_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_HOME", os.path.join(_HERE, "hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

SR = 48_000
AUDIO_EXT = {
    ".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma", ".aiff",
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".ts",
}


def _ff_err(stderr: bytes) -> str:
    """从 ffmpeg 输出里挑出真正有用的那几行，丢掉版本横幅和编译参数。"""
    noise = ("--enable-", "libavutil", "libavcodec", "libavformat", "libavdevice",
             "libavfilter", "libswscale", "libswresample", "built with",
             "configuration:", "Copyright", "ffmpeg version")
    lines = [l.strip() for l in stderr.decode("utf-8", "ignore").splitlines() if l.strip()]
    keep = [l for l in lines if not any(n in l for n in noise)]
    return " / ".join((keep or lines)[-3:])[:300] or "未知错误"


def decode(path: Path) -> np.ndarray:
    """ffmpeg 解码为 48k 单声道 float32，绕开 torchcodec 的后端依赖。"""
    with tempfile.TemporaryDirectory() as td:
        wav = os.path.join(td, "dec.wav")
        r = subprocess.run(
            ["ffmpeg", "-nostdin", "-y", "-i", str(path),
             "-vn", "-ac", "1", "-ar", str(SR), "-f", "wav", wav],
            capture_output=True,
        )
        if r.returncode != 0:
            raise RuntimeError(_ff_err(r.stderr))
        x, sr = sf.read(wav, dtype="float32", always_2d=False)
    if sr != SR:
        raise RuntimeError(f"解码采样率异常 {sr}")
    return x


def slug(text: str) -> str:
    return re.sub(r"[^0-9a-zA-Z\u4e00-\u9fa5]+", "_", text.strip()).strip("_") or "target"


def lowpass(x: np.ndarray, fc: float, sr: int = SR, order: int = 6) -> np.ndarray:
    """零相位低通。引擎声能量几乎全在 2kHz 以下，音乐残留主要在其上。"""
    from scipy.signal import butter, sosfiltfilt

    sos = butter(order, fc / (sr / 2), btype="low", output="sos")
    return sosfiltfilt(sos, x).astype(np.float32)


def chunk_plan(n: int, chunk: int, overlap: int):
    """返回 [(start, end), ...]，末段不足时向前补齐。"""
    if n <= chunk:
        return [(0, n)]
    step = chunk - overlap
    spans = []
    s = 0
    while s + chunk < n:
        spans.append((s, s + chunk))
        s += step
    spans.append((max(0, n - chunk), n))
    return spans


class Separator:
    def __init__(self, model_dir: str, device: str, use_spans: bool = False):
        from sam_audio import SAMAudio, SAMAudioProcessor

        t0 = time.time()
        self.model = SAMAudio.from_pretrained(model_dir).eval().to(device)
        # 不用 spans 时把 6.12GB 的 span 模型挪回 CPU，否则会把主模型挤到显存边缘导致降速
        if not use_spans and hasattr(self.model, "span_predictor"):
            self.model.span_predictor.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("  span 模型已卸载到 CPU（省 6GB 显存）", flush=True)
        self.processor = SAMAudioProcessor.from_pretrained(model_dir)
        self.device = device
        print(f"  模型加载完成 {time.time() - t0:.0f}s", flush=True)
        if torch.cuda.is_available():
            print(f"  显存占用 {torch.cuda.memory_allocated() / 1e9:.1f} GB", flush=True)

    @torch.inference_mode()
    def separate(self, wav: torch.Tensor, prompt: str, spans: bool) -> tuple[np.ndarray, np.ndarray]:
        batch = self.processor(audios=[wav], descriptions=[prompt]).to(self.device)
        out = self.model.separate(batch, predict_spans=spans, reranking_candidates=1)
        t = out.target[0].detach().float().cpu().numpy().squeeze()
        r = out.residual[0].detach().float().cpu().numpy().squeeze()
        return t, r

    def separate_purified(self, wav: torch.Tensor, prompt: str, purify: str,
                          spans: bool) -> tuple[np.ndarray, np.ndarray]:
        """二次净化：先把目标抽出来，再从目标里把残留的 purify 声音反向抽走。

        第一次 target = 引擎 + 少量音乐残留
        第二次以 purify（如 music）为目标再分离 → 其 residual 就是更纯的引擎
        代价是约 2 倍时间，且会引入二次生成伪影。
        """
        t1, r1 = self.separate(wav, prompt, spans)
        if len(t1) < SR // 4:
            return t1, r1
        _, t2 = self.separate(torch.from_numpy(t1).unsqueeze(0), purify, spans)
        n = min(len(t1), len(t2))
        return t2[:n].astype(np.float32), r1[:n].astype(np.float32)


def run_long(sep: Separator, audio: np.ndarray, prompt: str, spans: bool,
             chunk_sec: float, overlap_sec: float, purify: str | None = None):
    """长音频按段分离再交叉淡化拼接。"""
    chunk = int(chunk_sec * SR)
    overlap = int(overlap_sec * SR)
    plan = chunk_plan(len(audio), chunk, overlap)
    n = len(audio)

    acc_t = np.zeros(n, dtype=np.float64)
    acc_r = np.zeros(n, dtype=np.float64)
    wsum = np.zeros(n, dtype=np.float64)

    win = np.ones(chunk, dtype=np.float64)
    if overlap > 0:
        fade = np.hanning(2 * overlap)[:overlap]
        win[:overlap] = fade
        win[-overlap:] = fade[::-1]

    for i, (s, e) in enumerate(plan, 1):
        seg = audio[s:e]
        w = win[: len(seg)]
        if purify:
            t, r = sep.separate_purified(torch.from_numpy(seg).unsqueeze(0), prompt, purify, spans)
        else:
            t, r = sep.separate(torch.from_numpy(seg).unsqueeze(0), prompt, spans)
        m = min(len(t), len(seg))
        acc_t[s:s + m] += t[:m] * w[:m]
        acc_r[s:s + m] += r[:m] * w[:m]
        wsum[s:s + m] += w[:m]
        print(f"    段 {i}/{len(plan)}  {s / SR:.1f}-{e / SR:.1f}s", flush=True)

    wsum[wsum < 1e-6] = 1.0
    return (acc_t / wsum).astype(np.float32), (acc_r / wsum).astype(np.float32)


def collect(inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.suffix.lower() in AUDIO_EXT:
                    files.append(f)
        elif p.is_file():
            files.append(p)
        else:
            print(f"  [跳过] 找不到 {p}")
    return files


def main() -> None:
    ap = argparse.ArgumentParser(description="批量音源分离")
    ap.add_argument("inputs", nargs="+", help="音频/视频文件或目录（可多个）")
    ap.add_argument("-p", "--prompt", nargs="+", required=True, help="目标声音描述，小写英文")
    ap.add_argument("-m", "--model", default="models/sam-audio-small")
    ap.add_argument("-o", "--out", default="out")
    ap.add_argument("--spans", action="store_true", help="自动定位发声时段，建议开")
    ap.add_argument("--lowpass", type=float, default=None, metavar="HZ",
                    help="对分离结果做低通（如 2000）。引擎声能量几乎全在 2kHz 以下，"
                         "此步可把音乐残留压低 50dB 而引擎低频不受影响")
    ap.add_argument("--purify", default=None, metavar="SOUND",
                    help="[实测无效，不建议用] 二次净化：再从结果里抽走这个声音（如 music）。"
                         "实测会把引擎低频一起削掉，音乐相对占比反升 3.5dB")
    ap.add_argument("--chunk", type=float, default=15.0, help="长音频切段长度（秒）")
    ap.add_argument("--overlap", type=float, default=1.0, help="段间重叠（秒）")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    files = collect(args.inputs)
    if not files:
        sys.exit("没有找到可处理的音视频文件")

    print(f"共 {len(files)} 个文件，{len(args.prompt)} 个提示词：{', '.join(args.prompt)}")
    sep = Separator(args.model, args.device, use_spans=args.spans)

    out_root = Path(args.out)
    total = len(files) * len(args.prompt)
    done = 0
    failed = []

    for f in files:
        print(f"\n[{f.name}]")
        try:
            audio = decode(f)
        except Exception as e:
            print(f"  [失败] 解码: {e}")
            failed.append(f.name)
            continue

        dur = len(audio) / SR
        nseg = len(chunk_plan(len(audio), int(args.chunk * SR), int(args.overlap * SR)))
        print(f"  {dur:.1f}s，{nseg} 段")

        dest = out_root / f.stem
        dest.mkdir(parents=True, exist_ok=True)

        wav = torch.from_numpy(audio).unsqueeze(0)
        for prompt in args.prompt:
            t0 = time.time()
            print(f"  分离 {prompt!r} …", flush=True)
            try:
                if nseg == 1:
                    if args.purify:
                        t, r = sep.separate_purified(wav, prompt, args.purify, args.spans)
                    else:
                        t, r = sep.separate(wav, prompt, args.spans)
                else:
                    t, r = run_long(sep, audio, prompt, args.spans, args.chunk, args.overlap,
                                    purify=args.purify)
            except torch.cuda.OutOfMemoryError:
                print("  [失败] 显存不足，试试 --chunk 8")
                failed.append(f.name)
                break
            except Exception as e:
                print(f"  [失败] {type(e).__name__}: {e}")
                failed.append(f.name)
                break

            s = slug(prompt)
            if args.lowpass:
                t = lowpass(t, args.lowpass)
            sf.write(dest / f"{s}__target.wav", t, SR, subtype="PCM_24")
            sf.write(dest / f"{s}__residual.wav", r, SR, subtype="PCM_24")
            en = 20 * np.log10((np.sqrt((t ** 2).mean()) + 1e-9) / (np.sqrt((r ** 2).mean()) + 1e-9))
            print(f"    -> {dest / (s + '__target.wav')}  ({time.time() - t0:.0f}s, 能量比 {en:+.1f} dB)")
            done += 1

    print(f"\n完成 {done}/{total}" + (f"，失败 {len(failed)} 个: {failed}" if failed else ""))


if __name__ == "__main__":
    main()
