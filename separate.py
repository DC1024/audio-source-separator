"""SAM-Audio 分离器：用文本提示从混音中抠出任意声音。

用法:
    python separate.py input.mp3 --prompt "car engine"
    python separate.py input.mp3 --prompt "car engine" "engine revving" --spans --rerank 8
    python separate.py input.mp3 --prompt "music" --invert

输出: <out>/<prompt_slug>_target.wav (目标声) 与 <out>/<prompt_slug>_residual.wav (其余)
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import time

import numpy as np
import soundfile as sf
import torch

# 本机已备齐全部权重，默认离线 + 固定缓存目录，避免走失效代理卡在连接超时
_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_HOME", os.path.join(_HERE, "hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

SAMPLE_RATE = 48_000
MAX_WHOLE_SEC = 40  # 整段处理的硬上限，超过请用 batch.py 切段


def decode_to_array(path: str) -> np.ndarray:
    """用 ffmpeg 解码任意格式为 48k 单声道 float32，规避 torchcodec 后端问题。"""
    if not os.path.isfile(path):
        sys.exit(f"找不到文件: {path}")

    with tempfile.TemporaryDirectory() as td:
        wav = os.path.join(td, "dec.wav")
        cmd = [
            "ffmpeg", "-nostdin", "-y", "-i", path,
            "-vn",  # 丢弃视频流，只解音频
            "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "wav", wav,
        ]
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode != 0:
            sys.exit("ffmpeg 解码失败:\n" + r.stderr.decode("utf-8", "ignore")[-1500:])
        data, sr = sf.read(wav, dtype="float32", always_2d=False)
    if sr != SAMPLE_RATE:
        sys.exit(f"解码采样率异常: {sr}")
    return data


def slugify(text: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fa5]+", "_", text.strip()).strip("_")
    return s or "target"


def save(path: str, tensor: torch.Tensor) -> None:
    arr = tensor.detach().cpu().float().numpy()
    arr = np.squeeze(arr)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    sf.write(path, arr, SAMPLE_RATE, subtype="PCM_24")


def main() -> None:
    ap = argparse.ArgumentParser(description="SAM-Audio 文本提示音源分离")
    ap.add_argument("audio", help="输入音频/视频文件")
    ap.add_argument("-p", "--prompt", nargs="+", required=True,
                    help="目标声音描述，小写名词短语，如 'car engine'")
    ap.add_argument("-m", "--model", default="models/sam-audio-small",
                    help="模型目录 (含 config.json 与 checkpoint.pt) 或 HF repo id")
    ap.add_argument("-o", "--out", default="out", help="输出目录")
    ap.add_argument("--spans", action="store_true",
                    help="自动预测目标声音出现的时间段（对引擎轰鸣这类事件型声音效果好）")
    ap.add_argument("--rerank", type=int, default=1,
                    help="候选重排数量，>1 需要 CLAP 排序器权重（默认 1）")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if not os.path.isdir(args.model) and not torch.cuda.is_available():
        pass  # 交给 from_pretrained 走 HF 下载

    print(f"[1/4] 解码 {args.audio}")
    audio = decode_to_array(args.audio)
    dur = len(audio) / SAMPLE_RATE
    print(f"      {dur:.2f}s @ {SAMPLE_RATE}Hz")

    # 硬性上限：整段处理时显存随长度线性增长，128s 在 12GB 卡上会 OOM
    if dur > MAX_WHOLE_SEC:
        sys.exit(
            f"\n[中止] 音频 {dur:.0f}s 超过整段处理上限（{MAX_WHOLE_SEC}s）。\n"
            f"       128s 音频在 12GB 显存上需要约 15GB，必然 OOM。\n"
            f"       请改用切段版：\n\n"
            f"       sam-audio\\\\.venv\\\\Scripts\\\\python.exe batch.py \"{args.audio}\" "
            f"-p \"{args.prompt[0]}\" --chunk 15\n"
        )

    wav = torch.from_numpy(audio).unsqueeze(0)  # (1, T)

    print(f"[2/4] 加载模型 {args.model}")
    from sam_audio import SAMAudio, SAMAudioProcessor

    t0 = time.time()
    model = SAMAudio.from_pretrained(args.model).eval().to(args.device)
    # 不用 spans 时把 6.12GB 的 span 模型挪回 CPU，否则会把主模型挤到显存边缘导致降速
    if not args.spans and hasattr(model, "span_predictor"):
        model.span_predictor.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("      span 模型已卸载到 CPU（省 6GB 显存）")
    processor = SAMAudioProcessor.from_pretrained(args.model)
    print(f"      完成, {time.time() - t0:.1f}s, device={args.device}")
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        print(f"      显存占用 {alloc:.1f} GB")

    os.makedirs(args.out, exist_ok=True)

    print(f"[3/4] 分离中 (prompts={args.prompt}, spans={args.spans}, rerank={args.rerank})")
    batch = processor(audios=[wav] * len(args.prompt), descriptions=list(args.prompt)).to(args.device)

    t0 = time.time()
    with torch.inference_mode():
        result = model.separate(
            batch,
            predict_spans=args.spans,
            reranking_candidates=max(1, args.rerank),
        )
    print(f"      推理耗时 {time.time() - t0:.1f}s")

    print("[4/4] 写出")
    for i, prompt in enumerate(args.prompt):
        base = slugify(prompt)
        tgt = os.path.join(args.out, f"{base}__target.wav")
        res = os.path.join(args.out, f"{base}__residual.wav")
        save(tgt, result.target[i])
        save(res, result.residual[i])
        print(f"      {prompt!r} -> {tgt}")
        print(f"      {'':<{len(prompt) + 2}}    {res}")

    # 简单的能量参考，便于判断分离是否成功
    for i, prompt in enumerate(args.prompt):
        t = result.target[i].detach().float().cpu()
        r = result.residual[i].detach().float().cpu()
        ratio = 20 * np.log10((t.pow(2).mean().sqrt().item() + 1e-9) / (r.pow(2).mean().sqrt().item() + 1e-9))
        print(f"      能量比 {prompt!r}: target/residual = {ratio:+.1f} dB")


if __name__ == "__main__":
    main()
