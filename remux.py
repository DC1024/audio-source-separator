"""把分离出来的音频合回原视频，输出新的 mp4。

用法：
    # 用分离出的引擎声替换原音轨
    python remux.py 原视频.mp4 separated\\xxx\\car_engine__target.wav -o 输出.mp4

    # 保留原音轨，把分离结果叠加上去（混音）
    python remux.py 原视频.mp4 separated\\xxx\\car_engine__target.wav -o 输出.mp4 --mix 0.7

    # 只要画面 + 引擎声，并把引擎声放大
    python remux.py 原视频.mp4 target.wav -o 输出.mp4 --gain 3

视频流直接复制（不重新编码），只有音频重编码，所以很快。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def probe_duration(path: str) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", path],
        capture_output=True, text=True,
    )
    try:
        return float(r.stdout.strip())
    except ValueError:
        return -1.0


def has_audio_stream(path: str) -> bool:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=index", "-of", "csv=p=0", path],
        capture_output=True, text=True,
    )
    return bool(r.stdout.strip())


def main() -> None:
    ap = argparse.ArgumentParser(description="把分离出的音频合回视频")
    ap.add_argument("video", help="原视频文件")
    ap.add_argument("audio", help="分离出的音频（target 或 residual）")
    ap.add_argument("-o", "--out", required=True, help="输出 mp4 路径")
    ap.add_argument("--mix", type=float, default=None,
                    help="与原音轨混合的比例 0~1（不给则完全替换原音轨）")
    ap.add_argument("--gain", type=float, default=1.0, help="分离音频的增益倍数")
    ap.add_argument("--aac-bitrate", default="192k")
    args = ap.parse_args()

    for p in (args.video, args.audio):
        if not os.path.isfile(p):
            sys.exit(f"找不到文件: {p}")

    dv, da = probe_duration(args.video), probe_duration(args.audio)
    if dv > 0 and da > 0 and abs(dv - da) > 1.0:
        print(f"  [提示] 时长不一致：视频 {dv:.1f}s，音频 {da:.1f}s，将以视频为准")
    print(f"  视频 {dv:.1f}s，音频 {da:.1f}s")

    out_parent = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_parent, exist_ok=True)

    afilter = []
    if args.gain != 1.0:
        afilter.append(f"volume={args.gain}")

    cmd = ["ffmpeg", "-nostdin", "-y", "-i", args.video, "-i", args.audio]

    if args.mix is not None:
        if not has_audio_stream(args.video):
            sys.exit("原视频没有音轨，无法混合；去掉 --mix 直接替换即可")
        m = max(0.0, min(1.0, args.mix))
        # 原音轨保留 m，分离音频给 (1-m)，再叠加
        fc = (
            f"[0:a]aresample=48000,aformat=channel_layouts=stereo,volume={m}[a0];"
            f"[1:a]{','.join(afilter) if afilter else 'anull'},"
            f"aresample=48000,aformat=channel_layouts=stereo,volume={1 - m:.3f}[a1];"
            f"[a0][a1]amix=inputs=2:duration=first:normalize=0[aout]"
        )
        cmd += ["-filter_complex", fc, "-map", "0:v:0", "-map", "[aout]"]
    else:
        if afilter:
            cmd += ["-af", ",".join(afilter)]
        cmd += ["-map", "0:v:0", "-map", "1:a:0"]

    cmd += [
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", args.aac_bitrate,
        "-shortest",
        "-movflags", "+faststart",
        args.out,
    ]

    print("  合成中（视频流直接复制，不重新编码）…")
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        sys.exit("ffmpeg 失败:\n" + r.stderr.decode("utf-8", "ignore")[-1200:])

    size = os.path.getsize(args.out) / 1e6
    print(f"  完成 → {args.out}  ({size:.1f} MB)")


if __name__ == "__main__":
    main()
