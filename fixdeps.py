"""反复试导入，缺什么装什么，直到 sam_audio 全链路可导入。

放在 sam-audio 的上一级目录运行：
    python fixdeps.py

需要联网时自行设置代理环境变量（HTTPS_PROXY 等），脚本不写死。
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_WIN = os.path.join(HERE, "sam-audio", ".venv", "Scripts", "python.exe")
_NIX = os.path.join(HERE, "sam-audio", ".venv", "bin", "python")
PY = _WIN if os.path.exists(_WIN) else _NIX
UV = "uv"
MIRROR = os.environ.get("PIP_MIRROR", "https://pypi.org/simple")
TARGET = (
    "import sam_audio; "
    "from sam_audio import SAMAudio, SAMAudioProcessor; "
    "from imagebind import data; "
    "import torch"
)
ALIAS = {
    "flatten_dict": "flatten-dict",
    "audiotools": "descript-audiotools",
    "yaml": "pyyaml",
    "PIL": "pillow",
    "cv2": "opencv-python-headless",
}
GIT_DEPS = {
    "pytorchvideo": "pytorchvideo@git+https://github.com/facebookresearch/pytorchvideo.git",
}

ENV_EXTRA = {
    "UV_LINK_MODE": "copy",
}

env = dict(os.environ)
env.update(ENV_EXTRA)

for i in range(25):
    r = subprocess.run([PY, "-c", TARGET], capture_output=True, text=True, env=env)
    if r.returncode == 0:
        print(f"[OK] 全部依赖就绪（第 {i} 轮）")
        sys.exit(0)
    m = re.search(r"No module named '([\w\.]+)'", r.stderr)
    if not m:
        print("[FAIL] 非缺包错误：")
        print(r.stderr[-2500:])
        sys.exit(1)
    mod = m.group(1).split(".")[0]
    spec = GIT_DEPS.get(mod) or ALIAS.get(mod, mod)
    if spec.startswith("pytorchvideo@"):
        cmd = [UV, "pip", "install", "--python", PY, "--no-deps", spec]
    else:
        cmd = [UV, "pip", "install", "--python", PY, "--no-deps",
               "--only-binary", ":all:", "--index-url", MIRROR, spec]
    print(f"[{i+1}] 缺 {mod} -> 安装 {spec}")
    rr = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if rr.returncode != 0:
        print("  安装失败:", (rr.stderr or rr.stdout)[-400:])
        sys.exit(1)

print("[FAIL] 超过 25 轮仍未收敛")
sys.exit(1)
