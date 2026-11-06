"""对分离结果做量化评估：SI-SDR 与频谱相关性。"""
import numpy as np
import soundfile as sf

SR = 48000


def load(p):
    x, sr = sf.read(p, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(1)
    assert sr == SR, sr
    return x


def align(x, n):
    if len(x) >= n:
        return x[:n]
    return np.pad(x, (0, n - len(x)))


def si_sdr(est, ref):
    est, ref = est - est.mean(), ref - ref.mean()
    if est.shape != ref.shape:
        n = min(len(est), len(ref))
        est, ref = est[:n], ref[:n]
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12)
    proj = alpha * ref
    noise = est - proj
    return 10 * np.log10((np.sum(proj ** 2) + 1e-12) / (np.sum(noise ** 2) + 1e-12))


def best_delay_si_sdr(est, ref, max_lag=4800):
    """搜索最佳时延，排除对齐误差对 SI-SDR 的惩罚。"""
    best = (-1e9, 0)
    for lag in range(-max_lag, max_lag + 1, 24):
        if lag >= 0:
            a, b = est[lag:], ref[:len(ref) - lag]
        else:
            a, b = est[:lag], ref[-lag:]
        n = min(len(a), len(b))
        if n < SR:
            continue
        v = si_sdr(a[:n], b[:n])
        if v > best[0]:
            best = (v, lag)
    return best


def mel_corr(a, b, n_fft=1024, hop=256):
    """粗略频谱相关性：STFT 幅度谱的皮尔逊相关。"""
    def spec(x):
        n = 1 + (len(x) - n_fft) // hop
        frames = np.lib.stride_tricks.as_strided(
            x, shape=(n_fft, n), strides=(x.strides[0], hop * x.strides[0])
        )
        return np.abs(np.fft.rfft(frames * np.hanning(n_fft)[:, None], axis=0))

    sa, sb = spec(a), spec(b)
    m = min(sa.shape[1], sb.shape[1])
    sa, sb = sa[:, :m].ravel(), sb[:, :m].ravel()
    return float(np.corrcoef(sa, sb)[0, 1])


def band_energy(x, lo, hi):
    X = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    f = np.fft.rfftfreq(len(x), 1 / SR)
    return 10 * np.log10(np.sum(X[(f >= lo) & (f < hi)]) + 1e-12)


tgt = load("out_test/car_engine__target.wav")
res = load("out_test/car_engine__residual.wav")
ref_e = load("samples/ref_engine.wav")
ref_m = load("samples/ref_music.wav")
mix = load("samples/mix.wav")

n = min(len(tgt), len(res), len(ref_e), len(ref_m), len(mix))
tgt, res, ref_e, ref_m, mix = (align(x, n) for x in (tgt, res, ref_e, ref_m, mix))

print(f"{'信号对':<38}{'SI-SDR(dB)':>12}{'最佳时延':>10}")
print("-" * 62)


def row(name, est, ref):
    v, lag = best_delay_si_sdr(est, ref)
    print(f"{name:<38}{v:>12.2f}{lag:>9}")


row("混合原始音 vs 引擎参考（未分离基线）", mix, ref_e)
row("target vs 引擎参考（越高越好）", tgt, ref_e)
row("target vs 音乐参考（越低越好）", tgt, ref_m)
row("residual vs 音乐参考（越高越好）", res, ref_m)
row("residual vs 引擎参考（越低越好）", res, ref_e)

print()
print(f"{'频谱相关性':<38}{'相关系数':>12}")
print("-" * 62)
print(f"{'target 与引擎参考':<38}{mel_corr(tgt, ref_e):>12.3f}")
print(f"{'target 与音乐参考':<38}{mel_corr(tgt, ref_m):>12.3f}")
print(f"{'residual 与音乐参考':<38}{mel_corr(res, ref_m):>12.3f}")

print()
print(f"{'频段能量 (dB)':<20}{'20-250Hz':>12}{'250-2kHz':>12}{'2k-16kHz':>12}")
print("-" * 58)
for name, x in [("引擎参考", ref_e), ("音乐参考", ref_m), ("target", tgt), ("residual", res)]:
    print(f"{name:<20}{band_energy(x, 20, 250):>12.1f}{band_energy(x, 250, 2000):>12.1f}{band_energy(x, 2000, 16000):>12.1f}")
