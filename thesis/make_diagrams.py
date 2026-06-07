"""Generate schematic diagrams for the thesis (architecture, methods, ECG schematics).

These are author-made illustrations that complement the experimental figures
under runs/figures/. Output goes to thesis/figs/.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

OUT = os.path.join(os.path.dirname(__file__), "figs")
os.makedirs(OUT, exist_ok=True)


def _box(ax, xy, w, h, text, fc="#eef3fb", ec="#2c5d9b", fs=10, lw=1.4):
    x, y = xy
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                       linewidth=lw, edgecolor=ec, facecolor=fc, mutation_scale=1)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, wrap=True)
    return (x + w / 2, y + h / 2)


def _arrow(ax, p0, p1, color="#444", lw=1.4, style="-|>"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=14,
                                 lw=lw, color=color, shrinkA=2, shrinkB=2))


# ---------------------------------------------------------------- PQRST схема
def pqrst():
    fs = 360
    t = np.linspace(0, 1.0, fs)

    def g(c, w, a):
        return a * np.exp(-0.5 * ((t - c) / w) ** 2)

    sig = (g(0.22, 0.025, 0.15)            # P
           - g(0.38, 0.008, 0.10)          # Q
           + g(0.40, 0.010, 1.00)          # R
           - g(0.42, 0.009, 0.25)          # S
           + g(0.62, 0.045, 0.35))         # T
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.plot(t, sig, color="#19407a", lw=1.8)
    labels = {"P": (0.22, 0.18), "Q": (0.375, -0.16), "R": (0.40, 1.05),
              "S": (0.425, -0.30), "T": (0.62, 0.40)}
    for name, (x, y) in labels.items():
        ax.annotate(name, (x, y), fontsize=13, fontweight="bold", color="#b03030", ha="center")
    ax.annotate("", xy=(0.38, -0.42), xytext=(0.43, -0.42),
                arrowprops=dict(arrowstyle="<->", color="#555"))
    ax.text(0.405, -0.5, "QRS", ha="center", fontsize=10, color="#555")
    ax.set_xlabel("Час, с")
    ax.set_ylabel("Амплітуда, мВ")
    ax.set_ylim(-0.6, 1.25)
    ax.grid(alpha=0.25)
    fig.savefig(os.path.join(OUT, "pqrst.png"))
    plt.close(fig)


# ------------------------------------------------------------ архітектура системи
def architecture():
    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    ax.set_xlim(0, 12); ax.set_ylim(0, 12); ax.axis("off")

    _box(ax, (0.4, 10.4), 11.2, 1.0,
         "Вхідні файли ЕКГ: DICOM · WFDB · EDF/BDF · HL7 aECG · SCP-ECG · CSV · XML",
         fc="#f3f0e8", ec="#8a7a4a", fs=9.5)
    p_in = (6, 10.4)

    p_det = _box(ax, (4.2, 8.9), 3.6, 0.9, "Детектор формату\n(detect.py)")
    _box(ax, (0.4, 7.2), 11.2, 1.0,
         "Уніфікований парсер → проміжне подання у форматі XML (ekg.parsers)",
         fc="#eaf4ec", ec="#3f8a52", fs=9.5)
    p_xml = (6, 7.2)

    # signal layer
    sx = [0.4, 3.3, 6.2, 9.1]
    names = ["Фільтрація\n(signal)", "Детекція R-піків\n(AdvancedRPeak)",
             "Морфологія\n(Morphology)", "HRV\n(HRVAnalyzer)"]
    sig_ps = []
    for x, nm in zip(sx, names):
        sig_ps.append(_box(ax, (x, 5.5), 2.5, 1.0, nm, fc="#eef3fb", fs=9))

    p_win = _box(ax, (0.4, 3.9), 5.4, 0.9, "Сегментація ударів (windows)\n260 відліків навколо R-піка", fs=9)
    p_feat = _box(ax, (6.2, 3.9), 5.4, 0.9, "Витяг ознак (handcrafted)\n36/70-вим. вектор", fs=9)

    mx = [0.4, 2.6, 4.8, 7.0, 9.2]
    mnames = ["RF", "XGBoost", "SVM-RBF", "1D-CNN /\nCNN-BiLSTM", "Гібрид\n(новизна 1)"]
    mcol = ["#eef3fb", "#eef3fb", "#eef3fb", "#fbeeee", "#f6e8fb"]
    mod_ps = []
    for x, nm, c in zip(mx, mnames, mcol):
        mod_ps.append(_box(ax, (x, 2.2), 2.0, 1.0, nm, fc=c, fs=8.5))

    p_out = _box(ax, (0.4, 0.5), 5.4, 0.9, "REST API (FastAPI)\n/classify · /models · /health", fc="#f3f0e8", ec="#8a7a4a", fs=9)
    p_cli = _box(ax, (6.2, 0.5), 5.4, 0.9, "CLI (ekg)\nparse · train · evaluate · benchmark · serve", fc="#f3f0e8", ec="#8a7a4a", fs=9)

    _arrow(ax, p_in, p_det)
    _arrow(ax, p_det, p_xml)
    for sp in sig_ps:
        _arrow(ax, (sp[0], 7.2), (sp[0], 6.5))
    _arrow(ax, (3.0, 5.5), p_win)
    _arrow(ax, (8.0, 5.5), p_feat)
    for mp in mod_ps:
        _arrow(ax, (mp[0], 3.9), (mp[0], 3.2))
    _arrow(ax, (3.0, 2.2), p_out)
    _arrow(ax, (8.0, 2.2), p_cli)
    fig.savefig(os.path.join(OUT, "architecture.png"))
    plt.close(fig)


# ------------------------------------------------------------ beat window
def beat_window():
    fs = 360
    t = np.linspace(0, 2.5, int(2.5 * fs))
    rng = np.random.default_rng(3)

    def beat(center):
        x = t - center
        return (np.exp(-0.5 * (x / 0.010) ** 2) * 1.0
                - np.exp(-0.5 * ((x + 0.02) / 0.008) ** 2) * 0.12
                - np.exp(-0.5 * ((x - 0.02) / 0.009) ** 2) * 0.22
                + np.exp(-0.5 * ((x - 0.18) / 0.045) ** 2) * 0.3
                + np.exp(-0.5 * ((x + 0.16) / 0.025) ** 2) * 0.13)

    sig = sum(beat(c) for c in (0.45, 1.25, 2.05)) + rng.normal(0, 0.01, t.size)
    fig, ax = plt.subplots(figsize=(7.6, 3.0))
    ax.plot(t, sig, color="#19407a", lw=1.3)
    r = 1.25
    pre, post = 90 / fs, 170 / fs
    ax.axvspan(r - pre, r + post, color="#f0c419", alpha=0.25)
    ax.axvline(r, color="#b03030", ls="--", lw=1.2)
    ax.annotate("R-пік", (r, 1.05), color="#b03030", ha="center", fontsize=11, fontweight="bold")
    ax.annotate("", xy=(r - pre, -0.45), xytext=(r, -0.45), arrowprops=dict(arrowstyle="<->", color="#555"))
    ax.annotate("", xy=(r, -0.45), xytext=(r + post, -0.45), arrowprops=dict(arrowstyle="<->", color="#555"))
    ax.text(r - pre / 2, -0.55, "90 відл.", ha="center", fontsize=9, color="#555")
    ax.text(r + post / 2, -0.55, "170 відл.", ha="center", fontsize=9, color="#555")
    ax.set_xlabel("Час, с"); ax.set_ylabel("Амплітуда (норм.)")
    ax.set_ylim(-0.7, 1.25); ax.grid(alpha=0.2)
    fig.savefig(os.path.join(OUT, "beat_window.png"))
    plt.close(fig)


# ------------------------------------------------------------ hybrid late fusion
def hybrid():
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ax.set_xlim(0, 13.5); ax.set_ylim(0, 8); ax.axis("off")
    p_beat = _box(ax, (0.3, 3.3), 2.0, 1.2, "Удар ЕКГ\n(260 відл.)", fc="#f3f0e8", ec="#8a7a4a")
    p_feat = _box(ax, (3.2, 5.2), 2.6, 1.0, "Ручні ознаки\n(36/70-вим.)")
    p_win = _box(ax, (3.2, 1.4), 2.6, 1.0, "Сирове вікно\n(1/2 відведення)")
    p_xgb = _box(ax, (6.5, 5.2), 2.3, 1.0, "XGBoost", fc="#eef3fb")
    p_cnn = _box(ax, (6.5, 1.4), 2.3, 1.0, "1D-CNN", fc="#fbeeee")
    p_meta = _box(ax, (9.4, 3.3), 2.2, 1.2, "Мета-класиф.\n(логіст. регр.)", fc="#f6e8fb", ec="#7a4a8a")
    ax.text(7.65, 4.35, "p ∈ ℝ⁵", fontsize=8.5, color="#666", ha="center")
    ax.text(7.65, 2.5, "p ∈ ℝ⁵", fontsize=8.5, color="#666", ha="center")
    ax.text(11.75, 3.9, "N/S/V/F/Q", fontsize=9, color="#333", ha="left")
    _arrow(ax, (2.3, 4.2), (3.2, 5.6)); _arrow(ax, (2.3, 3.6), (3.2, 1.9))
    _arrow(ax, (5.8, 5.7), (6.5, 5.7)); _arrow(ax, (5.8, 1.9), (6.5, 1.9))
    _arrow(ax, (8.8, 5.6), (9.6, 4.3)); _arrow(ax, (8.8, 1.9), (9.6, 3.5))
    _arrow(ax, (11.6, 3.9), (11.7, 3.9))
    fig.savefig(os.path.join(OUT, "hybrid.png"))
    plt.close(fig)


# ------------------------------------------------------------ CNN architecture
def cnn_arch():
    fig, ax = plt.subplots(figsize=(9.2, 2.8))
    ax.set_xlim(0, 13); ax.set_ylim(0, 4); ax.axis("off")
    blocks = [
        ("Вхід\n260×C", "#f3f0e8"),
        ("Conv 7\n16", "#eef3fb"), ("Conv 5\n32", "#eef3fb"),
        ("Conv 5\n64", "#eef3fb"), ("Conv 3\n128", "#eef3fb"),
        ("GAP", "#eaf4ec"), ("FC 64", "#eaf4ec"), ("Softmax\n5", "#fbeeee"),
    ]
    x = 0.2; w = 1.4; gap = 0.2; prev = None
    for txt, c in blocks:
        cx = _box(ax, (x, 1.3), w, 1.3, txt, fc=c, fs=8.5)
        if prev is not None:
            _arrow(ax, (prev, 1.95), (x, 1.95))
        prev = x + w
        x += w + gap
    ax.text(6.5, 0.55, "Conv1d → BatchNorm → ReLU → MaxPool(2)", ha="center", fontsize=9, color="#555")
    fig.savefig(os.path.join(OUT, "cnn_arch.png"))
    plt.close(fig)


if __name__ == "__main__":
    pqrst(); architecture(); beat_window(); hybrid(); cnn_arch()
    print("diagrams written to", OUT)
    print(sorted(os.listdir(OUT)))
