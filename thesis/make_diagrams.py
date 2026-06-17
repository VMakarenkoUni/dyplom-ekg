"""Generate thesis diagrams.

Box/flow diagrams (architecture, hybrid, CNN) are rendered with Graphviz
(clean orthogonal arrow routing). Signal plots (PQRST, beat window) stay in
matplotlib. Output: thesis/figs/.
"""
from __future__ import annotations

import os
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "savefig.dpi": 200, "savefig.bbox": "tight"})

OUT = os.path.join(os.path.dirname(__file__), "figs")
os.makedirs(OUT, exist_ok=True)

GV_FONT = "DejaVu Sans"
NODE = (f'node [shape=box, style="rounded,filled", fontname="{GV_FONT}", '
        f'fontsize=12, margin="0.16,0.09", penwidth=1.3];')
EDGE = 'edge [color="#444444", arrowsize=0.85, penwidth=1.2];'


def render_dot(name: str, dot: str) -> None:
    src = os.path.join(OUT, name + ".dot")
    png = os.path.join(OUT, name + ".png")
    with open(src, "w", encoding="utf-8") as f:
        f.write(dot)
    subprocess.run(["dot", "-Tpng", "-Gdpi=200", src, "-o", png], check=True)
    print("rendered", png)


# ----------------------------------------------------------- architecture
def architecture():
    dot = f'''digraph arch {{
  rankdir=TB;
  graph [splines=polyline, nodesep=0.45, ranksep=0.55, fontname="{GV_FONT}", bgcolor="white"];
  {NODE}
  {EDGE}

  input  [label="Вхідні файли ЕКГ:  DICOM · WFDB · EDF/BDF · HL7 aECG · SCP-ECG · CSV · XML",
          fillcolor="#f3f0e8", color="#8a7a4a", width=9];
  detect [label="Детектор формату (detect.py)", fillcolor="#eef3fb", color="#2c5d9b"];
  parser [label="Уніфікований парсер → проміжне подання у форматі XML (ekg.parsers)",
          fillcolor="#eaf4ec", color="#3f8a52", width=9];

  filt  [label="Фільтрація\\n(signal)", fillcolor="#eef3fb", color="#2c5d9b"];
  rpeak [label="Детекція R-піків\\n(AdvancedRPeak)", fillcolor="#eef3fb", color="#2c5d9b"];
  morph [label="Морфологія\\n(Morphology)", fillcolor="#eef3fb", color="#2c5d9b"];
  hrv   [label="HRV\\n(HRVAnalyzer)", fillcolor="#eef3fb", color="#2c5d9b"];

  win  [label="Сегментація ударів (windows)\\n260 відліків навколо R-піка",
        fillcolor="#eef3fb", color="#2c5d9b", width=4];
  feat [label="Витяг ознак (handcrafted)\\n35 / 70-вимірний вектор",
        fillcolor="#eef3fb", color="#2c5d9b", width=4];

  rf  [label="RF", fillcolor="#eef3fb", color="#2c5d9b"];
  xgb [label="XGBoost", fillcolor="#eef3fb", color="#2c5d9b"];
  svm [label="SVM-RBF", fillcolor="#eef3fb", color="#2c5d9b"];
  cnn [label="1D-CNN /\\nCNN-BiLSTM", fillcolor="#fbeeee", color="#b03030"];
  hyb [label="Гібрид\\n(новизна 1)", fillcolor="#f6e8fb", color="#7a4a8a"];

  result [label="Класифікація удару:  N / S / V / F / Q",
          fillcolor="#fff7e0", color="#b58a00", width=6];

  api [label="REST API (FastAPI)\\n/classify · /models · /health",
       fillcolor="#f3f0e8", color="#8a7a4a"];
  cli [label="CLI (ekg)\\nparse · train · evaluate · benchmark · serve",
       fillcolor="#f3f0e8", color="#8a7a4a"];

  input -> detect -> parser;
  parser -> filt; parser -> rpeak; parser -> morph; parser -> hrv;
  filt -> win; rpeak -> win; morph -> feat; hrv -> feat;
  win -> cnn; win -> hyb;
  feat -> rf; feat -> xgb; feat -> svm; feat -> hyb;
  rf -> result; xgb -> result; svm -> result; cnn -> result; hyb -> result;
  result -> api; result -> cli;

  {{ rank=same; filt; rpeak; morph; hrv; }}
  {{ rank=same; win; feat; }}
  {{ rank=same; rf; xgb; svm; cnn; hyb; }}
  {{ rank=same; api; cli; }}
}}'''
    render_dot("architecture", dot)


# ----------------------------------------------------------- hybrid (novelty 1)
def hybrid():
    dot = f'''digraph hybrid {{
  rankdir=LR;
  graph [splines=polyline, nodesep=0.5, ranksep=0.7, fontname="{GV_FONT}", bgcolor="white"];
  {NODE}
  {EDGE}

  beat [label="Удар ЕКГ\\n(260 відліків)", fillcolor="#f3f0e8", color="#8a7a4a"];
  feat [label="Ручні ознаки\\n(35 / 70-вим.)", fillcolor="#eef3fb", color="#2c5d9b"];
  win  [label="Сире вікно\\n(1–2 відведення)", fillcolor="#eef3fb", color="#2c5d9b"];
  xgb  [label="XGBoost", fillcolor="#eef3fb", color="#2c5d9b"];
  cnn  [label="1D-CNN", fillcolor="#fbeeee", color="#b03030"];
  meta [label="Мета-класифікатор\\n(логістична регресія)", fillcolor="#f6e8fb", color="#7a4a8a"];
  out  [label="Клас\\nN / S / V / F / Q", shape=note, fillcolor="#eaf4ec", color="#3f8a52"];

  beat -> feat; beat -> win;
  feat -> xgb; win -> cnn;
  xgb -> meta [label="  p ∈ R⁵", fontname="{GV_FONT}", fontsize=10, fontcolor="#666666"];
  cnn -> meta [label="  p ∈ R⁵", fontname="{GV_FONT}", fontsize=10, fontcolor="#666666"];
  meta -> out;

  {{ rank=same; feat; win; }}
  {{ rank=same; xgb; cnn; }}
}}'''
    render_dot("hybrid", dot)


# ----------------------------------------------------------- CNN architecture
def cnn_arch():
    dot = f'''digraph cnn {{
  rankdir=LR;
  graph [nodesep=0.3, ranksep=0.45, fontname="{GV_FONT}", bgcolor="white"];
  {NODE}
  {EDGE}

  inp  [label="Вхід\\n260 × C", fillcolor="#f3f0e8", color="#8a7a4a"];
  c1   [label="Conv 7\\n16", fillcolor="#eef3fb", color="#2c5d9b"];
  c2   [label="Conv 5\\n32", fillcolor="#eef3fb", color="#2c5d9b"];
  c3   [label="Conv 5\\n64", fillcolor="#eef3fb", color="#2c5d9b"];
  c4   [label="Conv 3\\n128", fillcolor="#eef3fb", color="#2c5d9b"];
  gap  [label="GAP", fillcolor="#eaf4ec", color="#3f8a52"];
  fc   [label="FC 64", fillcolor="#eaf4ec", color="#3f8a52"];
  sm   [label="Softmax\\n5", fillcolor="#fbeeee", color="#b03030"];

  inp -> c1 -> c2 -> c3 -> c4 -> gap -> fc -> sm;
  label="Кожен згортковий блок: Conv1d → BatchNorm → ReLU → MaxPool(2)";
  labelloc="b"; fontname="{GV_FONT}"; fontsize=11; fontcolor="#555555";
}}'''
    render_dot("cnn_arch", dot)


# ----------------------------------------------------------- PQRST (matplotlib)
def pqrst():
    fs = 360
    t = np.linspace(0, 1.0, fs)

    def g(c, w, a):
        return a * np.exp(-0.5 * ((t - c) / w) ** 2)

    sig = (g(0.22, 0.025, 0.15) - g(0.38, 0.008, 0.10) + g(0.40, 0.010, 1.00)
           - g(0.42, 0.009, 0.25) + g(0.62, 0.045, 0.35))
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.plot(t, sig, color="#19407a", lw=1.8)
    for name, (x, y) in {"P": (0.22, 0.18), "Q": (0.375, -0.16), "R": (0.40, 1.05),
                         "S": (0.425, -0.30), "T": (0.62, 0.40)}.items():
        ax.annotate(name, (x, y), fontsize=13, fontweight="bold", color="#b03030", ha="center")
    ax.annotate("", xy=(0.38, -0.42), xytext=(0.43, -0.42),
                arrowprops=dict(arrowstyle="<->", color="#555"))
    ax.text(0.405, -0.5, "QRS", ha="center", fontsize=10, color="#555")
    ax.set_xlabel("Час, с"); ax.set_ylabel("Амплітуда, мВ")
    ax.set_ylim(-0.6, 1.25); ax.grid(alpha=0.25)
    fig.savefig(os.path.join(OUT, "pqrst.png")); plt.close(fig)
    print("rendered pqrst.png")


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
    r = 1.25; pre, post = 90 / fs, 170 / fs
    ax.axvspan(r - pre, r + post, color="#f0c419", alpha=0.25)
    ax.axvline(r, color="#b03030", ls="--", lw=1.2)
    ax.annotate("R-пік", (r, 1.05), color="#b03030", ha="center", fontsize=11, fontweight="bold")
    ax.annotate("", xy=(r - pre, -0.45), xytext=(r, -0.45), arrowprops=dict(arrowstyle="<->", color="#555"))
    ax.annotate("", xy=(r, -0.45), xytext=(r + post, -0.45), arrowprops=dict(arrowstyle="<->", color="#555"))
    ax.text(r - pre / 2, -0.55, "90 відл.", ha="center", fontsize=9, color="#555")
    ax.text(r + post / 2, -0.55, "170 відл.", ha="center", fontsize=9, color="#555")
    ax.set_xlabel("Час, с"); ax.set_ylabel("Амплітуда (норм.)")
    ax.set_ylim(-0.7, 1.25); ax.grid(alpha=0.2)
    fig.savefig(os.path.join(OUT, "beat_window.png")); plt.close(fig)
    print("rendered beat_window.png")


if __name__ == "__main__":
    architecture(); hybrid(); cnn_arch(); pqrst(); beat_window()
    print("all diagrams written to", OUT)
