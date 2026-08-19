"""
Figure 2.1 -- CONSORT-style participant flow diagram.
31 recruited -> 30 published (1 KOOS exclusion) -> 29 analytic cohort
(3 discarded, 1 missing IMU), with trial-level dispositions annotated.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

BASE = r"c:\Users\Deepak Babu P K\OneDrive\Desktop\Laura_Handover\PythonAnalysis"
OUT_PNG = f"{BASE}\\plots\\figure2_1_consort_flow.png"

fig, ax = plt.subplots(figsize=(8, 11))
ax.set_xlim(0, 10)
ax.set_ylim(0, 15)
ax.axis("off")

box_style = dict(boxstyle="round,pad=0.5", facecolor="#eaf2f5", edgecolor="#2a6f7f", linewidth=1.5)
excl_style = dict(boxstyle="round,pad=0.4", facecolor="#fbeaea", edgecolor="#c0392b", linewidth=1.2)

def box(x, y, text, style=box_style, width=5.6, fontsize=10.5, weight="normal"):
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, fontweight=weight,
             bbox=style, wrap=True, zorder=3)

def arrow(x, y0, y1):
    ax.annotate("", xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.5), zorder=2)

def side_excl(x_main, y, text):
    ax.annotate("", xy=(x_main + 3.1, y), xytext=(x_main + 1.6, y),
                arrowprops=dict(arrowstyle="-|>", color="#c0392b", lw=1.2), zorder=2)
    box(x_main + 4.9, y, text, style=excl_style, width=3.2, fontsize=9)

X = 3.2

box(X, 14, "Recruited\nN = 31\n(14 young adults 18-35y, 17 older adults 50-75y)", weight="bold")
arrow(X, 13.3, 12.3)

box(X, 11.7, "Published cohort (Burke et al. 2024)\nN = 30")
side_excl(X, 12.5, "1 young participant excluded\n(KOOS screening, threshold 85)")
arrow(X, 11.0, 9.7)

box(X, 9.0, "Final analytic cohort (present reprocessing)\nN = 29", weight="bold")
arrow(X, 8.3, 5.3)
side_excl(X, 10.6, "IDs 23, 28, 34 -- present in the source\ndata structure but discarded in the\noriginal MATLAB pipeline, never part of\nthe officially recruited/published counts")
side_excl(X, 9.8, "Participant 1 excluded --\nno corresponding Xsens IMU files")

ax.text(X, 6.3,
        "NOTE: the thesis text does not state whether IDs 23/28/34 are counted\n"
        "within the 31 recruited / 30 published totals above, or exist as separate\n"
        "IDs in the raw data directory outside that numbering. Verify this against\n"
        "your own records before finalising this figure.",
        ha="center", va="center", fontsize=8.5, style="italic", color="#c0392b",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff8e1", edgecolor="#c0392b", linewidth=0.8))

box(X, 4.7, "Trial-level exclusions within the N=29 cohort\n(not participant-level)", width=6.2, fontsize=10, weight="bold")
notes = (
    "P11: walking-incline & step-forward trials excluded\n(severe EMG sensor dropout)\n\n"
    "P13: incline-walking Xsens substituted with flat-walking\n(matches original MATLAB pipeline)\n\n"
    "P27: step-lateral EMG flagged as duplicate of\nwalking-EMG (SHA256-confirmed, caveat only)\n\n"
    "P29: incline-walking / step-forward Xsens swapped\n(matches original MATLAB pipeline)\n\n"
    "P33: step-lateral EMG corrected to load own file\n(was substituted with P3's data in error)"
)
box(X, 2.2, notes, width=6.4, fontsize=9)

fig.suptitle("Figure 2.1 — Participant Flow Diagram", fontsize=13, fontweight="bold", y=0.97)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT_PNG, dpi=150)
print(f"Saved -> {OUT_PNG}")
