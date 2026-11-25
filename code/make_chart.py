# -*- coding: utf-8 -*-
import csv, os
os.environ["MPLCONFIGDIR"] = os.path.abspath("../../mpl_tmp")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# Windows 中文字体
for f in ("Microsoft YaHei", "SimHei"):
    if any(f.lower() == x.name.lower() for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [f]
        break
plt.rcParams["axes.unicode_minus"] = False

rows = list(csv.DictReader(open("results/summary.csv", encoding="utf-8")))
ids = sorted({r["id"] for r in rows})
base = {r["id"]: int(r["correct"]) for r in rows if r["mode"] == "baseline"}
react = {r["id"]: int(r["correct"]) for r in rows if r["mode"] == "react"}
hops = {r["id"]: int(r["hop" + "s"]) for r in rows if r["mode"] == "react"}

acc_b = sum(base.values()) / len(base)
acc_r = sum(react.values()) / len(react)

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1, 2.4]})

# 左：总体准确率
ax = axes[0]
bars = ax.bar(["单次决策基线", "ReAct 循环"], [acc_b, acc_r],
              color=["#b0b7c3", "#2f6fb2"], width=0.55)
for b, v, n in zip(bars, [acc_b, acc_r], [sum(base.values()), sum(react.values())]):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.1%}\n({n}/12)",
            ha="center", va="bottom", fontsize=11)
ax.set_ylim(0, 1.18)
ax.set_ylabel("准确率")
ax.set_title("总体准确率（12 题，迷你 HotpotQA 环境）", fontsize=12)
ax.spines[["top", "right"]].set_visible(False)

# 右：逐题对比
ax = axes[1]
xs = range(len(ids))
w = 0.38
ax.bar([x - w / 2 for x in xs], [base[i] for i in ids], w,
       label="单次决策基线", color="#b0b7c3")
ax.bar([x + w / 2 for x in xs], [react[i] for i in ids], w,
       label="ReAct 循环", color="#2f6fb2")
hop_labels = [f"{i}\n({hops[i]}跳)" for i in ids]
ax.set_xticks(list(xs))
ax.set_xticklabels(hop_labels, fontsize=9)
ax.set_yticks([0, 1])
ax.set_yticklabels(["错", "对"])
ax.set_ylim(0, 1.35)
ax.set_xlabel("题目（跳数）")
ax.set_title("逐题对比：ReAct 通过多步检索修复了全部 2 跳与 3 跳失败（仅 Q12 仍错）", fontsize=12)
ax.legend(loc="upper right", frameon=False)
ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("mini-ReAct 复现结果  |  单次决策 16.7% → ReAct 91.7%（+75pp）", fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig("results/results-chart.png", dpi=150, bbox_inches="tight")
print("saved results-chart.png")
