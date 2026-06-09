"""
plot_losses.py

Generate all loss curve images for AlmondT5 documentation.
Output: docs/assets/*.png
"""

import matplotlib.pyplot as plt
import os

os.makedirs("docs/assets", exist_ok=True)


C_ROPE_TRAIN   = "#4C9BE8"
C_ROPE_EVAL    = "#1A5C9E"
C_BIAS_TRAIN   = "#E87B4C"
C_BIAS_EVAL    = "#9E3A0D"

# 1. PRE-TRAINING LOSS - Ablation RoPE vs Relative Bias
pretrain_epochs   = list(range(1, 16))
eval_epochs       = [3, 6, 9, 12, 15]

rope_train  = [5.7161, 5.4899, 5.4062, 5.0223, 5.1650, 5.1657,
               4.9483, 5.0170, 4.9207, 4.7851, 4.8435, 4.6378,
               4.6583, 4.6163, 4.5941]
rope_eval   = [5.3152, 5.0150, 4.8264, 4.7259, 4.6998]

bias_train  = [5.8252, 5.5438, 5.3452, 5.4302, 5.2171, 5.0496,
               4.9179, 4.7745, 4.6357, 4.7408, 4.8037, 4.8107,
               4.7200, 4.8197, 4.8328]
bias_eval   = [5.3604, 5.0235, 4.8356, 4.7370, 4.7165]

fig, ax = plt.subplots(figsize=(11, 6))

ax.plot(pretrain_epochs, rope_train,
        color=C_ROPE_TRAIN, linewidth=2, marker="o", markersize=4,
        label="RoPE — Train")
ax.plot(eval_epochs, rope_eval,
        color=C_ROPE_EVAL, linewidth=2.5, marker="s", markersize=7,
        linestyle="--", label="RoPE — Eval")

ax.plot(pretrain_epochs, bias_train,
        color=C_BIAS_TRAIN, linewidth=2, marker="o", markersize=4,
        label="Relative Bias — Train")
ax.plot(eval_epochs, bias_eval,
        color=C_BIAS_EVAL, linewidth=2.5, marker="s", markersize=7,
        linestyle="--", label="Relative Bias — Eval")

ax.fill_between(pretrain_epochs, rope_train, alpha=0.06, color=C_ROPE_TRAIN)
ax.fill_between(pretrain_epochs, bias_train, alpha=0.06, color=C_BIAS_TRAIN)

ax.set_title("AlmondT5 — Pre-Training Loss\nRoPE vs Relative Position Bias (Span Corruption)",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Loss", fontsize=12)
ax.set_xticks(pretrain_epochs)
ax.set_ylim(4.0, 6.5)
ax.legend(fontsize=10)
ax.grid(True, linestyle="--", alpha=0.4)

ax.annotate(f"RoPE final: {rope_eval[-1]:.4f}",
            xy=(15, rope_eval[-1]), xytext=(-100, -20),
            textcoords="offset points", fontsize=9, color=C_ROPE_EVAL,
            arrowprops=dict(arrowstyle="->", color=C_ROPE_EVAL, lw=1.2))
ax.annotate(f"Rel.Bias final: {bias_eval[-1]:.4f}",
            xy=(15, bias_eval[-1]), xytext=(-110, 15),
            textcoords="offset points", fontsize=9, color=C_BIAS_EVAL,
            arrowprops=dict(arrowstyle="->", color=C_BIAS_EVAL, lw=1.2))

plt.tight_layout()
plt.savefig("docs/assets/pretrain_loss.png", dpi=150)
plt.close()
print("Saved: docs/assets/pretrain_loss.png")

# 2. FINE-TUNING LOSS - Ablation RoPE vs Relative Bias
finetune_epochs   = list(range(1, 16))
ft_eval_epochs    = [3, 6, 9, 12, 15]

rope_ft_train = [3.8261, 3.7770, 3.6185, 3.4092, 3.2916, 3.2982,
                 3.3972, 2.6553, 2.6651, 3.0271, 2.8013, 2.5692,
                 2.8922, 2.4617, 2.9392]
rope_ft_eval  = [3.7097, 3.5070, 3.4480, 3.4430, 3.4557]

bias_ft_train = [4.0810, 4.3571, 3.6536, 3.4289, 3.0977, 3.4994,
                 3.2314, 2.5818, 3.0532, 2.7153, 2.9070, 2.9532,
                 2.3358, 2.4435, 2.3236]
bias_ft_eval  = [3.6486, 3.4581, 3.4338, 3.4978, 3.5100]

fig, ax = plt.subplots(figsize=(11, 6))

ax.plot(finetune_epochs, rope_ft_train,
        color=C_ROPE_TRAIN, linewidth=2, marker="o", markersize=4,
        label="RoPE — Train")
ax.plot(ft_eval_epochs, rope_ft_eval,
        color=C_ROPE_EVAL, linewidth=2.5, marker="s", markersize=7,
        linestyle="--", label="RoPE — Eval")

ax.plot(finetune_epochs, bias_ft_train,
        color=C_BIAS_TRAIN, linewidth=2, marker="o", markersize=4,
        label="Relative Bias — Train")
ax.plot(ft_eval_epochs, bias_ft_eval,
        color=C_BIAS_EVAL, linewidth=2.5, marker="s", markersize=7,
        linestyle="--", label="Relative Bias — Eval")

ax.fill_between(finetune_epochs, rope_ft_train, alpha=0.06, color=C_ROPE_TRAIN)
ax.fill_between(finetune_epochs, bias_ft_train, alpha=0.06, color=C_BIAS_TRAIN)

ax.set_title("AlmondT5 — Fine-Tuning Loss (EN→ID Translation)\nRoPE vs Relative Position Bias",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Loss", fontsize=12)
ax.set_xticks(finetune_epochs)
ax.set_ylim(1.5, 5.0)
ax.legend(fontsize=10)
ax.grid(True, linestyle="--", alpha=0.4)

best_rope = min(rope_ft_eval)
best_bias = min(bias_ft_eval)
ax.annotate(f"RoPE best: {best_rope:.4f}",
            xy=(9, rope_ft_eval[2]), xytext=(-110, -25),
            textcoords="offset points", fontsize=9, color=C_ROPE_EVAL,
            arrowprops=dict(arrowstyle="->", color=C_ROPE_EVAL, lw=1.2))
ax.annotate(f"Rel.Bias best: {best_bias:.4f}",
            xy=(9, bias_ft_eval[2]), xytext=(10, -25),
            textcoords="offset points", fontsize=9, color=C_BIAS_EVAL,
            arrowprops=dict(arrowstyle="->", color=C_BIAS_EVAL, lw=1.2))

plt.tight_layout()
plt.savefig("docs/assets/finetune_loss.png", dpi=150)
plt.close()
print("Saved: docs/assets/finetune_loss.png")

# 3. BLEU / chrF Comparison Bar Chart
models  = ["Relative Bias", "RoPE"]
bleu    = [4.7118, 2.9945]
chrf    = [21.2895, 18.3580]

x = [0, 1]
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
bars1 = ax.bar([i - width/2 for i in x], bleu, width,
               label="BLEU", color=C_BIAS_TRAIN, alpha=0.85)
bars2 = ax.bar([i + width/2 for i in x], chrf, width,
               label="chrF", color=C_ROPE_TRAIN, alpha=0.85)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=10)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=10)

ax.set_title("AlmondT5 — Translation Evaluation\nBLEU & chrF: RoPE vs Relative Bias",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_ylim(0, 28)
ax.legend(fontsize=11)
ax.grid(True, axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("docs/assets/eval_scores.png", dpi=150)
plt.close()
print("Saved: docs/assets/eval_scores.png")

print("\nAll AlmondT5 plots generated.")