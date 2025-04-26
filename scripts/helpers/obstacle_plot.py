import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update({"xtick.labelsize": 24, "ytick.labelsize": 24})

categories = ["MPD", "VAE", "FP", "FP + split"]
values = [0.03, 0.05, 0.15, 0.2]

fig, ax = plt.subplots(figsize=(12, 8))
# ax.set_facecolor("lightgray")

colormap = plt.get_cmap("Blues")
colors = colormap(np.linspace(0.2, 0.8, len(categories)))
bars = ax.bar(categories, values, color=colors, edgecolor="black", linewidth=1.5)

ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, integer=True))
ax.set_ylabel("Max obstacle size", fontsize=24)

ax.spines["top"].set_visible(False)  # Remove top border
ax.spines["right"].set_visible(False)  # Remove right border

ax.yaxis.grid(True, linestyle="--", alpha=0.7, color="gray")
ax.xaxis.grid(True, linestyle="--", alpha=0.7, color="gray")
ax.set_axisbelow(True)  # Ensure grid is behind bars

plt.tight_layout()

# plt.savefig("bar_chart_figure.pdf", format="pdf", bbox_inches="tight")
plt.savefig("bar_chart_figure.png", format="png", bbox_inches="tight")
# plt.show()
