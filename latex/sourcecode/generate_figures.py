import numpy as np
import matplotlib.pylab as plt
from matplotlib import gridspec
import ot
import ot.plot
import os
import qrcode

###############################################################################

path = os.path.dirname(__file__)
path = os.path.abspath(path)+"/"

f = open("presentation.tex", "r")
preamble = f.read()
preamble = preamble.replace("\\input{", "\\input{"+path)

plt.rcParams.update({
    "font.size": 12,
    "font.family": "sans-serif",
    "text.usetex": True,
    "pgf.rcfonts": False,
    "text.latex.preamble": preamble,
    "pgf.preamble": preamble,
})

###############################################################################

WHITE = "#FFFFFF"
BLACK = "#000000"
BLUE = "#0077BB"
CYAN = "#009988"
GREEN = "#009988"
ORANGE = "#EE7733"
RED = "#CC3311"
MAGENTA = "#EE3377"
GREY = "#BBBBBB"

###############################################################################
# Inspired from
# https://pythonot.github.io/auto_examples/plot_OT_2D_samples.html#sphx-glr-auto-examples-plot-ot-2d-samples-py


# Create the data
mu_s = np.array([0, 0])
cov_s = np.array([[1, 0], [0, 1]])
xs = ot.datasets.make_2D_samples_gauss(3, mu_s, cov_s, random_state=10)

xt = np.array([
    [-0.1, 0.3],
    [1.6, 0.1]])


###############################################################################
# Fig 1 - Prior distribution


# Loss surface
# from https://matplotlib.org/stable/plot_types/arrays/contourf.html
X, Y = np.meshgrid(np.linspace(-3, 3, 256), np.linspace(-3, 3, 256))
Z = (1 - X/2 + X**5 + Y**3) * np.exp(-X**2 - Y**2)
Z = (Z - np.min(Z))/(np.max(Z)-np.min(Z))
Z = 1.0-Z
levels = np.linspace(np.min(Z), np.max(Z), 7)

plt.figure(figsize=(11.69*0.4, 8.27*0.4))
plt.plot(xs[:, 0], xs[:, 1], "o", color=GREEN)
plt.contourf(X, Y, Z, levels=levels, cmap="Greys", alpha=0.7)

plt.tick_params(
    left=False, right=False, bottom=False, top=False,
    labelleft=False, labelright=False, labelbottom=False, labeltop=False)
os.makedirs("figures", exist_ok=False)
plt.savefig("figures/prior_dist.pdf", bbox_inches="tight")


###############################################################################
# Fig 2 - Posterior distribution


# Loss surface
# from https://matplotlib.org/stable/plot_types/arrays/contourf.html
X, Y = np.meshgrid(np.linspace(-3, 3, 256), np.linspace(-3, 3, 256))
Z = (1 - X/2 + X**5 + Y**3) * np.exp(-X**2 - Y**2)
Z = (Z - np.min(Z))/(np.max(Z)-np.min(Z))
Z = 1.0-Z
levels = np.linspace(np.min(Z), np.max(Z), 7)

plt.figure(figsize=(11.69*0.5, 8.27*0.4))
plt.plot(xt[:, 0], xt[:, 1], "^", color=ORANGE)
contour = plt.contourf(X, Y, Z, levels=levels, cmap="Greys", alpha=0.7)
plt.colorbar(contour)

plt.tick_params(
    left=False, right=False, bottom=False, top=False,
    labelleft=False, labelright=False, labelbottom=False, labeltop=False)
plt.savefig("figures/post_dist.pdf", bbox_inches="tight")

###############################################################################
# Fig 3 - KL divergence


def gibbs(x, f_alpha=None):
    if(f_alpha is None):
        f_alpha = np.zeros(x.shape)
    new_f_alpha = f_alpha[np.logical_not(np.isnan(f_alpha))]
    pdf = np.exp(-new_f_alpha)/np.sum(np.exp(-new_f_alpha))
    f_alpha[np.logical_not(np.isnan(f_alpha))] = pdf
    f_alpha[np.isnan(f_alpha)] = 0.0
    return f_alpha


x = np.arange(0, 5)
prior_list = np.array([1.0, 1.0, 1.0, np.nan, np.nan])
post_list = np.array([np.nan, np.nan, np.nan, 1.0, 1.0])
dist_dict = {
    "prior": gibbs(x, f_alpha=1.0*prior_list),
    "post": gibbs(x, f_alpha=1.0*post_list),
}
y_max = np.max(np.concatenate([
    dist_dict["prior"], dist_dict["post"]]))


for dist in ["prior", "post"]:
    fig, ax = plt.subplots(1, 1, figsize=(6, 1))

    if(dist == "prior"):
        ax.bar(x, dist_dict["prior"], width=0.8, color=GREEN, alpha=0.8)
        ax.bar(x, dist_dict["prior"], width=0.8, fill=None,
               edgecolor=GREEN, linewidth=1.0)
    elif(dist == "post"):
        ax.bar(x, dist_dict["post"], width=0.8, color=ORANGE, alpha=0.8)
        ax.bar(x, dist_dict["post"], width=0.8, fill=None,
               edgecolor=ORANGE, linewidth=1.0)

    ax.set_ylim(0.0, y_max)
    if(dist == "prior"):
        ax.set_ylabel(r"Prior $\P(\h)$")
    elif(dist == "post"):
        ax.set_ylabel(r"Posterior $\Q(\h)$")

    ax.set_xticks(x, [r"$\h_1$", r"$\h_2$", r"$\h_3$", r"$\h_4$", r"$\h_5$"])

    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.tick_params(left=False, labelleft=False,
                   bottom=False)

    os.makedirs("figures/", exist_ok=True)
    fig.savefig(f"figures/distribution_{dist}.pdf", bbox_inches="tight")


###############################################################################
# Fig 4 - Optimal transport (Loss)

# Loss surface
# from https://matplotlib.org/stable/plot_types/arrays/contourf.html
X, Y = np.meshgrid(np.linspace(-3, 3, 256), np.linspace(-3, 3, 256))
Z = (1 - X/2 + X**5 + Y**3) * np.exp(-X**2 - Y**2)
levels = np.linspace(np.min(Z), np.max(Z), 7)

# Compute the optimal transport
M = ot.dist(xs, xt)
lambd = 1e-2
a, b = np.ones((3,))/3, np.ones((2,))/2
Gs = ot.sinkhorn(a, b, M, lambd)

# Show the transport!
plt.figure(figsize=(11.69*0.4, 8.27*0.4))
ot.plot.plot2D_samples_mat(xs, xt, Gs, color="black")
plt.plot(xs[:, 0], xs[:, 1], "o", color=GREEN)
plt.plot(xt[:, 0], xt[:, 1], "^", color=ORANGE)
plt.contourf(X, Y, Z, levels=levels, cmap="Greys_r", alpha=0.7)

plt.tick_params(
    left=False, right=False, bottom=False, top=False,
    labelleft=False, labelright=False, labelbottom=False, labeltop=False)

plt.savefig("figures/ot_loss.pdf", bbox_inches="tight")

###############################################################################
# Fig 4 - Optimal transport (Gamma)

plt.figure(figsize=(8.27*0.4, 8.27*0.4))
gs = gridspec.GridSpec(3, 3)

ax_1 = plt.subplot(gs[0, 1:])
fig_1 = ax_1.figure
ax_1.bar([0, 1], [0.5, 0.5], width=0.8, color=ORANGE, alpha=0.8)
ax_1.bar([0, 1], [0.5, 0.5], width=0.8, fill=None,
         edgecolor=ORANGE, linewidth=1.0)
ax_1.set_yticks(())
ax_1.tick_params(
    left=False, right=True, bottom=True, top=False,
    labelleft=False, labelright=False, labelbottom=False, labeltop=False)

ax_2 = plt.subplot(gs[1:, 0])
fig_2 = ax_2.figure
ax_2.barh([0, 1, 2], [0.333, 0.333, 0.333], color=GREEN, alpha=0.8)
ax_2.barh([0, 1, 2], [0.333, 0.333, 0.333], fill=None,
          edgecolor=GREEN, linewidth=1.0)
fig_2.gca().invert_xaxis()
ax_2.set_xticks(())
ax_2.tick_params(
    left=False, right=True, bottom=True, top=False,
    labelleft=False, labelright=False, labelbottom=False, labeltop=False)

X, Y = np.meshgrid(np.arange(0, 2), np.arange(0, 3))

ax = plt.subplot(gs[1:, 1:], sharex=ax_1, sharey=ax_2)
ax.grid(True, zorder=-100)

ax.scatter(X, Y, s=1000*Gs, color=BLACK, linewidth=2, zorder=100)
ax.set_xticks([0, 1], [r"$h_4$", r"$h_5$"])
ax.set_yticks([0, 1, 2], [r"$h_1$", r"$h_2$", r"$h_3$"])

ax.set_xlim([-1.0, 2.0])
ax.set_ylim([-0.5, 2.5])

ax.tick_params(
    left=False, right=True, bottom=True, top=False,
    labelleft=False, labelright=True, labelbottom=True, labeltop=False)

plt.subplots_adjust(wspace=0.0, hspace=0.0)
plt.savefig("figures/ot_gamma.pdf", bbox_inches="tight")

###############################################################################
# Fig 5 - QR code

qr = qrcode.QRCode()
qr.add_data("https://arxiv.org/abs/2306.04375")
qr.make()
img = qr.make_image(fill_color=(0, 0, 0), back_color=(255, 255, 255))
img.save("figures/qrcode.png")
