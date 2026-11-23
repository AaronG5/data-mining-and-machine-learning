"""Grafikai (stilius kaip DBSCAN/)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_samples


def spalva(z):
   return f"C{int(z) % 10}"


def piesti(ax, Z, zymes, pavadinimas, asiu_vardas, vardai=None):
   for z in np.unique(zymes):
      m = zymes == z
      ax.scatter(
         Z[m, 0], Z[m, 1], s=8, alpha=0.7, color=spalva(z), label=vardai[z] if vardai is not None else f"klasteris {z}"
      )
   ax.set(title=pavadinimas, xlabel=f"{asiu_vardas} 1", ylabel=f"{asiu_vardas} 2")
   ax.legend(markerscale=2, fontsize=8)


def k_kreives(ax_alk, ax_sil, lentele, metodu_k, k_pasirinktas):
   """Raudonas taškas - metodo siūlomas k, pilka linija - pasirinktas k."""
   for ax, stulp, ylab, pav in [
      (ax_alk, "WCSS", "WCSS", "Elbow metodas"),
      (ax_sil, "Silhouette", "Vidutinis silueto koeficientas", "Vidutinio silueto metodas"),
   ]:
      k = metodu_k[stulp if stulp == "Silhouette" else "Elbow"]
      ax.plot(lentele["k"], lentele[stulp], marker="o", lw=1.2)
      ax.scatter([k], lentele.loc[lentele["k"] == k, stulp], c="red", s=60, zorder=3, label=f"metodo siūlomas k = {k}")
      ax.axvline(k_pasirinktas, ls="--", c="grey", lw=1, label=f"pasirinktas k = {k_pasirinktas}")
      ax.set(title=pav, xlabel="Klasterių skaičius k", ylabel=ylab, xticks=lentele["k"])
      ax.grid(alpha=0.3)
      ax.legend(fontsize=8)


def silueto_grafikas(ax, X, zymes, pavadinimas):
   reiksmes = np.array(silhouette_samples(X, zymes))
   apacia = 10
   for z in np.unique(zymes):
      r = np.sort(reiksmes[zymes == z])
      ax.fill_betweenx(np.arange(apacia, apacia + len(r)), 0, r, color=spalva(z), alpha=0.7)
      ax.text(
         -0.01, apacia + len(r) / 2, str(z), ha="right", va="center", fontsize=9, transform=ax.get_yaxis_transform()
      )
      apacia += len(r) + 10
   ax.axvline(reiksmes.mean(), c="red", ls="--", lw=1, label=f"vidurkis = {reiksmes.mean():.3f}")
   ax.set(
      title=pavadinimas,
      xlabel="Silueto koeficientas",
      ylabel="Klasteris",
      yticks=[],
      xlim=(min(-0.1, reiksmes.min() - 0.05), 1),
   )
   ax.legend(fontsize=8, loc="lower right")
   ax.grid(alpha=0.3, axis="x")


def issaugoti(kelias):
   plt.tight_layout()
   plt.savefig(kelias, dpi=150)
   plt.close()
