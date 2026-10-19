"""
Benteng Stelsel Optimizer (BSO) untuk benchmark fungsi standar
================================================================

Adaptasi kerangka Benteng Stelsel (De Kock, 1827-1829) dari versi
kriptanalisis Vigenere ke optimasi kontinu. Pemetaan tahapannya:

  Kriptanalisis Vigenere              ->  Optimasi kontinu (BSO)
  ----------------------------------------------------------------------
  [1] Pengintaian (Kasiski + IC)      ->  Inisialisasi + sebaran pasukan
  [2] Pembangunan benteng (kolom)     ->  K solusi terbaik dijadikan benteng
  [3] Pengepungan per kolom (chi^2)   ->  Siege per-dimensi (subset koordinat)
  [4] Jaringan benteng bertahap       ->  Benteng baru di antara dua benteng
  [5] reduce_to_minimal_period        ->  Penyempitan wilayah (K dan radius turun)
  [6] Pemutusan logistik              ->  Prajurit terburuk dikirim ulang

Jumlah benteng K dan radius patroli r menyusut seiring anggaran evaluasi,
meniru jaringan benteng yang makin rapat dan mengunci ruang gerak musuh.

Pembanding: PSO (inersia menurun), GA (real-coded, elitisme), DE/rand/1/bin.
Semua metode memakai ukuran populasi dan anggaran evaluasi fungsi (NFE)
yang sama, sehingga perbandingan adil.

Pemakaian:
    python benteng_stelsel_benchmark.py                   # default 30 run, D=30
    python benteng_stelsel_benchmark.py --runs 10 --dim 10 --evals 20000
    python benteng_stelsel_benchmark.py --funcs F1 F5 F9

Keluaran (folder results/):
    summary.csv              mean/std/best/worst/median/waktu per fungsi x metode
    wilcoxon.csv             uji Wilcoxon rank-sum BSO vs tiap pembanding
    table_results.tex        tabel LaTeX siap tempel ke paper (IEEEtran)
    convergence.png          kurva konvergensi rata-rata (log-scale)
    boxplot.png              sebaran fitness akhir
    raw_runs.csv             semua fitness akhir per run
"""

import argparse
import os
import time

import numpy as np

# ==========================================================================
# 1. FUNGSI BENCHMARK STANDAR (vektorisasi: X berbentuk (n, D))
# ==========================================================================


def sphere(X):
    return np.sum(X ** 2, axis=1)


def schwefel_222(X):
    a = np.abs(X)
    return np.sum(a, axis=1) + np.prod(a, axis=1)


def schwefel_12(X):
    return np.sum(np.cumsum(X, axis=1) ** 2, axis=1)


def rosenbrock(X):
    return np.sum(100.0 * (X[:, 1:] - X[:, :-1] ** 2) ** 2 + (X[:, :-1] - 1) ** 2, axis=1)


def step(X):
    return np.sum(np.floor(X + 0.5) ** 2, axis=1)


def rastrigin(X):
    return np.sum(X ** 2 - 10.0 * np.cos(2 * np.pi * X) + 10.0, axis=1)


def ackley(X):
    d = X.shape[1]
    s1 = np.sum(X ** 2, axis=1)
    s2 = np.sum(np.cos(2 * np.pi * X), axis=1)
    return -20.0 * np.exp(-0.2 * np.sqrt(s1 / d)) - np.exp(s2 / d) + 20.0 + np.e


def griewank(X):
    i = np.arange(1, X.shape[1] + 1)
    return np.sum(X ** 2, axis=1) / 4000.0 - np.prod(np.cos(X / np.sqrt(i)), axis=1) + 1.0


def schwefel_226(X):
    d = X.shape[1]
    return 418.9829 * d - np.sum(X * np.sin(np.sqrt(np.abs(X))), axis=1)


def zakharov(X):
    i = np.arange(1, X.shape[1] + 1)
    s = np.sum(0.5 * i * X, axis=1)
    return np.sum(X ** 2, axis=1) + s ** 2 + s ** 4


# kode: (nama, fungsi, lb, ub, tipe)
BENCHMARKS = {
    "F1": ("Sphere", sphere, -100, 100, "unimodal"),
    "F2": ("Schwefel 2.22", schwefel_222, -10, 10, "unimodal"),
    "F3": ("Schwefel 1.2", schwefel_12, -100, 100, "unimodal"),
    "F4": ("Rosenbrock", rosenbrock, -30, 30, "unimodal"),
    "F5": ("Step", step, -100, 100, "unimodal"),
    "F6": ("Zakharov", zakharov, -5, 10, "unimodal"),
    "F7": ("Rastrigin", rastrigin, -5.12, 5.12, "multimodal"),
    "F8": ("Ackley", ackley, -32, 32, "multimodal"),
    "F9": ("Griewank", griewank, -600, 600, "multimodal"),
    "F10": ("Schwefel 2.26", schwefel_226, -500, 500, "multimodal"),
}


class Problem:
    """Pembungkus fungsi yang menghitung NFE dan mencatat riwayat best-so-far."""

    def __init__(self, func, lb, ub, dim, max_evals):
        self.func = func
        self.lb = np.full(dim, float(lb))
        self.ub = np.full(dim, float(ub))
        self.dim = dim
        self.max_evals = max_evals
        self.evals = 0
        self.best = np.inf
        self.history = np.empty(max_evals)

    def __call__(self, X):
        X = np.atleast_2d(X)
        n_in = len(X)
        room = self.max_evals - self.evals
        if room <= 0:
            return np.full(n_in, np.inf)
        f = self.func(X[:room])
        bsf = np.minimum.accumulate(np.concatenate([[self.best], f]))[1:]
        self.history[self.evals:self.evals + len(f)] = bsf  # best-so-far
        self.best = bsf[-1]
        self.evals += len(f)
        if len(f) < n_in:  # anggaran habis di tengah batch
            f = np.concatenate([f, np.full(n_in - len(f), np.inf)])
        return f

    @property
    def done(self):
        return self.evals >= self.max_evals

    def clip(self, X):
        return np.clip(X, self.lb, self.ub)


# ==========================================================================
# 2. BENTENG STELSEL OPTIMIZER (BSO)
# ==========================================================================


def benteng_stelsel(prob, pop=30, rng=None, k0_ratio=0.2, siege_rate=0.3,
                    cut_ratio=0.1, stall_limit=20):
    """
    Parameter:
        pop         jumlah prajurit (populasi)
        k0_ratio    proporsi awal benteng terhadap populasi (K0 = pop * k0_ratio)
        siege_rate  peluang tiap dimensi ikut dikepung pada fase siege per-kolom
        cut_ratio   proporsi prajurit terburuk yang logistiknya diputus
        stall_limit iterasi tanpa perbaikan sebelum pemutusan logistik dipicu
    """
    rng = rng or np.random.default_rng()
    D, lb, ub = prob.dim, prob.lb, prob.ub
    span = ub - lb

    # [1] Pengintaian: sebar pasukan ke seluruh medan
    X = lb + rng.random((pop, D)) * span
    fit = prob(X)

    k0 = max(2, int(round(pop * k0_ratio)))
    n_cut = max(1, int(round(pop * cut_ratio)))
    stall, best_prev = 0, np.min(fit)

    while not prob.done:
        progress = prob.evals / prob.max_evals

        # [5] Penyempitan wilayah: benteng makin sedikit, radius makin kecil
        K = max(1, int(round(k0 * (1.0 - progress))))
        radius = 0.5 * span * np.exp(-8.0 * progress)

        # [2] Pembangunan benteng: K posisi terbaik
        order = np.argsort(fit)
        forts = X[order[:K]]

        k_idx = rng.integers(K, size=pop)
        F = forts[k_idx]
        phase = rng.random(pop)
        new = np.empty_like(X)

        # (a) Pengepungan: bergerak menuju benteng + patroli acak
        m = phase < 0.5
        if m.any():
            r = rng.random((m.sum(), 1))
            new[m] = X[m] + r * (F[m] - X[m]) + radius * rng.standard_normal((m.sum(), D)) * 0.3

        # (b) [3] Siege per-kolom: hanya sebagian dimensi yang diserang,
        #     disalin dari benteng lalu digeser (analog chi-square per kolom)
        m = (phase >= 0.5) & (phase < 0.8)
        if m.any():
            n = m.sum()
            mask = rng.random((n, D)) < siege_rate
            mask[np.arange(n), rng.integers(D, size=n)] = True  # min 1 kolom
            cand = X[m].copy()
            pert = F[m] + radius * rng.standard_normal((n, D))
            cand[mask] = pert[mask]
            new[m] = cand

        # (c) [4] Jaringan benteng: benteng baru di antara dua benteng
        m = phase >= 0.8
        if m.any():
            n = m.sum()
            a = forts[rng.integers(K, size=n)]
            b = forts[rng.integers(K, size=n)]
            w = rng.random((n, 1))
            new[m] = w * a + (1 - w) * b + 0.1 * radius * rng.standard_normal((n, D))

        new = prob.clip(new)
        new_fit = prob(new)
        better = new_fit < fit
        X[better], fit[better] = new[better], new_fit[better]

        # [6] Pemutusan logistik: kalau stagnan, prajurit terburuk dikirim
        #     ke posisi oposisi dari benteng utama atau dekat benteng utama
        cur_best = np.min(fit)
        stall = 0 if cur_best < best_prev - 1e-12 else stall + 1
        best_prev = min(best_prev, cur_best)
        if stall >= stall_limit and not prob.done:
            worst = np.argsort(fit)[-n_cut:]
            best_x = X[np.argmin(fit)]
            if rng.random() < 0.5:
                cand = lb + ub - X[worst] + rng.standard_normal((n_cut, D)) * radius
            else:
                cand = best_x + rng.standard_normal((n_cut, D)) * np.maximum(radius, 0.05 * span)
            cand = prob.clip(cand)
            cf = prob(cand)
            X[worst], fit[worst] = cand, cf
            stall = 0
    return prob.best


# ==========================================================================
# 3. PEMBANDING: PSO, GA, DE
# ==========================================================================


def pso(prob, pop=30, rng=None, c1=2.0, c2=2.0, w_max=0.9, w_min=0.4):
    rng = rng or np.random.default_rng()
    D, lb, ub = prob.dim, prob.lb, prob.ub
    vmax = 0.2 * (ub - lb)
    X = lb + rng.random((pop, D)) * (ub - lb)
    V = rng.uniform(-vmax, vmax, (pop, D))
    fit = prob(X)
    P, pfit = X.copy(), fit.copy()
    g = P[np.argmin(pfit)].copy()
    while not prob.done:
        w = w_max - (w_max - w_min) * prob.evals / prob.max_evals
        r1, r2 = rng.random((pop, D)), rng.random((pop, D))
        V = w * V + c1 * r1 * (P - X) + c2 * r2 * (g - X)
        V = np.clip(V, -vmax, vmax)
        X = prob.clip(X + V)
        fit = prob(X)
        imp = fit < pfit
        P[imp], pfit[imp] = X[imp], fit[imp]
        g = P[np.argmin(pfit)].copy()
    return prob.best


def ga(prob, pop=30, rng=None, pc=0.9, pm=None, elite=2, eta_c=20, eta_m=20):
    """GA real-coded: seleksi turnamen, SBX crossover, polynomial mutation."""
    rng = rng or np.random.default_rng()
    D, lb, ub = prob.dim, prob.lb, prob.ub
    pm = pm if pm is not None else 1.0 / D
    X = lb + rng.random((pop, D)) * (ub - lb)
    fit = prob(X)
    while not prob.done:
        order = np.argsort(fit)
        elites, efit = X[order[:elite]].copy(), fit[order[:elite]].copy()
        # turnamen biner
        a, b = rng.integers(pop, size=(2, pop))
        parents = np.where((fit[a] < fit[b])[:, None], X[a], X[b])
        children = parents.copy()
        # SBX
        for i in range(0, pop - 1, 2):
            if rng.random() < pc:
                u = rng.random(D)
                beta = np.where(u <= 0.5, (2 * u) ** (1 / (eta_c + 1)),
                                (1 / (2 * (1 - u))) ** (1 / (eta_c + 1)))
                p1, p2 = parents[i], parents[i + 1]
                children[i] = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
                children[i + 1] = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
        # polynomial mutation
        mmask = rng.random((pop, D)) < pm
        u = rng.random((pop, D))
        delta = np.where(u < 0.5, (2 * u) ** (1 / (eta_m + 1)) - 1,
                         1 - (2 * (1 - u)) ** (1 / (eta_m + 1)))
        children = np.where(mmask, children + delta * (ub - lb), children)
        children = prob.clip(children)
        n_child = pop - elite
        cfit = prob(children[:n_child])
        X = np.vstack([elites, children[:n_child]])
        fit = np.concatenate([efit, cfit])
    return prob.best


def de(prob, pop=30, rng=None, F=0.5, CR=0.9):
    """DE/rand/1/bin."""
    rng = rng or np.random.default_rng()
    D, lb, ub = prob.dim, prob.lb, prob.ub
    X = lb + rng.random((pop, D)) * (ub - lb)
    fit = prob(X)
    idx = np.arange(pop)
    while not prob.done:
        r = np.array([rng.choice(np.delete(idx, i), 3, replace=False) for i in idx])
        V = X[r[:, 0]] + F * (X[r[:, 1]] - X[r[:, 2]])
        cross = rng.random((pop, D)) < CR
        cross[idx, rng.integers(D, size=pop)] = True
        U = prob.clip(np.where(cross, V, X))
        ufit = prob(U)
        imp = ufit <= fit
        X[imp], fit[imp] = U[imp], ufit[imp]
    return prob.best


ALGORITHMS = {
    "BSO": benteng_stelsel,
    "PSO": pso,
    "GA": ga,
    "DE": de,
}


# ==========================================================================
# 4. EKSPERIMEN
# ==========================================================================


def run_experiment(funcs, dim, pop, max_evals, runs, out_dir, seed0=0):
    os.makedirs(out_dir, exist_ok=True)
    algs = list(ALGORITHMS)
    final = {f: {a: [] for a in algs} for f in funcs}
    times = {f: {a: [] for a in algs} for f in funcs}
    curves = {f: {a: np.zeros(max_evals) for a in algs} for f in funcs}

    for fk in funcs:
        name, func, lb, ub, _ = BENCHMARKS[fk]
        print(f"\n=== {fk}: {name}  (D={dim}, NFE={max_evals}, runs={runs}) ===")
        for a in algs:
            for r in range(runs):
                rng = np.random.default_rng(seed0 + r)
                prob = Problem(func, lb, ub, dim, max_evals)
                t0 = time.perf_counter()
                ALGORITHMS[a](prob, pop=pop, rng=rng)
                times[fk][a].append(time.perf_counter() - t0)
                final[fk][a].append(prob.best)
                curves[fk][a] += prob.history[:max_evals]
            curves[fk][a] /= runs
            v = np.array(final[fk][a])
            print(f"  {a:4s} mean={v.mean():.4e}  std={v.std():.4e}  "
                  f"best={v.min():.4e}  time={np.mean(times[fk][a]):.3f}s")

    save_outputs(funcs, algs, final, times, curves, dim, max_evals, runs, out_dir)
    return final, times, curves


def save_outputs(funcs, algs, final, times, curves, dim, max_evals, runs, out_dir):
    import csv

    # --- summary.csv dan raw_runs.csv
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Func", "Name", "Algorithm", "Mean", "Std", "Best", "Worst",
                    "Median", "Time_mean_s"])
        for fk in funcs:
            for a in algs:
                v = np.array(final[fk][a])
                w.writerow([fk, BENCHMARKS[fk][0], a, v.mean(), v.std(), v.min(),
                            v.max(), np.median(v), np.mean(times[fk][a])])
    with open(os.path.join(out_dir, "raw_runs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Func", "Algorithm", "Run", "FinalFitness", "Time_s"])
        for fk in funcs:
            for a in algs:
                for r, (v, t) in enumerate(zip(final[fk][a], times[fk][a])):
                    w.writerow([fk, a, r, v, t])

    # --- Wilcoxon rank-sum BSO vs pembanding (+ / = / -)
    try:
        from scipy.stats import ranksums
        has_scipy = True
    except ImportError:
        has_scipy = False
    wins = {a: [0, 0, 0] for a in algs if a != "BSO"}
    with open(os.path.join(out_dir, "wilcoxon.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Func", "Versus", "p_value", "Result(BSO)"])
        for fk in funcs:
            b = np.array(final[fk]["BSO"])
            for a in algs:
                if a == "BSO":
                    continue
                o = np.array(final[fk][a])
                p = ranksums(b, o).pvalue if has_scipy else np.nan
                if has_scipy and p < 0.05:
                    res = "+" if np.median(b) < np.median(o) else "-"
                else:
                    res = "="
                wins[a]["+=-".index(res)] += 1
                w.writerow([fk, a, p, res])

    # --- Ranking rata-rata (Friedman-style, berdasarkan mean)
    ranks = {a: [] for a in algs}
    for fk in funcs:
        means = np.array([np.mean(final[fk][a]) for a in algs])
        r = means.argsort().argsort() + 1
        for a, rv in zip(algs, r):
            ranks[a].append(rv)
    avg_rank = {a: np.mean(ranks[a]) for a in algs}

    # --- Tabel LaTeX (IEEEtran, table*)
    lines = [
        r"\begin{table*}[htbp]",
        rf"\caption{{Results on benchmark functions (D={dim}, NFE={max_evals}, {runs} runs). Best mean in bold.}}",
        r"\label{tab:benchmark}",
        r"\centering",
        r"\begin{tabular}{ll" + "c" * len(algs) + "}",
        r"\hline",
        r"\textbf{Func} & \textbf{Metric} & " + " & ".join(rf"\textbf{{{a}}}" for a in algs) + r" \\ \hline",
    ]
    for fk in funcs:
        means = [np.mean(final[fk][a]) for a in algs]
        bi = int(np.argmin(means))
        mrow = [(rf"\textbf{{{m:.3e}}}" if i == bi else f"{m:.3e}") for i, m in enumerate(means)]
        srow = [f"{np.std(final[fk][a]):.3e}" for a in algs]
        lines.append(rf"\multirow{{2}}{{*}}{{{fk}}} & Mean & " + " & ".join(mrow) + r" \\")
        lines.append(r" & Std & " + " & ".join(srow) + r" \\ \hline")
    lines.append(r"Avg. rank & & " + " & ".join(f"{avg_rank[a]:.2f}" for a in algs) + r" \\")
    wl = ["--"] + [f"{wins[a][0]}/{wins[a][1]}/{wins[a][2]}" for a in algs if a != "BSO"]
    lines.append(r"BSO +/=/$-$ & & " + " & ".join(wl) + r" \\ \hline")
    lines += [r"\end{tabular}", r"\end{table*}"]
    with open(os.path.join(out_dir, "table_results.tex"), "w") as f:
        f.write("% butuh \\usepackage{multirow}\n" + "\n".join(lines) + "\n")

    # --- Grafik
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib tidak ada, grafik dilewati.")
        return

    colors = {"BSO": "#2a9d8f", "PSO": "#3a6ea5", "GA": "#e76f51", "DE": "#8d6cab"}
    n = len(funcs)
    cols = min(5, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.8 * rows), squeeze=False)
    x = np.arange(1, max_evals + 1)
    for ax, fk in zip(axes.flat, funcs):
        for a in algs:
            y = np.maximum(curves[fk][a], 1e-300)
            ax.semilogy(x, y, label=a, color=colors.get(a), lw=1.4)
        ax.set_title(f"{fk}: {BENCHMARKS[fk][0]}", fontsize=9)
        ax.set_xlabel("NFE", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    axes.flat[0].set_ylabel("Best fitness (mean)", fontsize=8)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(algs), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(os.path.join(out_dir, "convergence.png"), dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.8 * rows), squeeze=False)
    for ax, fk in zip(axes.flat, funcs):
        data = [np.maximum(final[fk][a], 1e-300) for a in algs]
        try:  # matplotlib >= 3.9
            bp = ax.boxplot(data, tick_labels=algs, patch_artist=True, widths=0.6)
        except TypeError:  # matplotlib lama (mis. Colab versi lama)
            bp = ax.boxplot(data, labels=algs, patch_artist=True, widths=0.6)
        for patch, a in zip(bp["boxes"], algs):
            patch.set_facecolor(colors.get(a))
            patch.set_alpha(0.6)
        ax.set_yscale("log")
        ax.set_title(f"{fk}: {BENCHMARKS[fk][0]}", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, axis="y")
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "boxplot.png"), dpi=200)
    plt.close(fig)

    print(f"\nRata-rata ranking: " + ", ".join(f"{a}={avg_rank[a]:.2f}" for a in algs))
    for a in wins:
        print(f"BSO vs {a}: +{wins[a][0]} ={wins[a][1]} -{wins[a][2]}")
    print(f"Semua keluaran disimpan di folder: {out_dir}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Benchmark Benteng Stelsel Optimizer")
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--dim", type=int, default=30)
    ap.add_argument("--pop", type=int, default=30)
    ap.add_argument("--evals", type=int, default=30000, help="anggaran NFE per run")
    ap.add_argument("--funcs", nargs="+", default=list(BENCHMARKS))
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run_experiment(args.funcs, args.dim, args.pop, args.evals, args.runs, args.out, args.seed)
