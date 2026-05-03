"""
Семінарське завдання №4 — Імовірнісне машинне навчання
Алгоритм Метрополіса для розподілу π(x) ∝ exp(-(x⁴ - x²))

Допоміжні розподіли:
  1. Нормальний:  q(x'|x) = N(x, σ²)
  2. Лапласа:     q(x'|x) = (λ/2) * exp(-λ|x' - x|)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.integrate import quad
import warnings

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 1. ЦІЛЬОВИЙ РОЗПОДІЛ
# ─────────────────────────────────────────────

def log_target(x: float) -> float:
    """
    Логарифм ненормованої щільності π(x) ∝ exp(-(x⁴ - x²)).
    Використовуємо логарифм для числової стабільності:
    log π(x) = -(x⁴ - x²) = x² - x⁴
    """
    return x ** 2 - x ** 4


def target(x: float) -> float:
    """Ненормована щільність π(x) = exp(-(x⁴ - x²))."""
    return np.exp(log_target(x))


# Нормуюча константа Z = ∫ π(x) dx  (для побудови теоретичної кривої)
Z, _ = quad(target, -np.inf, np.inf)


def target_pdf(x: np.ndarray) -> np.ndarray:
    """Нормована щільність для порівняння з гістограмою."""
    return target(x) / Z


# ─────────────────────────────────────────────
# 2. АЛГОРИТМ МЕТРОПОЛІСА
# ─────────────────────────────────────────────

def metropolis(
        n_samples: int,
        proposal: str = "normal",
        sigma: float = 1.0,
        lam: float = 1.0,
        x0: float = 0.0,
        seed: int = 42,
) -> tuple[np.ndarray, float]:
    """
    Алгоритм Метрополіса–Гастінгса.

    Параметри
    ----------
    n_samples : кількість ітерацій (довжина ланцюга)
    proposal  : 'normal' або 'laplace' — тип допоміжного розподілу
    sigma     : стандартне відхилення для нормального пропозалу
    lam       : параметр λ для розподілу Лапласа
    x0        : початкова точка ланцюга
    seed      : seed для відтворюваності

    Повертає
    -------
    samples   : масив з прийнятих / поточних значень
    acc_rate  : частота прийняття кроків
    """
    rng = np.random.default_rng(seed)
    samples = np.empty(n_samples)
    x_cur = x0
    accepted = 0

    for i in range(n_samples):
        # ── Крок 1: Генерація кандидата x' із допоміжного розподілу ──
        if proposal == "normal":
            # Симетричний пропозал: x' ~ N(x_cur, σ²)
            x_prop = rng.normal(loc=x_cur, scale=sigma)
        elif proposal == "laplace":
            # Симетричний пропозал: x' ~ Laplace(x_cur, 1/λ)
            # Лапласовий розподіл зі scale=1/λ: (λ/2)*exp(-λ|x'-x|)
            x_prop = rng.laplace(loc=x_cur, scale=1.0 / lam)
        else:
            raise ValueError("proposal має бути 'normal' або 'laplace'")

        # ── Крок 2: Коефіцієнт прийняття ──
        # Обидва пропозали симетричні → q(x|x') = q(x'|x)
        # Тому α = min(1, π(x') / π(x))
        # У логарифмах: log α = log π(x') - log π(x)
        log_alpha = log_target(x_prop) - log_target(x_cur)

        # ── Крок 3: Прийняти або відхилити ──
        if np.log(rng.uniform()) < log_alpha:
            x_cur = x_prop
            accepted += 1

        samples[i] = x_cur

    acc_rate = accepted / n_samples
    return samples, acc_rate


# ─────────────────────────────────────────────
# 3. ПАРАМЕТРИ ЕКСПЕРИМЕНТУ
# ─────────────────────────────────────────────

N_SAMPLES = 100_000  # довжина ланцюга
N_BURNIN = 5_000  # кількість ітерацій «розігріву» (burn-in)

# Обґрунтування σ:
#   π(x) має два моди ≈ ±0.707, відстань між ними ~1.41.
#   σ ≈ 1.0 — пропозал охоплює ~1 одиницю, частота прийняття ~44%
#   (оптимум для 1D за Roberts & Rosenthal, 1998), гарне mixing між модами.
#   Менші σ (0.1–0.3) дають ~90%+ прийняття, але повільне mixing.
#   Більші σ (3+) дають <10% прийняття — надто часто відхиляє кроки.
SIGMA = 1.0

# Обґрунтування λ:
#   Лаплас scale = 1/λ; λ ≈ 1.0 → scale = 1.0.
#   Важкі хвости Лапласа порівняно з N(0,1) дають більшу ймовірність
#   «стрибнути» між двома модами за один крок, що покращує mixing.
LAM = 1.0

# ─────────────────────────────────────────────
# 4. ЗАПУСК ОБОХ АЛГОРИТМІВ
# ─────────────────────────────────────────────

samples_norm, acc_norm = metropolis(
    N_SAMPLES, proposal="normal", sigma=SIGMA, seed=42
)
samples_lap, acc_lap = metropolis(
    N_SAMPLES, proposal="laplace", lam=LAM, seed=42
)

# Відкинути burn-in
sn = samples_norm[N_BURNIN:]
sl = samples_lap[N_BURNIN:]

print(f"Нормальний пропозал   | частота прийняття: {acc_norm:.3f}")
print(f"Лапласівський пропозал | частота прийняття: {acc_lap:.3f}")


# ─────────────────────────────────────────────
# 5. СТАТИСТИЧНИЙ ТЕСТ (Kolmogorov–Smirnov)
# ─────────────────────────────────────────────

# Генеруємо «еталонні» зразки через rejection sampling
# для порівняльного KS-тесту

def rejection_sampling(n: int, seed: int = 0) -> np.ndarray:
    """
    Rejection sampling для π(x) ∝ exp(-(x⁴ - x²)) на [-3, 3].
    Мажоруюча функція M = 1 (рівномірний розподіл на [-3,3]).
    """
    rng = np.random.default_rng(seed)
    # Максимум π(x) досягається в x≈±0.707: π_max = exp(0.25)
    pi_max = np.exp(0.25)
    result = []
    while len(result) < n:
        x = rng.uniform(-3, 3)
        u = rng.uniform(0, pi_max)
        if u <= target(x):
            result.append(x)
    return np.array(result)


ref_samples = rejection_sampling(50_000, seed=99)

# KS-тест: порівнюємо емпіричний CDF з «еталонним» розподілом
ks_norm = stats.ks_2samp(sn, ref_samples)
ks_lap = stats.ks_2samp(sl, ref_samples)

print(f"\n── KS-тест (порівняння з rejection sampling) ──")
print(f"Нормальний:   статистика={ks_norm.statistic:.4f}, p-значення={ks_norm.pvalue:.4f}")
print(f"Лапласівський: статистика={ks_lap.statistic:.4f}, p-значення={ks_lap.pvalue:.4f}")

# ─────────────────────────────────────────────
# 6. ГРАФІКИ
# ─────────────────────────────────────────────

x_grid = np.linspace(-2.5, 2.5, 1000)
pdf_true = target_pdf(x_grid)

FIG_W, FIG_H = 16, 18
fig = plt.figure(figsize=(FIG_W, FIG_H))
fig.patch.set_facecolor("#0f1117")

# Кольорова схема
C_NORM = "#4fc3f7"  # блакитний — нормальний
C_LAP = "#f48fb1"  # рожевий  — лапласівський
C_TRUE = "#a5d6a7"  # зелений  — теоретична крива
C_TXT = "#e0e0e0"
plt.rcParams.update({
    "text.color": C_TXT, "axes.labelcolor": C_TXT,
    "xtick.color": C_TXT, "ytick.color": C_TXT,
    "axes.edgecolor": "#444", "grid.color": "#333",
    "font.family": "DejaVu Sans",
})

gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.35)

# ── (A) Траєкторії ланцюгів ──────────────────
ax_tr1 = fig.add_subplot(gs[0, 0])
ax_tr2 = fig.add_subplot(gs[0, 1])

for ax, smp, color, label in [
    (ax_tr1, samples_norm[:3000], C_NORM, f"Нормальний σ={SIGMA}"),
    (ax_tr2, samples_lap[:3000], C_LAP, f"Лапласівський λ={LAM}"),
]:
    ax.set_facecolor("#1a1d27")
    ax.plot(smp, color=color, lw=0.5, alpha=0.8)
    ax.axhline(0, color="#555", lw=0.5)
    ax.axvline(N_BURNIN if N_BURNIN < 3000 else 3000,
               color="orange", lw=1.2, ls="--", label="burn-in межа")
    ax.set_title(f"Траєкторія: {label}", color=C_TXT, fontsize=11)
    ax.set_xlabel("Ітерація", fontsize=9)
    ax.set_ylabel("x", fontsize=9)
    ax.legend(fontsize=8, facecolor="#222", edgecolor="#555")
    ax.grid(True, ls=":", alpha=0.4)

# ── (B) Гістограми з теоретичною кривою ──────
ax_h1 = fig.add_subplot(gs[1, 0])
ax_h2 = fig.add_subplot(gs[1, 1])

for ax, smp, color, label, acc in [
    (ax_h1, sn, C_NORM, f"Нормальний σ={SIGMA}", acc_norm),
    (ax_h2, sl, C_LAP, f"Лапласівський λ={LAM}", acc_lap),
]:
    ax.set_facecolor("#1a1d27")
    ax.hist(smp, bins=120, density=True, color=color, alpha=0.55,
            label=f"MCMC ({acc:.1%} прийн.)")
    ax.plot(x_grid, pdf_true, color=C_TRUE, lw=2.2, label="π(x) теорет.")
    ax.set_xlim(-2.5, 2.5)
    ax.set_title(f"Гістограма: {label}", color=C_TXT, fontsize=11)
    ax.set_xlabel("x", fontsize=9)
    ax.set_ylabel("Щільність", fontsize=9)
    ax.legend(fontsize=8, facecolor="#222", edgecolor="#555")
    ax.grid(True, ls=":", alpha=0.4)

# ── (C) Автокореляція ──────────────────────────
ax_ac1 = fig.add_subplot(gs[2, 0])
ax_ac2 = fig.add_subplot(gs[2, 1])

MAX_LAG = 150


def autocorr(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Автокореляційна функція для лагів 0..max_lag."""
    x = x - x.mean()
    var = np.dot(x, x)
    ac = np.array([np.dot(x[:len(x) - k], x[k:]) / var for k in range(max_lag + 1)])
    return ac


lags = np.arange(MAX_LAG + 1)
ac_n = autocorr(sn, MAX_LAG)
ac_l = autocorr(sl, MAX_LAG)

for ax, ac, color, label in [
    (ax_ac1, ac_n, C_NORM, f"Нормальний σ={SIGMA}"),
    (ax_ac2, ac_l, C_LAP, f"Лапласівський λ={LAM}"),
]:
    ax.set_facecolor("#1a1d27")
    ax.bar(lags, ac, color=color, alpha=0.7, width=1.0)
    ax.axhline(0, color="#888", lw=0.8)
    ax.axhline(1.96 / np.sqrt(len(sn)), color="orange",
               lw=1, ls="--", label="95% CI")
    ax.axhline(-1.96 / np.sqrt(len(sn)), color="orange", lw=1, ls="--")
    ax.set_xlim(0, MAX_LAG)
    ax.set_title(f"Автокореляція: {label}", color=C_TXT, fontsize=11)
    ax.set_xlabel("Лаг", fontsize=9)
    ax.set_ylabel("ACF", fontsize=9)
    ax.legend(fontsize=8, facecolor="#222", edgecolor="#555")
    ax.grid(True, ls=":", alpha=0.4)

# ── (D) Кумулятивні середні (збіжність) ────────
ax_cm = fig.add_subplot(gs[3, :])
ax_cm.set_facecolor("#1a1d27")

true_mean = quad(lambda x: x * target(x), -np.inf, np.inf)[0] / Z  # ≈ 0

cum_n = np.cumsum(sn) / (np.arange(len(sn)) + 1)
cum_l = np.cumsum(sl) / (np.arange(len(sl)) + 1)

ax_cm.plot(cum_n, color=C_NORM, lw=1.2, label=f"Нормальний σ={SIGMA}")
ax_cm.plot(cum_l, color=C_LAP, lw=1.2, label=f"Лапласівський λ={LAM}")
ax_cm.axhline(true_mean, color=C_TRUE, lw=2, ls="--",
              label=f"Теор. середнє ≈ {true_mean:.4f}")
ax_cm.set_title("Збіжність кумулятивного середнього (після burn-in)", color=C_TXT, fontsize=11)
ax_cm.set_xlabel("Кількість зразків", fontsize=9)
ax_cm.set_ylabel("Кумулятивне середнє", fontsize=9)
ax_cm.legend(fontsize=9, facecolor="#222", edgecolor="#555")
ax_cm.grid(True, ls=":", alpha=0.4)

# ── Загальний заголовок ──────────────────────
fig.suptitle(
    "Алгоритм Метрополіса для π(x) ∝ exp(−(x⁴ − x²))\n"
    "Семінарське завдання №4 — Імовірнісне машинне навчання",
    fontsize=14, color=C_TXT, y=0.99,
)

plt.savefig("metropolis_results.png",
            dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("\nГрафік збережено → metropolis_results.png (в папці скрипту)")

# ─────────────────────────────────────────────
# 7. ДОДАТКОВІ СТАТИСТИКИ
# ─────────────────────────────────────────────

print("\n── Описова статистика (після burn-in) ──")
for name, smp in [("Нормальний", sn), ("Лапласівський", sl)]:
    print(f"\n{name}:")
    print(f"  Середнє:          {smp.mean():.5f}  (теор. ≈ {true_mean:.5f})")
    print(f"  Стандартне відх.: {smp.std():.5f}")
    print(f"  Медіана:          {np.median(smp):.5f}")
    print(f"  95% CI середнього: [{smp.mean() - 1.96 * smp.std() / np.sqrt(len(smp)):.5f}, "
          f"{smp.mean() + 1.96 * smp.std() / np.sqrt(len(smp)):.5f}]")


# Ефективний розмір вибірки (ESS) — груба оцінка через час змішування
def ess(samples: np.ndarray, max_lag: int = 500) -> float:
    """Effective Sample Size через автокореляцію."""
    ac = autocorr(samples, max_lag)
    # Підсумовуємо позитивну частину ACF (Geyer's truncation)
    tau = 1 + 2 * np.sum(ac[1:][ac[1:] > 0])
    return len(samples) / tau


ess_n = ess(sn)
ess_l = ess(sl)
print(f"\n── Ефективний розмір вибірки (ESS) ──")
print(f"Нормальний:    ESS ≈ {ess_n:.0f}  ({100 * ess_n / len(sn):.1f}% від {len(sn)})")
print(f"Лапласівський: ESS ≈ {ess_l:.0f}  ({100 * ess_l / len(sl):.1f}% від {len(sl)})")
