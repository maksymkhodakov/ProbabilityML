"""
Семінарське завдання №2
Імовірнісне машинне навчання

Завдання:
  1. 2D Gaussian μ=(1,1), R=I  — байєсівський лінійний + оптимальний класифікатор
  2. 2D Gaussian μ=(0,1) з 4 різними кореляційними матрицями
  3. 3D Gaussian з 4 різними математичними сподіваннями, R=I
  4. 3D Gaussian μ=(1,1,1) з 4 різними кореляційними матрицями
  7. Візуалізація результатів
  8. Перевірка гіпотез про закон розподілу
  9. Набір даних із 5 ознак (сильна/слабка/нульова кореляція)
"""

import logging
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Ellipse
from scipy import stats
from scipy.stats import chi2, kstest, multivariate_normal, norm, shapiro

# ─── Налаштування логування ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("seminar2.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ─── Глобальні параметри ───────────────────────────────────────────────────
N = 500          # кількість реалізацій у кожному експерименті
SEED = 42        # seed для відтворюваності результатів
np.random.seed(SEED)

matplotlib.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"] = 10


# ══════════════════════════════════════════════════════════════════════════════
# ЗАВДАННЯ 1
# Змоделювати N реалізацій гауссівського вектора: μ=(1,1), R=I
# Побудувати байєсівський лінійний та оптимальний класифікатори
# Оцінити похибку класифікації
# ══════════════════════════════════════════════════════════════════════════════
def task1() -> None:
    logger.info("=" * 65)
    logger.info("ЗАВДАННЯ 1 — 2D Gaussian, μ=(1,1), R=I")
    logger.info("Байєсівський лінійний та оптимальний класифікатори")
    logger.info("=" * 65)

    # Два класи з однаковою коваріаційною матрицею (R=I) і різними μ.
    # Клас 0 симетричний до класу 1 відносно початку координат.
    mu1 = np.array([1.0, 1.0])   # математичне сподівання класу 1
    mu0 = np.array([-1.0, -1.0]) # математичне сподівання класу 0
    R = np.eye(2)                # кореляційна матриця (одинична)

    logger.info(f"Клас 1 : μ₁ = {mu1}")
    logger.info(f"Клас 0 : μ₀ = {mu0}")
    logger.info(f"R = I (однакова для обох класів)")
    logger.info(f"N = {N} (по {N//2} зразків на клас)")

    # ── Генерація вибірки ────────────────────────────────────────────────────
    X0 = np.random.multivariate_normal(mu0, R, N // 2)
    X1 = np.random.multivariate_normal(mu1, R, N // 2)
    X  = np.vstack([X0, X1])          # матриця ознак (N × 2)
    y  = np.hstack([np.zeros(N // 2), # мітки класів
                    np.ones(N // 2)])

    logger.info(f"Вибірку згенеровано: {X.shape[0]} зразків")

    # ── Байєсівський ЛІНІЙНИЙ класифікатор (LDA) ────────────────────────────
    # При однакових коваріаційних матрицях межа рішення — лінійна:
    #   w = R⁻¹ (μ₁ − μ₀)
    #   рішення: w ᵀ x ≥ θ  →  клас 1, інакше клас 0
    #   θ = ½ w ᵀ (μ₁ + μ₀)  (для рівних апріорних ймовірностей)
    R_inv = np.linalg.inv(R)
    w_lin  = R_inv @ (mu1 - mu0)
    theta  = 0.5 * w_lin @ (mu1 + mu0)

    logger.info(f"\n[Лінійний класифікатор (LDA)]")
    logger.info(f"  Ваговий вектор  w = R⁻¹(μ₁−μ₀) = {w_lin}")
    logger.info(f"  Поріг θ = ½·w ᵀ(μ₁+μ₀) = {theta:.4f}")

    scores_lin    = X @ w_lin
    y_pred_lin    = (scores_lin >= theta).astype(int)
    error_lin     = np.mean(y_pred_lin != y)

    logger.info(f"  Похибка класифікації = {error_lin:.4f}  ({error_lin*100:.2f}%)")

    # ── Оптимальний байєсівський класифікатор (MAP) ──────────────────────────
    # Класифікуємо за максимальною апостеріорною ймовірністю:
    #   log p(x|C=1) vs log p(x|C=0)
    # При однакових коваріаціях і рівних апріорних ймовірностях
    # рішення збігається з лінійним, але реалізація — через повні густини.
    log_p0 = multivariate_normal.logpdf(X, mean=mu0, cov=R)
    log_p1 = multivariate_normal.logpdf(X, mean=mu1, cov=R)
    y_pred_opt = (log_p1 >= log_p0).astype(int)
    error_opt  = np.mean(y_pred_opt != y)

    logger.info(f"\n[Оптимальний класифікатор (MAP/Байєс)]")
    logger.info(f"  Правило: argmax_k  log p(x|C=k)")
    logger.info(f"  Похибка класифікації = {error_opt:.4f}  ({error_opt*100:.2f}%)")

    # ── Теоретична похибка Байєса ─────────────────────────────────────────────
    # P_e = Φ(−d/2), де d — відстань Махаланобіса між класами
    d_mahala = np.sqrt((mu1 - mu0) @ R_inv @ (mu1 - mu0))
    p_bayes  = norm.cdf(-d_mahala / 2)

    logger.info(f"\n[Теоретична оцінка]")
    logger.info(f"  Відстань Махаланобіса d = {d_mahala:.4f}")
    logger.info(f"  Теоретична похибка Байєса P_e = Φ(−d/2) = {p_bayes:.4f}  ({p_bayes*100:.2f}%)")

    # ── Візуалізація ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (title, y_pred, err) in zip(axes, [
        ("Лінійний класифікатор (LDA)",      y_pred_lin, error_lin),
        ("Оптимальний класифікатор (MAP)",    y_pred_opt, error_opt),
    ]):
        # Точки двох класів (істинні мітки)
        ax.scatter(X[y == 0, 0], X[y == 0, 1],
                   c="royalblue", alpha=0.4, s=18, label="Клас 0 (істинний)")
        ax.scatter(X[y == 1, 0], X[y == 1, 1],
                   c="tomato", alpha=0.4, s=18, label="Клас 1 (істинний)")

        # Межа рішення (hyperplane: w ᵀ x = θ)
        x1_range = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 400)
        # w[0]*x1 + w[1]*x2 = θ  →  x2 = (θ − w[0]*x1) / w[1]
        x2_boundary = (theta - w_lin[0] * x1_range) / w_lin[1]
        ax.plot(x1_range, x2_boundary, "k-", lw=2, label="Межа рішення")

        # Закрашуємо напівпростори
        xx, yy = np.meshgrid(
            np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, 200),
            np.linspace(X[:, 1].min()-1, X[:, 1].max()+1, 200)
        )
        zz = (np.c_[xx.ravel(), yy.ravel()] @ w_lin >= theta).astype(int).reshape(xx.shape)
        ax.contourf(xx, yy, zz, alpha=0.08, cmap="RdBu")

        # Центри класів
        ax.scatter(*mu0, color="royalblue", s=150, marker="*", zorder=6, label="μ₀")
        ax.scatter(*mu1, color="tomato",    s=150, marker="*", zorder=6, label="μ₁")

        ax.set_title(f"{title}\nПохибка = {err:.4f}  ({err*100:.2f}%)")
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"Завдання 1: Байєсівські класифікатори\n"
        f"μ₁=(1,1), μ₀=(−1,−1), R=I, N={N}",
        fontsize=13
    )
    plt.tight_layout()
    plt.savefig("task1_classifiers.png", bbox_inches="tight")
    logger.info("Збережено: task1_classifiers.png")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# ЗАВДАННЯ 2
# Змоделювати N реалізацій гауссівського вектора: μ=(0,1)
# з 4 різними кореляційними матрицями
# ══════════════════════════════════════════════════════════════════════════════
def task2() -> None:
    logger.info("\n" + "=" * 65)
    logger.info("ЗАВДАННЯ 2 — 2D Gaussian, μ=(0,1), 4 кореляційні матриці")
    logger.info("=" * 65)

    mu = np.array([0.0, 1.0])

    # Словник: назва → кореляційна матриця
    corr_matrices = {
        "R₁ (ρ= 0.8)":  np.array([[1.0,  0.8], [ 0.8, 1.0]]),
        "R₂ (ρ=−0.5)":  np.array([[1.0, -0.5], [-0.5, 1.0]]),
        "R₃ (ρ= 0.6)":  np.array([[1.0,  0.6], [ 0.6, 1.0]]),
        "R₄ (ρ=−0.7)":  np.array([[1.0, -0.7], [-0.7, 1.0]]),
    }

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    axes = axes.ravel()

    for idx, (name, R) in enumerate(corr_matrices.items()):
        logger.info(f"\n--- {name} ---")
        logger.info(f"  R =\n{R}")

        # Перевірка позитивної визначеності (власні числа > 0)
        eigvals = np.linalg.eigvalsh(R)
        logger.info(f"  Власні числа: {eigvals.round(4)}")
        logger.info(f"  Позитивно визначена: {np.all(eigvals > 0)}")

        # Генерація вибірки
        samples = np.random.multivariate_normal(mu, R, N)

        # Вибіркові статистики
        s_mean = samples.mean(axis=0)
        s_corr = np.corrcoef(samples.T)
        logger.info(f"  Вибіркове середнє : {s_mean.round(4)}")
        logger.info(f"  Вибіркова кореляція : ρ̂ = {s_corr[0,1]:.4f}  (теор. ρ = {R[0,1]})")

        ax = axes[idx]
        ax.scatter(samples[:, 0], samples[:, 1],
                   alpha=0.35, s=14, c=f"C{idx}", label=f"N={N} зразків")

        # Довірчий еліпс 95% (χ²₂,₀.₉₅ ≈ 5.991)
        eigvals_el, eigvecs_el = np.linalg.eigh(R)
        order = eigvals_el.argsort()[::-1]
        eigvals_el = eigvals_el[order]
        eigvecs_el = eigvecs_el[:, order]
        angle = np.degrees(np.arctan2(*eigvecs_el[:, 0][::-1]))
        chi2_95 = chi2.ppf(0.95, df=2)
        ell = Ellipse(
            xy=mu,
            width  = 2 * np.sqrt(eigvals_el[0] * chi2_95),
            height = 2 * np.sqrt(eigvals_el[1] * chi2_95),
            angle  = angle,
            edgecolor="crimson", facecolor="none", lw=2, label="95%-еліпс"
        )
        ax.add_patch(ell)

        ax.axvline(x=mu[0], color="gray", linestyle="--", alpha=0.4)
        ax.axhline(y=mu[1], color="gray", linestyle="--", alpha=0.4)
        ax.scatter(*mu, color="red", s=80, zorder=5, marker="*", label=f"μ={tuple(mu)}")

        ax.set_title(f"{name}\nρ̂_вибір.={s_corr[0,1]:.3f}", fontsize=11)
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"Завдання 2: 2D гауссівський вектор μ=(0,1)\n"
        f"4 кореляційні матриці (N={N})",
        fontsize=13
    )
    plt.tight_layout()
    plt.savefig("task2_gaussian_2d.png", bbox_inches="tight")
    logger.info("Збережено: task2_gaussian_2d.png")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# ЗАВДАННЯ 3
# 3D Gaussian вектор з 4 різними математичними сподіваннями та R=I
# ══════════════════════════════════════════════════════════════════════════════
def task3() -> None:
    logger.info("\n" + "=" * 65)
    logger.info("ЗАВДАННЯ 3 — 3D Gaussian, R=I, 4 різних математичних сподівання")
    logger.info("=" * 65)

    R = np.eye(3)  # кореляційна матриця — одинична (незалежні компоненти)

    means = {
        "μ=(0,0,0)": np.array([0, 0, 0]),
        "μ=(1,0,0)": np.array([1, 0, 0]),
        "μ=(0,1,0)": np.array([0, 1, 0]),
        "μ=(0,0,1)": np.array([0, 0, 1]),
    }

    # Зберігаємо вибірки для повторного використання
    all_samples: dict[str, np.ndarray] = {}

    for name, mu in means.items():
        logger.info(f"\n--- {name} ---")
        samples = np.random.multivariate_normal(mu, R, N)
        all_samples[name] = samples

        s_mean = samples.mean(axis=0)
        s_std  = samples.std(axis=0)
        logger.info(f"  Теоретичне μ    : {mu}")
        logger.info(f"  Вибіркове μ̂     : {s_mean.round(4)}")
        logger.info(f"  Вибіркове σ̂     : {s_std.round(4)}  (теор. σ=1 для всіх)")

    # ── 3D scatter ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 11))
    colors = ["royalblue", "tomato", "green", "darkorange"]

    for idx, (name, mu) in enumerate(means.items()):
        samples = all_samples[name]
        ax = fig.add_subplot(2, 2, idx + 1, projection="3d")
        ax.scatter(
            samples[:, 0], samples[:, 1], samples[:, 2],
            alpha=0.25, s=8, c=colors[idx]
        )
        ax.scatter(*mu, color="black", s=200, marker="*", zorder=5, label="μ")
        ax.set_title(f"{name}\nR=I, N={N}", fontsize=10)
        ax.set_xlabel("X₁")
        ax.set_ylabel("X₂")
        ax.set_zlabel("X₃")
        ax.legend(fontsize=8)

    plt.suptitle("Завдання 3: 3D гауссівські вектори (R=I)", fontsize=13)
    plt.tight_layout()
    plt.savefig("task3_3d_scatter.png", bbox_inches="tight")
    logger.info("Збережено: task3_3d_scatter.png")
    plt.show()

    # ── 2D проекції для кожного з 4 середніх ──────────────────────────────────
    projections = [(0, 1, "x₁", "x₂"), (0, 2, "x₁", "x₃"), (1, 2, "x₂", "x₃")]
    fig, axes = plt.subplots(4, 3, figsize=(14, 17))

    for row, (name, mu) in enumerate(means.items()):
        samples = all_samples[name]
        for col, (i, j, xl, yl) in enumerate(projections):
            ax = axes[row, col]
            ax.scatter(samples[:, i], samples[:, j],
                       alpha=0.3, s=10, c=colors[row])
            ax.scatter(mu[i], mu[j],
                       color="black", s=100, marker="*", zorder=5)
            ax.set_xlabel(xl)
            ax.set_ylabel(yl)
            ax.set_title(f"{name}: ({xl}, {yl})", fontsize=9)
            ax.grid(True, alpha=0.3)

    plt.suptitle("Завдання 3: 2D проекції 3D вибірок", fontsize=13)
    plt.tight_layout()
    plt.savefig("task3_projections.png", bbox_inches="tight")
    logger.info("Збережено: task3_projections.png")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# ЗАВДАННЯ 4
# 3D Gaussian μ=(1,1,1) з 4 різними кореляційними матрицями
# ══════════════════════════════════════════════════════════════════════════════
def task4() -> None:
    logger.info("\n" + "=" * 65)
    logger.info("ЗАВДАННЯ 4 — 3D Gaussian, μ=(1,1,1), 4 кореляційні матриці")
    logger.info("=" * 65)

    mu = np.array([1.0, 1.0, 1.0])

    corr_matrices = {
        "R₁ (всі +)":
            np.array([[1.0,  0.8,  0.6],
                      [0.8,  1.0,  0.7],
                      [0.6,  0.7,  1.0]]),
        "R₂ (мішані знаки 1)":
            np.array([[1.0, -0.8, -0.6],
                      [-0.8, 1.0,  0.6],
                      [-0.6, 0.6,  1.0]]),
        "R₃ (мішані знаки 2)":
            np.array([[1.0,  0.8, -0.6],
                      [0.8,  1.0, -0.4],
                      [-0.6,-0.4,  1.0]]),
        "R₄ (від'ємні кореляції)":
            np.array([[1.0, -0.8, -0.7],
                      [-0.8, 1.0,  0.8],
                      [-0.7, 0.8,  1.0]]),
    }

    valid_matrices: dict[str, tuple] = {}  # (R, samples)

    for name, R in corr_matrices.items():
        logger.info(f"\n--- {name} ---")
        logger.info(f"  R =\n{R}")

        eigvals = np.linalg.eigvalsh(R)
        is_pd   = np.all(eigvals > 0)
        logger.info(f"  Власні числа  : {eigvals.round(4)}")
        logger.info(f"  Поз. визначена: {is_pd}")

        if not is_pd:
            logger.warning(f"  УВАГА: {name} — матриця НЕ є позитивно визначеною! Пропускаємо.")
            continue

        samples = np.random.multivariate_normal(mu, R, N)
        s_mean  = samples.mean(axis=0)
        s_corr  = np.corrcoef(samples.T)

        logger.info(f"  Вибіркове середнє       : {s_mean.round(4)}")
        logger.info(f"  Вибіркова кор. матриця  :\n{s_corr.round(3)}")

        valid_matrices[name] = (R, samples)

    # ── Теплові карти: теоретична vs вибіркова кореляція ─────────────────────
    n_valid = len(valid_matrices)
    fig, axes = plt.subplots(n_valid, 2, figsize=(11, 4 * n_valid))
    if n_valid == 1:
        axes = axes[np.newaxis, :]

    feature_names_3d = ["x₁", "x₂", "x₃"]

    for row, (name, (R, samples)) in enumerate(valid_matrices.items()):
        s_corr = np.corrcoef(samples.T)

        for col, (matrix, title) in enumerate([
            (R,      "Теоретична R"),
            (s_corr, "Вибіркова R̂"),
        ]):
            ax  = axes[row, col]
            im  = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            for i in range(3):
                for j in range(3):
                    ax.text(
                        j, i, f"{matrix[i,j]:.2f}",
                        ha="center", va="center", fontsize=11, fontweight="bold",
                        color="white" if abs(matrix[i,j]) > 0.6 else "black"
                    )

            ax.set_xticks(range(3))
            ax.set_yticks(range(3))
            ax.set_xticklabels(feature_names_3d)
            ax.set_yticklabels(feature_names_3d)
            ax.set_title(f"{name}\n{title}", fontsize=10)

    plt.suptitle("Завдання 4: Кореляційні матриці 3D гауссівських векторів, μ=(1,1,1)", fontsize=12)
    plt.tight_layout()
    plt.savefig("task4_correlation_heatmaps.png", bbox_inches="tight")
    logger.info("Збережено: task4_correlation_heatmaps.png")
    plt.show()

    # ── Scatter-матриця для першої валідної кореляційної матриці ───────────────
    first_name, (R_first, samples_first) = next(iter(valid_matrices.items()))
    projections_3d = [(0, 1, "x₁", "x₂"), (0, 2, "x₁", "x₃"), (1, 2, "x₂", "x₃")]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, (i, j, xl, yl) in zip(axes, projections_3d):
        ax.scatter(samples_first[:, i], samples_first[:, j],
                   alpha=0.35, s=14, c="steelblue")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(f"Проекція ({xl}, {yl})\nρ={R_first[i,j]}")
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Завдання 4: Scatter для {first_name}\nμ=(1,1,1), N={N}", fontsize=12)
    plt.tight_layout()
    plt.savefig("task4_scatter_r1.png", bbox_inches="tight")
    logger.info("Збережено: task4_scatter_r1.png")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# ЗАВДАННЯ 8
# Перевірка гіпотез про закон розподілу (нормальність)
# Тести: Шапіро-Вілка, Колмогорова-Смірнова
# Графіки: Q-Q plot, гістограма з теоретичною густиною
# ══════════════════════════════════════════════════════════════════════════════
def _normality_tests(samples: np.ndarray, label: str) -> None:
    """
    Перевіряє нормальність кожного виміру вибірки.
    Використовує тести Шапіро-Вілка (N≤5000) та Колмогорова-Смірнова.
    """
    logger.info(f"\n[Нормальність: {label}]")
    alpha = 0.05  # рівень значущості

    for dim in range(samples.shape[1]):
        col = samples[:, dim]

        # Shapiro-Wilk test (точніший для малих вибірок N<5000)
        if len(col) <= 5000:
            stat_sw, p_sw = shapiro(col)
            verdict_sw = "✓ Нормальний" if p_sw > alpha else "✗ НЕ нормальний"
            logger.info(
                f"  Вимір {dim+1} | Shapiro-Wilk  : W={stat_sw:.4f}, p={p_sw:.4f}  → {verdict_sw}"
            )

        # Kolmogorov-Smirnov test (порівнюємо зі стандартним N(0,1))
        col_std    = (col - col.mean()) / col.std()
        stat_ks, p_ks = kstest(col_std, "norm")
        verdict_ks = "✓ Нормальний" if p_ks > alpha else "✗ НЕ нормальний"
        logger.info(
            f"  Вимір {dim+1} | Kolmogorov-Smirnov: D={stat_ks:.4f}, p={p_ks:.4f}  → {verdict_ks}"
        )


def task8() -> None:
    logger.info("\n" + "=" * 65)
    logger.info("ЗАВДАННЯ 8 — Перевірка гіпотез про закон розподілу")
    logger.info("=" * 65)
    logger.info("Нульова гіпотеза H₀: дані мають нормальний закон розподілу")
    logger.info("Альтернатива  H₁: розподіл ≠ нормальному")
    logger.info("Рівень значущості α = 0.05")

    # Генеруємо тестові вибірки
    samples_2d = np.random.multivariate_normal([1.0, 1.0], np.eye(2), N)
    samples_3d = np.random.multivariate_normal([1.0, 0.0, 0.0], np.eye(3), N)

    _normality_tests(samples_2d, "2D Gaussian μ=(1,1), R=I")
    _normality_tests(samples_3d, "3D Gaussian μ=(1,0,0), R=I")

    # ── Q-Q plot + гістограми ──────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))

    # Рядок 0–1: Q-Q графіки для 2D (2 виміри) та 3D (3 виміри)
    all_dims = [
        ("2D вимір 1", samples_2d[:, 0]),
        ("2D вимір 2", samples_2d[:, 1]),
        ("3D вимір 1", samples_3d[:, 0]),
        ("3D вимір 2", samples_3d[:, 1]),
        ("3D вимір 3", samples_3d[:, 2]),
    ]

    flat_axes = axes.ravel()
    for ax_idx, (dim_name, col) in enumerate(all_dims):
        ax = flat_axes[ax_idx]
        stats.probplot(col, dist="norm", plot=ax)
        ax.set_title(f"Q-Q plot: {dim_name}\nμ̂={col.mean():.2f}, σ̂={col.std():.2f}", fontsize=9)
        ax.grid(True, alpha=0.3)

    # Останній підграфік — гістограма 2D виміру 1 з теоретичною густиною
    ax = flat_axes[5]
    col = samples_2d[:, 0]
    ax.hist(col, bins=35, density=True, alpha=0.6, color="steelblue", label="Емпірична")
    x_rng = np.linspace(col.min(), col.max(), 200)
    ax.plot(x_rng, norm.pdf(x_rng, col.mean(), col.std()), "r-", lw=2, label="N(μ̂,σ̂²)")
    ax.set_title("Гістограма 2D вимір 1", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Гістограма 3D вимірів 1 та 2
    for offset, (col, title) in enumerate([
        (samples_3d[:, 0], "Гістограма 3D вимір 1"),
        (samples_3d[:, 1], "Гістограма 3D вимір 2"),
        (samples_3d[:, 2], "Гістограма 3D вимір 3"),
    ]):
        ax = flat_axes[6 + offset]
        ax.hist(col, bins=35, density=True, alpha=0.6, color="darkorange", label="Емпірична")
        x_rng = np.linspace(col.min(), col.max(), 200)
        ax.plot(x_rng, norm.pdf(x_rng, col.mean(), col.std()), "r-", lw=2, label="N(μ̂,σ̂²)")
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Завдання 8: Перевірка нормальності (Q-Q plot + гістограми)", fontsize=13)
    plt.tight_layout()
    plt.savefig("task8_normality.png", bbox_inches="tight")
    logger.info("Збережено: task8_normality.png")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# ЗАВДАННЯ 9
# Набір даних із 5 ознак:
#   x1, x2 — сильно корельовані (ρ=0.9)
#   x3     — незалежна від усіх (ρ≈0)
#   x4, x5 — слабо корельовані (ρ=0.3)
# Нормальний закон розподілу з відомими параметрами
# ══════════════════════════════════════════════════════════════════════════════
def task9() -> pd.DataFrame:
    logger.info("\n" + "=" * 65)
    logger.info("ЗАВДАННЯ 9 — Набір даних із 5 ознак")
    logger.info("  x₁,x₂  → сильна кореляція  ρ=0.9")
    logger.info("  x₃     → незалежна         ρ=0.0")
    logger.info("  x₄,x₅  → слабка кореляція  ρ=0.3")
    logger.info("=" * 65)

    # Кореляційна матриця 5×5
    # Блокова структура: [x1,x2] | [x3] | [x4,x5]
    R = np.array([
        [1.0, 0.9, 0.0, 0.0, 0.0],   # x₁
        [0.9, 1.0, 0.0, 0.0, 0.0],   # x₂  (сильна кореляція з x₁)
        [0.0, 0.0, 1.0, 0.0, 0.0],   # x₃  (незалежна)
        [0.0, 0.0, 0.0, 1.0, 0.3],   # x₄
        [0.0, 0.0, 0.0, 0.3, 1.0],   # x₅  (слабка кореляція з x₄)
    ])

    # Вектор математичних сподівань
    mu = np.array([2.0, 2.5, 0.0, -1.0, 1.5])

    logger.info(f"\nВектор середніх μ = {mu}")
    logger.info(f"Кореляційна матриця R:\n{R}")

    # Перевірка позитивної визначеності
    eigvals = np.linalg.eigvalsh(R)
    logger.info(f"Власні числа R : {eigvals.round(4)}")
    logger.info(f"Позитивно визначена: {np.all(eigvals > 0)}")

    # Генерація вибірки
    samples = np.random.multivariate_normal(mu, R, N)
    feature_names = ["x₁", "x₂", "x₃", "x₄", "x₅"]
    df = pd.DataFrame(samples, columns=feature_names)

    # Вибіркова кореляційна матриця
    s_corr = np.corrcoef(samples.T)
    logger.info(f"\nВибіркова кореляційна матриця:\n{np.round(s_corr, 3)}")

    # ── Виведення перших 10 векторів даних ────────────────────────────────────
    logger.info("\n--- Перші 10 векторів даних ---")
    logger.info(f"\n{df.head(10).to_string()}")

    print("\n" + "─" * 65)
    print("ЗАВДАННЯ 9 — Перші 10 векторів даних:")
    print("─" * 65)
    print(df.head(10).to_string())
    print("─" * 65)

    # Пояснення значень кореляційної матриці
    logger.info("\n--- Пояснення кореляційної матриці ---")
    logger.info("  R[0,1] =  0.9 → x₁ та x₂ СИЛЬНО позитивно корельовані:")
    logger.info("              при зростанні x₁, x₂ зростає пропорційно")
    logger.info("  R[2,*] =  0.0 → x₃ НЕЗАЛЕЖНА від усіх інших ознак:")
    logger.info("              зміна x₃ не несе інформації про x₁,x₂,x₄,x₅")
    logger.info("  R[3,4] =  0.3 → x₄ та x₅ СЛАБО позитивно корельовані:")
    logger.info("              між ними є помірний лінійний зв'язок")
    logger.info("  Діагональ = 1.0 → дисперсія кожної ознаки нормована до 1")

    # ── Теплова карта: теоретична та вибіркова кореляція ──────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, (matrix, title) in zip(axes, [
        (R,      "Теоретична кореляційна матриця"),
        (s_corr, f"Вибіркова кореляційна матриця (N={N})"),
    ]):
        im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax, label="Кореляція")

        for i in range(5):
            for j in range(5):
                ax.text(
                    j, i, f"{matrix[i,j]:.2f}",
                    ha="center", va="center", fontsize=11, fontweight="bold",
                    color="white" if abs(matrix[i,j]) > 0.6 else "black"
                )

        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xticklabels(feature_names)
        ax.set_yticklabels(feature_names)
        ax.set_title(title, fontsize=11)

    plt.suptitle("Завдання 9: Кореляційні матриці (5 ознак)", fontsize=13)
    plt.tight_layout()
    plt.savefig("task9_correlation_heatmap.png", bbox_inches="tight")
    logger.info("Збережено: task9_correlation_heatmap.png")
    plt.show()

    # ── Scatter-матриця (pairplot) ─────────────────────────────────────────────
    fig, axes = plt.subplots(5, 5, figsize=(15, 15))
    colors_5 = ["royalblue", "tomato", "green", "darkorange", "purple"]

    for i in range(5):
        for j in range(5):
            ax = axes[i, j]
            if i == j:
                # Гістограма з теоретичною кривою на діагоналі
                col = samples[:, i]
                ax.hist(col, bins=25, density=True, color=colors_5[i], alpha=0.7)
                x_rng = np.linspace(col.min(), col.max(), 100)
                ax.plot(x_rng, norm.pdf(x_rng, col.mean(), col.std()), "k-", lw=1.5)
                ax.set_ylabel(feature_names[i], fontsize=9)
            else:
                ax.scatter(samples[:, j], samples[:, i], alpha=0.2, s=5, c=colors_5[i])
                # Відображаємо значення кореляції у заголовку
                rho = s_corr[i, j]
                ax.set_title(f"ρ={rho:.2f}", fontsize=7, pad=1)

            if i == 4:
                ax.set_xlabel(feature_names[j], fontsize=9)
            ax.tick_params(labelsize=6)

    plt.suptitle(
        f"Завдання 9: Матриця розсіювання (5 ознак, N={N})\n"
        f"Діагональ — гістограма; поза діагоналлю — scatter, ρ — вибіркова кореляція",
        fontsize=12
    )
    plt.tight_layout()
    plt.savefig("task9_scatter_matrix.png", bbox_inches="tight")
    logger.info("Збережено: task9_scatter_matrix.png")
    plt.show()

    # ── Порівняльні графіки трьох типів кореляції ─────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Сильна кореляція: x₁ vs x₂
    axes[0].scatter(samples[:, 0], samples[:, 1], alpha=0.3, s=14, c="crimson")
    axes[0].set_xlabel("x₁")
    axes[0].set_ylabel("x₂")
    axes[0].set_title(
        f"Сильна кореляція: x₁ та x₂\n"
        f"ρ теор.=0.9 | ρ̂={s_corr[0,1]:.3f}"
    )
    axes[0].grid(True, alpha=0.3)

    # Незалежна: x₃ vs x₁
    axes[1].scatter(samples[:, 0], samples[:, 2], alpha=0.3, s=14, c="green")
    axes[1].set_xlabel("x₁")
    axes[1].set_ylabel("x₃")
    axes[1].set_title(
        f"Незалежна: x₃ та x₁\n"
        f"ρ теор.=0.0 | ρ̂={s_corr[0,2]:.3f}"
    )
    axes[1].grid(True, alpha=0.3)

    # Слабка кореляція: x₄ vs x₅
    axes[2].scatter(samples[:, 3], samples[:, 4], alpha=0.3, s=14, c="steelblue")
    axes[2].set_xlabel("x₄")
    axes[2].set_ylabel("x₅")
    axes[2].set_title(
        f"Слабка кореляція: x₄ та x₅\n"
        f"ρ теор.=0.3 | ρ̂={s_corr[3,4]:.3f}"
    )
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("Завдання 9: Ілюстрація типів кореляцій", fontsize=13)
    plt.tight_layout()
    plt.savefig("task9_correlation_types.png", bbox_inches="tight")
    logger.info("Збережено: task9_correlation_types.png")
    plt.show()

    return df


# ══════════════════════════════════════════════════════════════════════════════
# ГОЛОВНА ФУНКЦІЯ
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logger.info("╔" + "═" * 63 + "╗")
    logger.info("║       СЕМІНАРСЬКЕ ЗАВДАННЯ №2                              ║")
    logger.info("║       ІМОВІРНІСНЕ МАШИННЕ НАВЧАННЯ                         ║")
    logger.info("╚" + "═" * 63 + "╝")
    logger.info(f"N = {N} реалізацій,  random seed = {SEED}")

    task1()   # Байєсівські класифікатори (2D, R=I)
    task2()   # 2D Gaussian з 4 кореляційними матрицями
    task3()   # 3D Gaussian з 4 математичними сподіваннями
    task4()   # 3D Gaussian з 4 кореляційними матрицями
    task8()   # Перевірка гіпотез про нормальність
    task9()   # 5-ознаковий набір даних

    logger.info("\n╔" + "═" * 63 + "╗")
    logger.info("║  Всі завдання виконано успішно!                            ║")
    logger.info("╚" + "═" * 63 + "╝")
