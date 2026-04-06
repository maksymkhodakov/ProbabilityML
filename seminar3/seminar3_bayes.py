"""
Семінарське завдання №3 — Імовірнісне машинне навчання
Тема: Теорема Байєса у медичній діагностиці
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


# ============================================================
# ЗАВДАННЯ 1
# ============================================================

def bayes_update(prior, likelihood_pos, likelihood_neg):
    """
    Застосовує теорему Байєса для оновлення апріорної ймовірності.

    Параметри:
        prior           — P(C): апріорна ймовірність захворювання
        likelihood_pos  — P(+|C): чутливість тесту (True Positive Rate)
        likelihood_neg  — P(+|¬C): частота хибних тривог (False Positive Rate)

    Повертає:
        posterior       — P(C|+): апостеріорна ймовірність захворювання
    """
    p_cancer = prior
    p_no_cancer = 1 - prior

    # Повна ймовірність позитивного результату (знаменник Байєса)
    p_positive = likelihood_pos * p_cancer + likelihood_neg * p_no_cancer

    # Теорема Байєса: P(C|+) = P(+|C)·P(C) / P(+)
    posterior = (likelihood_pos * p_cancer) / p_positive

    return posterior, p_positive


print("=" * 60)
print("  ЗАВДАННЯ 1")
print("=" * 60)

# Вхідні дані — Завдання 1
prior_1 = 0.01  # P(C)  — базова поширеність раку
sensitivity_1 = 0.80  # P(+|C)
fpr_1 = 0.096  # P(+|¬C)

posterior_1, p_pos_1 = bayes_update(prior_1, sensitivity_1, fpr_1)

print(f"\nАпріорна ймовірність раку  P(C)    = {prior_1:.3f}  ({prior_1 * 100:.1f}%)")
print(f"Чутливість тесту           P(+|C)  = {sensitivity_1:.3f} ({sensitivity_1 * 100:.1f}%)")
print(f"Частота хибних тривог      P(+|¬C) = {fpr_1:.3f} ({fpr_1 * 100:.1f}%)")
print()
print("Обчислення:")
print(f"  P(+) = P(+|C)·P(C) + P(+|¬C)·P(¬C)")
print(f"       = {sensitivity_1}·{prior_1} + {fpr_1}·{1 - prior_1}")
print(f"       = {sensitivity_1 * prior_1:.5f} + {fpr_1 * (1 - prior_1):.5f}")
print(f"       = {p_pos_1:.5f}")
print()
print(f"  P(C|+) = P(+|C)·P(C) / P(+)")
print(f"         = {sensitivity_1 * prior_1:.5f} / {p_pos_1:.5f}")
print(f"         = {posterior_1:.5f}")
print()
print(f">>> Апостеріорна ймовірність раку після 1-го позитивного тесту:")
print(f"    P(C | +₁) ≈ {posterior_1:.4f}  ({posterior_1 * 100:.2f}%)")

# ============================================================
# ЗАВДАННЯ 2
# ============================================================

print("\n" + "=" * 60)
print("  ЗАВДАННЯ 2")
print("=" * 60)

# Вхідні дані — Завдання 2
# Апріор = апостеріор Завдання 1
prior_2 = posterior_1
sensitivity_2 = 0.90  # P(+₂|C) — нова лабораторія
fpr_2 = 0.056  # P(+₂|¬C) — нова лабораторія

posterior_2, p_pos_2 = bayes_update(prior_2, sensitivity_2, fpr_2)

print(f"\nАпріор для 2-го тесту      P(C|+₁) = {prior_2:.5f}  ({prior_2 * 100:.2f}%)")
print(f"Чутливість 2-го тесту      P(+₂|C)  = {sensitivity_2:.3f} ({sensitivity_2 * 100:.1f}%)")
print(f"Хибні тривоги 2-ї лаб.     P(+₂|¬C) = {fpr_2:.3f} ({fpr_2 * 100:.1f}%)")
print()
print("Обчислення (послідовне оновлення Байєса):")
print(f"  P(+₂) = P(+₂|C)·P(C|+₁) + P(+₂|¬C)·P(¬C|+₁)")
print(f"        = {sensitivity_2}·{prior_2:.5f} + {fpr_2}·{1 - prior_2:.5f}")
print(f"        = {sensitivity_2 * prior_2:.6f} + {fpr_2 * (1 - prior_2):.6f}")
print(f"        = {p_pos_2:.6f}")
print()
print(f"  P(C|+₁,+₂) = P(+₂|C)·P(C|+₁) / P(+₂)")
print(f"             = {sensitivity_2 * prior_2:.6f} / {p_pos_2:.6f}")
print(f"             = {posterior_2:.5f}")
print()
print(f">>> Апостеріорна ймовірність раку після двох позитивних тестів:")
print(f"    P(C | +₁, +₂) ≈ {posterior_2:.4f}  ({posterior_2 * 100:.2f}%)")

# ============================================================
# ВІЗУАЛІЗАЦІЯ
# ============================================================

fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor('#FAFAFA')
gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

stages = ["Апріор\nP(C)", "Після тесту 1\nP(C|+₁)", "Після тесту 2\nP(C|+₁,+₂)"]
probs = [prior_1, posterior_1, posterior_2]
colors = ['#5B9BD5', '#ED7D31', '#70AD47']

# --- Графік 1: Еволюція ймовірності ---
ax1 = fig.add_subplot(gs[0, :2])
bars = ax1.bar(stages, [p * 100 for p in probs], color=colors,
               width=0.5, edgecolor='white', linewidth=1.5, zorder=3)
for bar, p in zip(bars, probs):
    ax1.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 1.5,
             f'{p * 100:.2f}%', ha='center', va='bottom',
             fontsize=13, fontweight='bold', color='#333333')
ax1.set_title('Еволюція байєсівського оновлення ймовірності раку',
              fontsize=13, fontweight='bold', pad=12)
ax1.set_ylabel('Ймовірність (%)', fontsize=11)
ax1.set_ylim(0, max(probs) * 100 * 1.25)
ax1.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.tick_params(labelsize=11)

# --- Графік 2: Кругова діаграма після тесту 1 ---
ax2 = fig.add_subplot(gs[0, 2])
sizes1 = [posterior_1 * 100, (1 - posterior_1) * 100]
wedges1, texts1, autotexts1 = ax2.pie(
    sizes1,
    labels=['Рак', 'Немає раку'],
    autopct='%1.2f%%',
    colors=['#ED7D31', '#5B9BD5'],
    startangle=90,
    wedgeprops=dict(edgecolor='white', linewidth=2),
    textprops={'fontsize': 10}
)
ax2.set_title('Після тесту 1', fontsize=12, fontweight='bold')

# --- Графік 3: Кругова діаграма після тесту 2 ---
ax3 = fig.add_subplot(gs[1, 2])
sizes2 = [posterior_2 * 100, (1 - posterior_2) * 100]
wedges2, texts2, autotexts2 = ax3.pie(
    sizes2,
    labels=['Рак', 'Немає раку'],
    autopct='%1.2f%%',
    colors=['#ED7D31', '#70AD47'],
    startangle=90,
    wedgeprops=dict(edgecolor='white', linewidth=2),
    textprops={'fontsize': 10}
)
ax3.set_title('Після тесту 2', fontsize=12, fontweight='bold')

# --- Графік 4: Порівняння характеристик тестів ---
ax4 = fig.add_subplot(gs[1, :2])
metrics = ['Чутливість\n(Sensitivity)', 'Хибні тривоги\n(FPR)']
lab1_vals = [sensitivity_1 * 100, fpr_1 * 100]
lab2_vals = [sensitivity_2 * 100, fpr_2 * 100]

x = np.arange(len(metrics))
w = 0.3
b1 = ax4.bar(x - w / 2, lab1_vals, w, label='Лабораторія 1',
             color='#5B9BD5', edgecolor='white', zorder=3)
b2 = ax4.bar(x + w / 2, lab2_vals, w, label='Лабораторія 2',
             color='#70AD47', edgecolor='white', zorder=3)

for bar in list(b1) + list(b2):
    ax4.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.5,
             f'{bar.get_height():.1f}%', ha='center',
             fontsize=10, fontweight='bold', color='#333333')

ax4.set_title('Порівняння характеристик двох тестів', fontsize=12, fontweight='bold', pad=10)
ax4.set_xticks(x)
ax4.set_xticklabels(metrics, fontsize=11)
ax4.set_ylabel('Значення (%)', fontsize=11)
ax4.set_ylim(0, 110)
ax4.legend(fontsize=10)
ax4.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

plt.suptitle('Байєсівське оновлення ймовірності: мамографічна діагностика',
             fontsize=15, fontweight='bold', y=1.01, color='#1F3864')

plt.savefig('/Users/maksymkhodakov/ProbabilityML/ProbabilityML/seminar3/bayes_visualization.png', dpi=150,
            bbox_inches='tight', facecolor='#FAFAFA')
plt.close()
print("\n[Візуалізацію збережено: bayes_visualization.png]")
