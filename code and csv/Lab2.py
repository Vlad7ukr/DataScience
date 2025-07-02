import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, pearsonr, linregress
import statsmodels.api as sm

# Завантаження даних
file_path = "Budget.csv"
data = pd.read_csv(file_path)

if 'Unnamed: 0' in data.columns:
    data = data.drop(columns=['Unnamed: 0'])

print("\n===== Перші 5 рядків даних =====")
print(data.head().to_string(index=False))

# 1. Розрахунок середньої кількості дітей та стандартного відхилення
mean_children = data['children'].mean()
std_children = data['children'].std()
print("\n===== Кількість дітей у сім'ї =====")
print(f"Середня кількість дітей: {mean_children:.2f}")
print(f"Стандартне відхилення: {std_children:.2f}\n")

# 2. Перевірка нормальності розподілу доходів
stat, p = shapiro(data['income'])
print("===== Перевірка нормальності доходів (Шапіро-Уілк) =====")
print(f"Статистика тесту: {stat:.3f}")
print(f"p-значення: {p:.3f}")
if p > 0.05:
    print("Доходи мають нормальний розподіл. (H0 не відхиляється)\n")
else:
    print("Доходи НЕ мають нормального розподілу. (H0 відхиляється)\n")

# Візуалізація розподілу доходів
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(data['income'], bins=20, kde=True, color='blue')
plt.title("Гістограма доходів")
plt.xlabel("Доходи")

plt.subplot(1, 2, 2)
sns.boxplot(x=data['income'], color='blue')
plt.title("Boxplot доходів")
plt.xlabel("Доходи")

plt.show()

# 3. Кореляція між витратами на пальне та транспорт
corr_coef, p_value = pearsonr(data['wfuel'], data['wtrans'])
print("===== Кореляція між витратами на пальне та транспорт =====")
print(f"Коефіцієнт Пірсона: {corr_coef:.3f}")
print(f"p-значення: {p_value:.3f}")
if p_value < 0.05:
    print("Є статистично значущий зв'язок між витратами на пальне та транспортом. (H0 відхиляється)\n")
else:
    print("Статистично значущого зв'язку немає. (H0 не відхиляється)\n")

# 4. Лінійна регресія: витрати на їжу від доходу
X = data['income']
y = data['wfood']
X_const = sm.add_constant(X)
model_sm = sm.OLS(y, X_const).fit()

print("===== Лінійна регресія: витрати на їжу ~ дохід =====")
print(model_sm.summary())

# Візуалізація регресії
plt.figure(figsize=(8, 6))
sns.regplot(x='income', y='wfood', data=data, color='blue', line_kws={'color': 'red'})
plt.xlabel('Дохід')
plt.ylabel('Витрати на їжу')
plt.title('Регресія: витрати на їжу від доходу')
plt.show()
