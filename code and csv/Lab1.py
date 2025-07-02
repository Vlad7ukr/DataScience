import numpy as np
import pandas as pd
import os

# Завдання 1
print("\n--- Завдання 1 ---")

# а) Створення різних типів масивів
print("\nа) Генерація масивів:")

arr1 = np.arange(1, 16, 2)
print("Масив із кроком 2 у діапазоні [1, 16):\n", arr1)

arr2 = np.ones(9, dtype=int)
print("Масив із одиниць:\n", arr2)

arr3 = np.zeros((2, 4), dtype=float)
print("Двовимірний масив нулів:\n", arr3)

arr4 = np.linspace(0, 1, 5)
print("Рівномірний розподіл значень від 0 до 1:\n", arr4)

arr5 = np.random.rand(4, 3)
print("Масив випадкових чисел (4x3):\n", arr5)

arr6 = np.random.randint(0, 100, size=(10, 10))
print("Масив випадкових цілих чисел (4x4):\n", arr6)

print(arr6[-3:, ::2])

arr7 = np.empty(5)
print("Порожній масив із 5 елементів:\n", arr7)

# б) Доступ до елементів масиву
print("\nб) Індексація та зрізи:")
print("Третій рядок у масиві arr6:\n", arr6[2])
print("Третій елемент з кінця:\n", arr6[-3])
print("Елемент першого рядка та останнього стовпця:\n", arr6[0, -1])

print("\nВиділення підмасивів:")
print("Перші два рядки:\n", arr6[:2])
print("Другий та третій стовпці:\n", arr6[:, 1:3])

# в) Арифметичні операції
print("\nв) Операції з масивами:")
arr8 = np.array([1, 2, 3])
arr9 = np.array([4, 5, 6])

print("Сума масивів:\n", arr8 + arr9)
print("Добуток масивів:\n", arr8 * arr9)
print("Функція reduce (сума елементів arr8):\n", np.add.reduce(arr8))
print("Функція accumulate (накопичена сума arr8):\n", np.add.accumulate(arr8))
print("Функція outer (перемноження кожного з кожним):\n", np.multiply.outer(arr8, arr9))

# Завдання 2
print("\nЗавдання 2:")

file_path = 'CommerceShipping.csv'

if os.path.exists(file_path):
    # Завантаження даних
    data = pd.read_csv(file_path)

    print("\nВиведення перших рядків таблиці:")
    print(data.head().to_string(index=False))

    # Вивід статистичних характеристик
    print("\nОсновні статистичні характеристики кількісних ознак:")
    print(data.describe().to_string())

    # Фільтрація товарів
    low_prior_under_2kg = data[(data['Product_importance'] == 'low') &
                               (data['Mode_of_Shipment'] == 'Ship') &
                               (data['Weight_in_gms'] < 2000)]

    # Виведення загальної кількості знайдених товарів
    print(
        f"\nКількість товарів з низьким пріоритетом, вагою до 2кг, що будуть відправлені кораблем: {len(low_prior_under_2kg)}")

    # Виведення лише перших 5 рядків таблиці
    if not low_prior_under_2kg.empty:
        print("\nПерелік таких товарів:")
        print(low_prior_under_2kg.to_string(index=False))
    else:
        print("\nТоварів, що відповідають цим критеріям, немає.")

    # Додаємо новий стовпець з ціною за кг
    data['Price_per_kg'] = data['Cost_of_the_Product'] / (data['Weight_in_gms'] / 1000)

    print("\nТаблиця з новим стовпцем:")
    # Перевіряємо результат (виведемо перші 5 рядків)
    print(data.head().to_string(index=False))

else:
    print("Файл не знайдено!")
