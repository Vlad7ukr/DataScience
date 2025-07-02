import pandas as pd

# Зчитування даних з файлу
df = pd.read_csv("insurance.csv")

# 1. Інформація про набір даних та типи ознак
print("=" * 50)
print("ІНФОРМАЦІЯ ПРО НАБІР ДАНИХ")
print("=" * 50)
print(df.info())

print("\nТипи ознак:")
print(df.dtypes)

# Категоріальні ознаки: sex, smoker, region
# Кількісні ознаки: age, bmi, children, expenses

# 2. Операції над заданим набором даних
print("\n" + "=" * 50)
print("СПИСОК НАЗВ СТОВПЦІВ")
print("=" * 50)
columns_list = df.columns.tolist()
print(", ".join(columns_list))

print("\n" + "=" * 50)
print("КІЛЬКІСТЬ КУРЦІВ І НЕ КУРЦІВ")
print("=" * 50)
smoker_counts = df['smoker'].value_counts()
for category, count in smoker_counts.items():
    print(f"{category.capitalize()}: {count}")

print("\n" + "=" * 50)
print("ВИПАДКОВИЙ ЧОЛОВІК-КУРЕЦЬ З ВИТРАТАМИ > 30 000")
print("=" * 50)
random_male_smoker = df[(df['sex'] == 'male') & (df['smoker'] == 'yes') & (df['expenses'] > 30000)].sample(n=1, random_state=42)
print(random_male_smoker.to_string(index=False))

print("\n" + "=" * 50)
print("ДОДАНО НОВИЙ РЯДОК")
print("=" * 50)
new_row = {
    'age': 40,
    'sex': 'female',
    'bmi': 25.5,
    'children': 2,
    'smoker': 'no',
    'region': 'southwest',
    'expenses': 20000
}
df.loc[len(df)] = new_row
print(df.tail(1).to_string(index=False))

# 3. Робота із групованими даними
print("\n" + "=" * 50)
print("МЕДІАННИЙ ВІК ЗА РЕГІОНОМ")
print("=" * 50)
median_age_by_region = df.groupby('region')['age'].median()
print(median_age_by_region.to_string())

print("\n" + "=" * 50)
print("СЕРЕДНІЙ BMI ЗА РЕГІОНОМ")
print("=" * 50)
df['avg_bmi'] = df.groupby('region')['bmi'].transform('mean')
print(df[['region', 'bmi', 'avg_bmi']].drop_duplicates().head(10).to_string(index=False))

print("\n" + "=" * 50)
print("КЛІЄНТИ, ДЕ СЕРЕДНЯ КІЛЬКІСТЬ ДІТЕЙ < 0.5")
print("=" * 50)
ages_with_few_children = df.groupby('age')['children'].mean()
ages_selected = ages_with_few_children[ages_with_few_children < 0.5].index
clients_selected = df[df['age'].isin(ages_selected)]
print("\nДані клієнтів віку, де середня кількість дітей < 0.5:\n", clients_selected.head(10).to_string(index=False))

# 4. Pivot-таблиця
print("\n" + "=" * 50)
print("PIVOT-ТАБЛИЦЯ (СЕРЕДНІ ВИТРАТИ ТА BMI)")
print("=" * 50)
pivot_df = df.pivot_table(index=['sex', 'region'], values=['expenses', 'bmi'], aggfunc='mean')
print(pivot_df.round(2).to_string())

# 4.1 Отримати середній BMI для чоловіків з північно-західного регіону
male_nw_avg_bmi = pivot_df.loc[('male', 'northwest'), 'bmi']
print("\n" + "=" * 50)
print(f"Середній BMI для чоловіків з північно-західного регіону: {male_nw_avg_bmi:.2f}")
print("=" * 50)
