import pandas as pd
import matplotlib.pyplot as plt

# Завантаження даних
amazon_data = pd.read_csv('Amazon.csv', parse_dates=['Date'])
amazon_data.set_index('Date', inplace=True)

# Завдання 1: Побудова графіків зміни ціни на час відкриття біржі
# 1а. Загальний графік
plt.figure(figsize=(10, 6))
amazon_data['Open'].plot(title='Зміна ціни на час відкриття біржі (Загальний графік)')
plt.xlabel('Дата')
plt.ylabel('Ціна')
plt.grid(True)
plt.show()

# 1б. За 2019 рік
plt.figure(figsize=(10, 6))
amazon_data['Open']['2019'].plot(title='Зміна ціни на час відкриття біржі за 2019 рік', label='2019')
plt.xlabel('Дата')
plt.ylabel('Ціна')
plt.grid(True)
plt.show()

# 1в. За липень 2020 року
plt.figure(figsize=(10, 6))
amazon_data['Open']['2020-07'].plot(title='Зміна ціни на час відкриття біржі за липень 2020 року')
plt.xlabel('Дата')
plt.ylabel('Ціна')
plt.grid(True)
plt.show()

# 1г. Жовтень 2014 – червень 2016
plt.figure(figsize=(10, 6))
amazon_data['Open']['2014-10':'2016-06'].plot(title='Зміна ціни (Жовтень 2014 - Червень 2016)')
plt.xlabel('Дата')
plt.ylabel('Ціна')
plt.grid(True)
plt.show()

# 1д. Паралельне порівняння 2015 та 2017 за днем року
plt.figure(figsize=(10, 6))
for year in [2015, 2017]:
    df_year = amazon_data[amazon_data.index.year == year].copy()
    df_year['DayOfYear'] = df_year.index.dayofyear
    plt.plot(df_year['DayOfYear'], df_year['Open'], label=str(year))
plt.title('Паралельне порівняння ціни відкриття за 2015 та 2017 (за днем року)')
plt.xlabel('День року')
plt.ylabel('Ціна')
plt.legend()
plt.grid(True)
plt.show()

# Завдання 2: Мінімальні значення Low
# 2а. За 2018 рік
min_low_2018 = amazon_data['Low']['2018'].min()
print(f"Мінімальна ціна Low у 2018 році: {min_low_2018}")

# 2б. За кожний місяць 2019 року
min_low_monthly_2019 = amazon_data['Low']['2019'].resample('ME').min()
print("Мінімальні Low по місяцях 2019:\n", min_low_monthly_2019)

# 2в. За кожний тиждень Q2 2015
min_low_weekly_q2_2015 = amazon_data['Low']['2015-04':'2015-06'].resample('W').min()
print("Мінімальні Low по тижнях Q2 2015:\n", min_low_weekly_q2_2015)

# 2г. Зміни Low у відсотках за кожні 2 дні осені 2016
low_fall_2016 = amazon_data['Low']['2016-09':'2016-11']
low_change_pct = low_fall_2016.pct_change(periods=2) * 100
plt.figure(figsize=(10, 6))
plt.plot(low_change_pct.index, low_change_pct, label='Зміна Low (%) кожні 2 дні')
plt.title('Зміни найменшої ціни Low у відсотках (осінь 2016, кожні 2 дні)')
plt.xlabel('Дата')
plt.ylabel('Зміна (%)')
plt.grid(True)
plt.legend()
plt.show()

# 2д. Ковзне середнє Low за травень 2016 з вікном 7 днів
low_may_2016 = amazon_data['Low']['2016-05']
moving_avg = low_may_2016.rolling(window=7).mean()
plt.figure(figsize=(10, 6))
plt.plot(moving_avg.index, moving_avg, label='7-денне ковзне середнє')
plt.title('Ковзне середнє Low за травень 2016 (7 днів)')
plt.xlabel('Дата')
plt.ylabel('Low (MA)')
plt.grid(True)
plt.legend()
plt.show()
