import pandas as pd
from fuzzywuzzy import process
import numpy as np

df = pd.read_excel('Version 10.xlsx')

df.rename(columns={
    'ID': 'ID',
    'Warehouse_block': 'Warehouse',
    'Mode_of_Shipment': 'Shipment_Mode',
    'Customer_care_calls': 'Care_Calls',
    'Customer_rating': 'Rating',
    'Cost_of_the_Product': 'Cost',
    'Prior_purchases': 'Purchases',
    'Product_importance': 'Importance',
    'Gender': 'Gender',
    'Discount_offered': 'Discount',
    'Weight_in_gms': 'Weight',
    'Reached.on.Time_Y.N': 'DeliveredOnTime'
}, inplace=True)


def correct_spelling_fuzzy(series, valid_values, min_score=70):
    corrected = []
    for value in series:
        str_value = str(value).strip()
        if not str_value or str_value in ['???', '?', 'nan', 'NaN', 'None']:
            corrected.append(np.nan)
        else:
            match, score = process.extractOne(str_value, valid_values)
            corrected.append(match if score >= min_score else value)
    return pd.Series(corrected)


valid_mode = ['Flight', 'Ship', 'Road']
valid_importance = ['low', 'medium', 'high']
valid_gender = ['M', 'F']
valid_warehouse = ['A', 'B', 'C', 'D', 'F']

df['Shipment_Mode'] = correct_spelling_fuzzy(df['Shipment_Mode'], valid_mode)
df['Importance'] = correct_spelling_fuzzy(df['Importance'], valid_importance)
df['Gender'] = correct_spelling_fuzzy(df['Gender'], valid_gender)
df['Warehouse'] = correct_spelling_fuzzy(df['Warehouse'], valid_warehouse)



df.drop_duplicates(inplace=True)

df.fillna({col: df[col].mean() for col in df.select_dtypes(include='number')}, inplace=True)

df.fillna({col: df[col].mode()[0] for col in df.select_dtypes(include='object')}, inplace=True)

df.to_excel('cleaned_output.xlsx', index=False)


print("\nПерші 5 рядків:")
print(df.head().to_string(index=False))

print(df.describe().to_string())
