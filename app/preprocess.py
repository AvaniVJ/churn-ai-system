import pandas as pd

def preprocess_input(data: dict):
    df = pd.DataFrame([data])

    df['Gender'] = df['Gender'].map({'Male':1, 'Female':0})
    df['Gender'] = df['Gender'].fillna(0)

    num_cols = [
        'Age','Income','SpendingScore','PurchaseAmount',
        'Returns','ReviewScore','SessionTime','DaysSinceLastPurchase'
    ]

    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].mean())

    return df