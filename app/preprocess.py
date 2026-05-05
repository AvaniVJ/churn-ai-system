import pandas as pd

def preprocess_input(data: dict):
    df = pd.DataFrame([data])

    # ---------------------------
    # GENDER ENCODING
    # ---------------------------
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    df['Gender'] = df['Gender'].fillna(0)

    # ---------------------------
    # NUMERIC CLEANING
    # ---------------------------
    num_cols = [
        'Age', 'Income', 'SpendingScore', 'PurchaseAmount',
        'Returns', 'ReviewScore', 'SessionTime', 'DaysSinceLastPurchase'
    ]

    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].mean())

    # ---------------------------
    # 🔥 FEATURE ENGINEERING (CRITICAL)
    # ---------------------------
    # Engagement
    df['EngagementScore'] = df['SessionTime'] * df['SpendingScore']

    # Avg Spend per Session (NEW FEATURE)
    df['AvgSpendPerSession'] = df['PurchaseAmount'] / (df['SessionTime'] + 1)

    # ---------------------------
    # FEATURE ORDER (MUST MATCH train.py)
    # ---------------------------
    feature_order = [
        'Age', 'Income', 'SpendingScore', 'PurchaseAmount',
        'DaysSinceLastPurchase', 'Returns', 'ReviewScore',
        'SessionTime', 'Gender',
        'EngagementScore', 'AvgSpendPerSession'
    ]

    df = df[feature_order]

    return df
