import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# ---------------------------
# LOAD DATA
# ---------------------------
df = pd.read_csv("churn prediction.csv")
df.columns = df.columns.str.strip()

# ---------------------------
# BASIC CLEANING
# ---------------------------
df = df.drop_duplicates()

# Fix Gender
df['Gender'] = df['Gender'].replace({'M': 'Male', 'F': 'Female'})
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df['Gender'] = df['Gender'].fillna(0)

# Fix Churn (clean version, no warning)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# ---------------------------
# NUMERIC CLEANING
# ---------------------------
num_cols = [
    'Age', 'Income', 'SpendingScore',
    'PurchaseAmount', 'Returns',
    'ReviewScore', 'SessionTime'
]

for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].fillna(df[col].mean())

# ---------------------------
# DATE FEATURE
# ---------------------------
df['LastPurchaseDate'] = pd.to_datetime(df['LastPurchaseDate'], errors='coerce')

today = pd.Timestamp.today()
df['DaysSinceLastPurchase'] = (today - df['LastPurchaseDate']).dt.days

df['DaysSinceLastPurchase'] = df['DaysSinceLastPurchase'].fillna(
    df['DaysSinceLastPurchase'].mean()
)

df = df.drop(columns=['LastPurchaseDate'])

# ---------------------------
# 🔥 FEATURE ENGINEERING
# ---------------------------
# Engagement
df['EngagementScore'] = df['SessionTime'] * df['SpendingScore']

# Avg Spend per Session (NEW SIGNAL)
df['AvgSpendPerSession'] = df['PurchaseAmount'] / (df['SessionTime'] + 1)

# ---------------------------
# FEATURES & TARGET
# ---------------------------
features = [
    'Age', 'Income', 'SpendingScore', 'PurchaseAmount',
    'DaysSinceLastPurchase', 'Returns', 'ReviewScore',
    'SessionTime', 'Gender',
    'EngagementScore', 'AvgSpendPerSession'
]

X = df[features]
y = df['Churn']

# ---------------------------
# TRAIN TEST SPLIT
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# SCALING
# ---------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------------------
# MODEL (Improved)
# ---------------------------
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=5,
    class_weight='balanced',
    random_state=42
)

model.fit(X_train, y_train)

# ---------------------------
# EVALUATION
# ---------------------------
y_pred = model.predict(X_test)

print("\n📊 Model Performance:")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))

print("\n📄 Classification Report:")
print(classification_report(y_test, y_pred))

print("\n📉 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ---------------------------
# 🔥 FEATURE IMPORTANCE
# ---------------------------
importance = model.feature_importances_

feat_importance = pd.DataFrame({
    'feature': features,
    'importance': importance
}).sort_values(by='importance', ascending=False)

print("\n🔥 Feature Importance:")
print(feat_importance)

# ---------------------------
# SAVE ARTIFACTS
# ---------------------------
joblib.dump(model, "models/churn_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(feat_importance, "models/feature_importance.pkl")

print("\n✅ Model trained, evaluated, and saved successfully")
