import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load data
df = pd.read_csv("churn prediction.csv")
df.columns = df.columns.str.strip()

# Basic cleaning
df = df.drop_duplicates()

# Fix Gender
df['Gender'] = df['Gender'].replace({'M':'Male','F':'Female'})
df['Gender'] = df['Gender'].map({'Male':1, 'Female':0})
df['Gender'] = df['Gender'].fillna(0)

# Fix churn
df['Churn'] = df['Churn'].replace({'Yes':1,'No':0})

# Numeric columns
num_cols = ['Age','Income','SpendingScore','PurchaseAmount','Returns','ReviewScore','SessionTime']
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].fillna(df[col].mean())

# Date feature
df['LastPurchaseDate'] = pd.to_datetime(df['LastPurchaseDate'], errors='coerce')
today = pd.Timestamp.today()
df['DaysSinceLastPurchase'] = (today - df['LastPurchaseDate']).dt.days
df['DaysSinceLastPurchase'] = df['DaysSinceLastPurchase'].fillna(df['DaysSinceLastPurchase'].mean())

# Features
features = [
    'Age','Income','SpendingScore','PurchaseAmount',
    'DaysSinceLastPurchase','Returns','ReviewScore','SessionTime','Gender'
]

X = df[features]
y = df['Churn']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model
model = RandomForestClassifier(class_weight='balanced')
model.fit(X_train, y_train)

# Save
joblib.dump(model, "models/churn_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("✅ Model trained and saved")
