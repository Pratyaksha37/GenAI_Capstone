import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

print("Starting training process...")

# 1. Load data
print("Loading data...")
df = pd.read_csv('data/kaggle_london_house_price_data.csv')

# Only keep important columns
important_columns = [
    'latitude', 'longitude', 'bedrooms', 'bathrooms', 
    'floorAreaSqM', 'livingRooms', 'tenure', 'propertyType', 
    'saleEstimate_currentPrice'
]
df = df[important_columns]


# Drop rows where target is missing
df = df.dropna(subset=['saleEstimate_currentPrice'])

# 2. Data Cleaning - Missing Values
print("Cleaning data...")
# Numeric columns
num_cols = ['latitude', 'longitude', 'bedrooms', 'bathrooms', 'floorAreaSqM', 'livingRooms']
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Categorical columns
cat_cols = ['tenure', 'propertyType']
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Map categories to match our simple web app
df['tenure'] = df['tenure'].apply(lambda x: 'FREEHOLD' if 'Freehold' in str(x) else 'LEASEHOLD')

def map_property(p):
    p = str(p).upper()
    if 'SEMI' in p: return 'SEMI_DETACHED'
    elif 'DETACHED' in p: return 'DETACHED'
    elif 'TERRACE' in p: return 'TERRACED'
    else: return 'FLAT'

df['propertyType'] = df['propertyType'].apply(map_property)

# 3. Encoding Categorical Data
print("Encoding categorical data...")
# Make sure categories are ordered so drop_first drops 'FREEHOLD' and 'FLAT'
df['tenure'] = pd.Categorical(df['tenure'], categories=['FREEHOLD', 'LEASEHOLD'])
df['propertyType'] = pd.Categorical(df['propertyType'], categories=['FLAT', 'DETACHED', 'SEMI_DETACHED', 'TERRACED'])

# Get dummies
df = pd.get_dummies(df, columns=['tenure', 'propertyType'], drop_first=True)

# 4. Log Transform Target
print("Applying log transform to target...")
df['saleEstimate_currentPrice'] = np.log(df['saleEstimate_currentPrice'])

# 5. Split Data
print("Splitting data...")
X = df.drop('saleEstimate_currentPrice', axis=1)
y = df['saleEstimate_currentPrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Scale Features
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. Train Model
print("Training model (this might take a moment)...")
model = RandomForestRegressor(n_estimators=50,max_depth=15,random_state=42,n_jobs=-1)
model.fit(X_train_scaled, y_train)

# 8. Evaluate Model
print("Evaluating model...")
y_pred = model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("-" * 30)
print("MODEL EVALUATION METRICS:")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")
print("-" * 30)

# 9. Save Model and Scaler
print("Saving model and scaler...")
joblib.dump(model, "model.joblib")
joblib.dump(scaler, "scaler.joblib")

print("Training finished successfully! Model and scaler are saved.")
