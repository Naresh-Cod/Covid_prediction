import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import pickle

# Load dataset
df = pd.read_csv("covid_toy.csv")

# Encode categorical values
le_gender = LabelEncoder()
le_cough = LabelEncoder()
le_city = LabelEncoder()
le_target = LabelEncoder()

df['gender'] = le_gender.fit_transform(df['gender'])
df['cough'] = le_cough.fit_transform(df['cough'])
df['city'] = le_city.fit_transform(df['city'])
df['has_covid'] = le_target.fit_transform(df['has_covid'])

# Handle missing values
imputer = SimpleImputer(strategy="mean")
df['fever'] = imputer.fit_transform(df[['fever']])

# Features and target
X = df.drop('has_covid', axis=1)
y = df['has_covid']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model and encoders
with open("model.pkl", "wb") as f:
    pickle.dump((model, le_gender, le_cough, le_city, le_target, imputer), f)

print("Model trained and saved as model.pkl")
