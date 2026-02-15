import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pickle

# Paths
TEST_CSV = 'test_data/test.csv'
MODEL_OUT = 'spam.pkl'

# Map form fields to dataset columns (choose closest match where needed)
feature_mapping = [
    ('gen [kW]', 'generation'),
    ('House overall [kW]', 'House overall'),
    ('Dishwasher [kW]', 'Dishwasher'),
    ('Furnace 1 [kW]', 'Furnace'),
    ('Home office [kW]', 'Home office'),
    ('Fridge [kW]', 'Fridge'),
    ('Wine cellar [kW]', 'Wine cellar'),
    ('Garage door [kW]', 'Garage door'),
    ('Kitchen 12 [kW]', 'Kitchen'),
    ('Barn [kW]', 'Barn'),
    ('Well [kW]', 'Well'),
    ('Microwave [kW]', 'Microwave'),
    ('Living room [kW]', 'Living room'),
    ('Solar [kW]', 'Solar'),
    ('temperature', 'temperature'),
    ('humidity', 'humidity'),
    ('visibility', 'visibility'),
    ('apparentTemperature', 'apparentTemperature'),
    ('pressure', 'pressure'),
    ('windSpeed', 'windSpeed'),
    ('windBearing', 'windBearing'),
    ('precipIntensity', 'precipIntensity'),
]

# Load dataset
print('Loading dataset from', TEST_CSV)
df = pd.read_csv(TEST_CSV)

# Prepare target
print('Preparing target labels from "class" column')
# Convert strings like "1(spam)" / "0(no spam)" to ints 1/0
y = df['class'].astype(str).str.contains('1').astype(int)

# Build feature matrix in the exact order expected by the form
X_cols = [src for src, _ in feature_mapping]
missing = [c for c in X_cols if c not in df.columns]
if missing:
    raise RuntimeError(f'Missing required feature columns in test.csv: {missing}')

X = df[X_cols].astype(float).values
print('Feature matrix shape:', X.shape)

# Train a simple classifier for demo purposes
print('Training DecisionTreeClassifier')
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X, y)

# Save the model to the expected path
print('Saving model to', MODEL_OUT)
with open(MODEL_OUT, 'wb') as f:
    pickle.dump(clf, f)

print('Done. Model saved as spam.pkl and ready for app.')