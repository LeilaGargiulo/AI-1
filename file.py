import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from ucimlrepo import fetch_ucirepo

# Fetch the dataset
ai4i_2020_predictive_maintenance_dataset = fetch_ucirepo(id=601)

# Load the dataset
X = ai4i_2020_predictive_maintenance_dataset.data.features
y = ai4i_2020_predictive_maintenance_dataset.data.targets

# Strip any leading or trailing spaces from the column names
X.columns = X.columns.str.strip()

# Print the column names to verify
print(X.columns)
print(X.head())
print(X.info())


