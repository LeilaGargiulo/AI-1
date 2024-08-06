import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from ucimlrepo import fetch_ucirepo

# Load the dataset
dataset_path = "ai4i+2020+predictive+maintenance+dataset.zip"
df = pd.read_csv(dataset_path)

# Define features (X) and target (y)
X = df.drop(columns=["Machine failure"])
y = df["Machine failure"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train our magical Decision Tree
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Now let's predict using X_test
predictions = model.predict(X_test)
