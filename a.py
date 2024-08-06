import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, precision_recall_curve
import seaborn as sns
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo

def load_dataset():
    dataset = fetch_ucirepo(id=601)
    X = dataset.data.features
    y = dataset.data.targets
    print(f"Shape of y after loading: {y.shape}")  # Check the shape of y after loading
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]  # Ensure y is a single column
    y = np.squeeze(y)  # Remove any extra dimensions
    print(f"Shape of y after flattening: {y.shape}")  # Check the shape of y after flattening
    return X, y





# Preprocess the dataset
def preprocess_data(X):
    return pd.get_dummies(X, columns=["Type"])

def split_data(X, y):
    print(f"Shape of y before splitting: {y.shape}")  # Check the shape of y before splitting
    return train_test_split(X, y, test_size=0.3, random_state=42)




# Train the model
def train_model(X_train, y_train):
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    return model

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, predictions),
        'precision': precision_score(y_test, predictions, average='weighted'),
        'recall': recall_score(y_test, predictions, average='weighted'),
        'f1': f1_score(y_test, predictions, average='weighted')
    }
    return metrics, predictions

# Plot confusion matrix
def plot_confusion_matrix(y_test, predictions):
    conf_matrix = confusion_matrix(y_test, predictions)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

# Feature importances
def feature_importances(model, X_encoded):
    importances = model.feature_importances_
    feature_names = X_encoded.columns
    feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    return feature_importances.sort_values(by='Importance', ascending=False)

# Hyperparameter tuning
def hyperparameter_tuning(X_train, y_train):
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    grid_search = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

# Cross-validation
def cross_validation(model, X_encoded, y):
    cv_scores = cross_val_score(model, X_encoded, y, cv=5, scoring='accuracy')
    return cv_scores

# Polynomial features
def polynomial_features(X_encoded):
    poly = PolynomialFeatures(degree=2, interaction_only=True)
    return poly.fit_transform(X_encoded)

# Model comparison
def compare_models(X_train, y_train, X_test, y_test):
    rf_model = RandomForestClassifier()
    gb_model = GradientBoostingClassifier()
    rf_model.fit(X_train, y_train)
    gb_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)
    gb_predictions = gb_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_predictions)
    gb_accuracy = accuracy_score(y_test, gb_predictions)
    return rf_accuracy, gb_accuracy

# Additional evaluation metrics
def additional_metrics(model, X_test, y_test):
    y_test = np.squeeze(y_test)  # Ensure y_test is 1D
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')
    precision, recall, _ = precision_recall_curve(y_test, model.predict_proba(X_test)[:, 1])
    return roc_auc, precision, recall


def main():
    X, y = load_dataset()
    X_encoded = preprocess_data(X)
    X_train, X_test, y_train, y_test = split_data(X_encoded, y)
    print(f"Shape of y_train: {y_train.shape}")  # Ensure y_train is 1D
    print(f"Shape of y_test: {y_test.shape}")    # Ensure y_test is 1D
    model = train_model(X_train, y_train)
    metrics, predictions = evaluate_model(model, X_test, y_test)
    print(f"Accuracy: {metrics['accuracy']}")
    print(f"Precision: {metrics['precision']}")
    print(f"Recall: {metrics['recall']}")
    print(f"F1 Score: {metrics['f1']}")
    plot_confusion_matrix(y_test, predictions)
    feature_importances_df = feature_importances(model, X_encoded)
    print(feature_importances_df)
    best_model = hyperparameter_tuning(X_train, y_train)
    cv_scores = cross_validation(model, X_encoded, y)
    print(f"Cross-Validation Accuracy Scores: {cv_scores}")
    print(f"Mean Cross-Validation Accuracy: {cv_scores.mean()}")
    X_poly = polynomial_features(X_encoded)
    rf_accuracy, gb_accuracy = compare_models(X_train, y_train, X_test, y_test)
    print(f"Random Forest Accuracy: {rf_accuracy}")
    print(f"Gradient Boosting Accuracy: {gb_accuracy}")
    roc_auc, precision, recall = additional_metrics(model, X_test, y_test)
    print(f"ROC-AUC Score: {roc_auc}")

if __name__ == "__main__":
    main()

