from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pandas as pd

def train_lda_classifier(X, y, test_size=0.2, random_state=42):
    """
    Trains a Linear Discriminant Analysis (LDA) classifier on the given dataset.

    Parameters:
        X (array-like): Features matrix (n_samples, n_features).
        y (array-like): Target vector (n_samples,).
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.

    Returns:
        lda (LinearDiscriminantAnalysis): Trained LDA classifier.
        accuracy (float): Accuracy of the model on the test set.
        report (str): Classification report for the test set.
    """
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Initialize the LDA classifier
    lda = LinearDiscriminantAnalysis()

    # Train the LDA classifier
    lda.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = lda.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Generate a classification report
    report = classification_report(y_test, y_pred)

    # Print results
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)

    return lda, accuracy, report

# Example usage with synthetic data
if __name__ == "__main__":
    from sklearn.datasets import make_classification

    # Generate a synthetic dataset
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=15, n_classes=3, random_state=42
    )

    # Train the LDA classifier
    lda_model, test_accuracy, test_report = train_lda_classifier(X, y)
