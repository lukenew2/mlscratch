"""Module containing tests for mlscratch/supervised/logistic.py."""
from numpy.testing import assert_array_almost_equal
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from mlscratch.utils.metrics import accuracy_score
from mlscratch.utils.preprocessing import StandardScaler
from mlscratch.supervised.logistic import LogisticRegression

def test_logistic_simple():
    """Test logistic regression with a simple dataset."""
    X = [[-1, -1], [-1, 0], [0, 1], [1, 1]]
    y = [0, 0, 1, 1]

    clf = LogisticRegression(n_iter=3000)

    clf.fit(X, y)

    preds = clf.predict(X)

    assert preds.shape == (len(y),)
    assert_array_almost_equal(clf.predict(X), y)

    probas = clf.predict_proba(X)

    assert probas.shape == (len(y),)

def test_logistic_iris():
    """Test logistic regression with the iris dataset."""
    # Load iris dataset and transform into binary classification problem.
    iris = load_iris()
    X = iris.data[iris.target != 0]
    y = iris.target[iris.target != 0]
    y[y==1] = 0
    y[y==2] = 1

    # Split dataset into training and test sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,
                                                        random_state=42)

    # Preprocess data by standardization.
    scaler = StandardScaler()
    scaler.fit_transform(X_train)
    scaler.transform(X_test)

    # Train model and predict test set.
    clf = LogisticRegression(n_iter=10000, lr=1e-3)
    clf.fit(X_train, y_train)
    y_preds = clf.predict(X_test)

    # Compute accuracy of predictions.
    score = accuracy_score(y_test, y_preds)

    assert score > 0.80
    
    

