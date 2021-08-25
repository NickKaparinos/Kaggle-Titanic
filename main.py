"""
Nick Kaparinos
Titanic - Machine Learning from Disaster
Kaggle Competition
"""

from utilities import *
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
import time


def main():
    # Options
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    save_submission = False
    start = time.perf_counter()

    # Read data
    X_train, y_train, X_test = read_data()

    # Classifier
    # classifier = RandomForestClassifier(random_state=0)
    # classifier = DecisionTreeClassifier(random_state=0)
    classifier = BaggingClassifier(base_estimator=DecisionTreeClassifier(random_state=0), random_state=0)
    # # classifier = LogisticRegression(random_state=0)
    # classifier = SVC(random_state=0)
    # classifier = KNeighborsClassifier()
    # classifier = MLPClassifier(random_state=0)
    # classifier = GaussianNB()
    # classifier = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(random_state=0),
    #                                 random_state=0)
    # classifier = GradientBoostingClassifier(random_state=0)

    # Pipeline
    pipe = Pipeline([
        ('scale', MinMaxScaler()),
        ('clf', classifier)])
    print(pipe)

    # Hyper parameter grid
    param_grid = {}
    if 'LogisticRegression' in str(classifier):
        param_grid = {'clf__C': [0.1, 0.5, 1, 2, 5, 10]}

    if 'SVC' in str(classifier):
        param_grid = {'clf__kernel': ['rbf'], 'clf__C': [0.1, 0.5, 1, 2, 5, 10],
                      'clf__gamma': ['scale', 'auto', 0.1, 0.5, 1, 2, 5, 10]}

    if 'KNeighbors' in str(classifier):
        param_grid = {'clf__n_neighbors': [3, 5, 7, 9]}

    if 'DecisionTree' in str(classifier):
        param_grid = {'clf__criterion': ['gini', 'entropy'], 'clf__min_samples_leaf': [1, 2, 3],
                      'clf__max_depth': [1, 2, 3, 5, 7, 10, 15, None], 'clf__min_samples_split': [2, 3, 4, 5, 6],
                      'clf__max_features': ['auto', 'sqrt', 'log2']}

    if 'RandomForest' in str(classifier):
        param_grid = {'clf__n_estimators': [200, 300, 500], 'clf__criterion': ['gini', 'entropy'],
                      'clf__min_samples_leaf': [1, 2, 3], 'clf__max_depth': [3, 5, 7, 10, 15, None],
                      'clf__min_samples_split': [2, 3, 4, 5], 'clf__max_features': ['auto', 'sqrt', 'log2']}

    if 'MLPClassifier' in str(classifier):
        param_grid = {
            'clf__hidden_layer_sizes': [(10), (50), (100), (10, 10), (50, 50), (100, 100), (10, 10, 10), (50, 50, 50),
                                        (100, 100, 100)],
            'clf__activation': ['identity', 'relu'], 'clf__solver': ['lbfgs', 'adam'],
            'clf__alpha': [0.0001, 0.0005, 0.001],
            'clf__max_iter': [200, 400]}

    if 'BaggingClassifier' in str(classifier):
        param_grid = {'clf__n_estimators': [25, 50, 100, 200],
                      'clf__max_features': [1.0, 0.8, 0.5],
                      'clf__max_samples': [1.0, 0.8, 0.5],
                      'clf__base_estimator__max_depth': [1, 2, 3, 5, 10, 15, None],
                      'clf__base_estimator__criterion': ['gini', 'entropy'],
                      'clf__base_estimator__min_samples_split': [2, 3]}

    if 'AdaBoost' in str(classifier):
        param_grid = {'clf__n_estimators': [10, 25, 50, 100, 200], 'clf__learning_rate': [0.5, 0.75, 1.0, 1.25, 1.5],
                      'clf__base_estimator__criterion': ['gini', 'entropy'],
                      'clf__base_estimator__min_samples_leaf': [1, 2, 3],
                      'clf__base_estimator__max_depth': [2, 3, 5, 10, None],
                      'clf__base_estimator__min_samples_split': [2, 3]}

    if 'GradientBoosting' in str(classifier):
        param_grid = {'clf__learning_rate': [0.25, 0.5, 1.0, 1.5], 'clf__n_estimators': [50, 100, 200],
                      'clf__subsample': [1.0, 0.75, 0.5], 'clf__max_depth': [2, 3, 5, None],
                      'clf__max_features': ['auto', 'sqrt'], 'clf__min_samples_leaf': [1, 2, 3],
                      'clf__min_samples_split': [2, 3, 4, 5]}
    # Cross validation
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    grid = GridSearchCV(pipe, param_grid=param_grid, cv=cv, n_jobs=5, verbose=2, scoring='accuracy',
                        return_train_score=False, refit=True).fit(X_train, y_train.iloc[:, 0])

    # Results
    cv_results = pd.DataFrame(grid.cv_results_)
    print(pipe)
    print(cv_results)

    # Predictions
    y_pred = grid.predict(X_test)
    results_pd = pd.DataFrame(columns=['PassengerId', 'Survived'])
    results_pd['PassengerId'] = X_test.index
    results_pd['Survived'] = y_pred

    if save_submission:
        results_pd.to_csv('submission.csv', index=False)

    # Execution Time
    end = time.perf_counter()
    print(f"\nExecution time = {end - start:.2f} second(s)")


if __name__ == '__main__':
    main()
