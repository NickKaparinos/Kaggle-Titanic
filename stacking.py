"""
Nick Kaparinos
Titanic - Machine Learning from Disaster
Kaggle Competition
"""

import pandas as pd
from utilities import *
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.ensemble import VotingClassifier, BaggingClassifier, AdaBoostClassifier, RandomForestClassifier, \
    GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from itertools import combinations
import time


def main():
    # Options
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    save_submission = False
    start = time.perf_counter()

    cvOuter = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    cvInner = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

    # Read data
    X_train, y_train, X_test = read_data()

    # Optimal Classifiers
    random_forest_optimal = RandomForestClassifier(random_state=0, criterion='entropy', max_depth=7,
                                                   max_features='log2', min_samples_leaf=2, min_samples_split=5,
                                                   n_estimators=300)
    decision_tree_optimal = DecisionTreeClassifier(random_state=0, criterion='entropy', max_depth=10,
                                                   max_features='sqrt', min_samples_split=3)
    bagging_optimal = BaggingClassifier(
        base_estimator=DecisionTreeClassifier(random_state=0, criterion='entropy', max_depth=10, min_samples_split=2),
        max_features=1.0, max_samples=1.0, n_estimators=25, random_state=0)
    adaboost_optimal = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(random_state=0, max_depth=2),
                                          learning_rate=1, n_estimators=25, random_state=0)
    gradient_boosting_optimal = GradientBoostingClassifier(random_state=0, learning_rate=0.5, max_depth=3,
                                                           max_features='sqrt', n_estimators=100, subsample=1.0)
    logistic_regression = LogisticRegressionCV(random_state=0)
    svm_optimal = SVC(random_state=0, gamma=2, C=2)
    knn_optimal = KNeighborsClassifier(n_neighbors=5)
    mlp_optimal = MLPClassifier(random_state=0, hidden_layer_sizes=(10, 10), solver='lbfgs')

    final_estimator = LogisticRegressionCV(random_state=0)

    # Combinations of classifiers
    listOfClassifiers = [('bagging', bagging_optimal), ('adaBoost', adaboost_optimal), ('DT', decision_tree_optimal),
                         ('randomForest', random_forest_optimal), ('gradientBoosting', gradient_boosting_optimal),
                         ('knn', knn_optimal), ('logistic', logistic_regression), ('mlpOptimal', mlp_optimal),
                         ('svmOptimal', svm_optimal)]
    number_of_combinations = 4
    comb = combinations(listOfClassifiers, number_of_combinations)
    allCombinations = [i for i in list(comb)]
    print(f"Num of combs = {len(allCombinations)}")

    # Pipeline
    best_cls = StackingClassifier(estimators=[], final_estimator=final_estimator, cv=cvInner,
                                  stack_method='predict', passthrough=True)

    pipe = Pipeline([
        ('scale', MinMaxScaler()),
        ('voting', best_cls)])

    # Cross validation
    param_grid = {'voting__estimators': allCombinations}
    grid = GridSearchCV(estimator=pipe, cv=cvOuter, scoring=['accuracy', 'f1'], param_grid=param_grid,
                        refit='accuracy', n_jobs=10, verbose=1).fit(X_train, y_train.iloc[:, 0])
    best_index = grid.best_index_
    cv_results = pd.DataFrame(grid.cv_results_)
    print(cv_results)
    print(f"Best index = {best_index}")
    print(f"Best params = {grid.best_params_}")

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
