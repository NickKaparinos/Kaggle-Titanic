"""
Nick Kaparinos
Titanic - Machine Learning from Disaster
Kaggle Competition
"""

from utilities import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
import time

# Options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
save_submission = True
start = time.perf_counter()

# Read train data
X_train, y_train, X_test = read_data()

# Pipeline
classifier = RandomForestClassifier(random_state=0)
randomforest_optimal = RandomForestClassifier(n_estimators=200, criterion='entropy', max_depth=10, min_samples_leaf=2,
                                              min_samples_split=5, random_state=0)

pipe = Pipeline([
    ('scale', StandardScaler()),
    ('clf', classifier)])
print(pipe)

# Cross validation
# params = {'clf__kernel': ['rbf', 'linear', 'sigmoid', 'poly']}
# params = {'clf__n_neighbors': [3, 5, 7, 9]}
# params = {'clf__n_estimators': [100, 200, 300], 'clf__criterion': ['gini', 'entropy'],
#           'clf__min_samples_leaf': [1, 2, 3],
#           'clf__max_depth': [1, 2, 3, 5, 7, 10, 12, None],
#           'clf__min_samples_split': [2, 3, 4, 5, 6],
#           'clf__max_features': ['auto', 'sqrt', 'log2']}
params = {'clf__n_estimators': [200], 'clf__criterion': ['gini'],
          'clf__min_samples_leaf': [1],
          'clf__max_depth': [10],
          'clf__min_samples_split': [2]}
# params = {'clf__activation': ['identity', 'logistic', 'tanh', 'relu'], 'clf__solver': ['lbfgs', 'adam'],
#           'clf__learning_rate': ['constant', 'invscaling', 'adaptive'], 'clf__max_iter': [200, 300, 400]}
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
grid = GridSearchCV(pipe, param_grid=params, cv=cv, n_jobs=5, verbose=2, scoring='accuracy',
                    return_train_score=False, refit=True)
grid.fit(X_train, y_train.iloc[:,0])
cv_results = pd.DataFrame(grid.cv_results_)
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
debug = True
