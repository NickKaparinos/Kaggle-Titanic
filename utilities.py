"""
Nick Kaparinos
Titanic - Machine Learning from Disaster
Kaggle Competition
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def read_data():
    train_data = pd.read_csv("../Datasets/Titanic/train.csv")
    test_data = pd.read_csv("../Datasets/Titanic/test.csv")

    # print(train_data.info())
    # print(test_data.info())

    # Split
    y_train = train_data[['Survived']].copy()
    X_train = train_data.drop(columns=['Survived'])
    X_test = test_data

    # Preprocess # TODO title from name and alone from name
    columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    X_test.index = X_test['PassengerId']
    X_train = X_train.drop(columns=columns_to_drop)
    X_test = X_test.drop(columns=columns_to_drop)

    # Encode
    features_to_encode = ['Sex', 'Embarked']
    const_imputer = SimpleImputer(strategy='constant', fill_value='NaN')
    encoder = LabelEncoder()

    for feature in features_to_encode:
        if X_train[feature].isnull().values.any():
            # If there are missing values, use dummy imputation
            X_train[feature] = const_imputer.fit_transform(X_train[[feature]])
            X_test[feature] = const_imputer.transform(X_test[[feature]])

            # Then encode
            X_train[feature] = encoder.fit_transform(X_train[feature])
            X_test[feature] = encoder.transform(X_test[feature])

            # Remove the encoded 'NaN' value
            nan = np.array(['NaN'])
            X_train[feature] = X_train[feature].replace(int(encoder.transform(nan)[0]), np.nan)
        else:
            # If there are no missing values, encode
            X_train[feature] = encoder.fit_transform(X_train[feature])
            X_test[feature] = encoder.transform(X_test[feature])

    # Impute age
    mean_imputer = SimpleImputer(strategy='mean', fill_value='NaN')
    X_train['Age'] = mean_imputer.fit_transform(X_train[['Age']])
    X_test['Age'] = mean_imputer.transform(X_test[['Age']])

    # Impute 'Embarked'
    median_imputer = SimpleImputer(strategy='median', fill_value='NaN')
    X_train['Embarked'] = median_imputer.fit_transform(X_train[['Embarked']])
    X_test['Embarked'] = median_imputer.transform(X_test[['Embarked']])

    # Impute fare
    mean_imputer.fit(X_train[['Fare']])
    X_test['Fare'] = mean_imputer.transform(X_test[['Fare']])

    # One hot encode embarked
    use_onehot_encoding = False
    if use_onehot_encoding:
        onehot_encoder = OneHotEncoder()
        temp = onehot_encoder.fit_transform(X_train[['Embarked']]).toarray()
        temp = pd.DataFrame(data=temp, columns=onehot_encoder.categories_[0])
        X_train = X_train.join(temp)

        temp = onehot_encoder.transform(X_test[['Embarked']]).toarray()
        temp = pd.DataFrame(data=temp, index=X_test.index, columns=onehot_encoder.categories_[0])
        X_test = X_test.join(temp)

    return X_train, y_train, X_test
