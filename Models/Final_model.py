# Data manipulation and handling libraries

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

df = pd.read_csv("../Datasets/Cleaned_Tourist_Dataset.csv")
df = df.drop('Unnamed: 0', axis=1)

# defining target feature and independent features
Y = df["ProdTaken"]
X = df.drop("ProdTaken", axis=1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=25)

cat_cols = ['CityTier', 'TypeofContact', 'Occupation', 'Gender', 'NumberOfPersonVisited',
            'NumberOfFollowups', 'ProductPitched', 'MaritalStatus', 'PreferredPropertyStar', 'NumberOfTrips',
            'Passport', 'PitchSatisfactionScore', 'OwnCar', 'NumberOfChildrenVisited', 'Designation']
num_cols = ["Age", "DurationOfPitch", "MonthlyIncome"]

# Create transformers for categorical and numeric features
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])
# Combine transformers using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, cat_cols),
        ('num', numeric_transformer, num_cols)
    ]
)

# Create the full pipeline with feature selection
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('k_best', SelectKBest(score_func=f_classif, k=19))  # Adjust 'k' as needed
])

# Fit the pipeline on training data and transform it
X_train_preprocessed = pipeline.fit_transform(X_train, Y_train)
X_train_preprocessed_df = pd.DataFrame(X_train_preprocessed.toarray())
print(X_train_preprocessed_df.head())

# Transform the testing data using the same pipeline
X_test_preprocessed = pipeline.transform(X_test)
X_test_preprocessed_df = pd.DataFrame(X_test_preprocessed.toarray())
print(X_test_preprocessed_df.head())

"""## **Model Building**"""

# ML libraries
# from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

params = {
    'objective': 'binary:logistic',
    'max_depth': 10,
    'alpha': 10,
    'learning_rate': 1,
    'n_estimators': 100
}

xgb_clf = XGBClassifier(**params)

xgb_clf.fit(X_train_preprocessed_df, Y_train)
"""### **Decision Tree Classifier**"""

# # DecisionTreeClassifier with gini and class_weight for appropriate importance
# dtc = DecisionTreeClassifier(criterion="gini", class_weight={0: 0.15, 1: 0.85}, random_state=1)
# # fit the model on training dataset
# dtc.fit(X_train_preprocessed_df, Y_train)


"""Accuracy score"""
# predicting on train and tests
pred_train = xgb_clf.predict(X_train_preprocessed_df)
pred_test = xgb_clf.predict(X_test_preprocessed_df)

# accuracy of the model
train_acc = xgb_clf.score(X_train_preprocessed_df, Y_train)
test_acc = xgb_clf.score(X_test_preprocessed_df, Y_test)

print("train: ", train_acc)
print("test: ", test_acc)

P_value = xgb_clf.predict_proba(X_test_preprocessed_df)
print("Prob", P_value)

"""## **Storing model in pkl file**"""

import pickle

pickle.dump(xgb_clf, open("model.pkl", "wb"))
model = pickle.load(open("model.pkl", "rb"))

pickle.dump(pipeline, open("preprocessing_pipeline.pkl", "wb"))
preprocessing_pipeline = pickle.load(open("preprocessing_pipeline.pkl", "rb"))
