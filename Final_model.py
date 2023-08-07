# Data manipulation and handling libraries
import numpy as np
import pandas as pd

df = pd.read_csv("Tourist_Dataset.csv")

"""### **Defining Target Feature**"""

Y = df["ProdTaken"]
X = df.drop("ProdTaken", axis=1)

"""### **Segregating numerical and categorical**"""

df_num = df.select_dtypes(['int64', 'float64'])
df_num.head().transpose()

convert_typelist = ['ProdTaken', 'CityTier', 'NumberOfPersonVisited', 'NumberOfFollowups', 'PreferredPropertyStar',
                    'NumberOfTrips', 'Passport', 'PitchSatisfactionScore', 'OwnCar', 'NumberOfChildrenVisited']

for col in convert_typelist:
    df[col] = df[col].astype('object')
df_cat = df.select_dtypes(['object'])

for i in df_cat.columns:
    df[i] = df[i].astype("category")
    df_cat[i] = df_cat[i].astype("category")
df_num = df.select_dtypes(['int64', 'float64'])

# filling the median value
for col in df_num.columns:
    df[col] = df_num[col].fillna(df[col].median())

# filling up mode in the categorical columns
for col in df_cat.columns:
    df[col] = df_cat[col].fillna(df[col].value_counts().index[0])

df_num = df.select_dtypes(['int64', 'float64'])
df_num = df_num.drop('CustomerID', axis=1)
"""### **Looking through Categorical columns**"""

"""### **Cleaning data**"""
df_cat.Gender = df.Gender.replace("Fe Male", "Female")

# Treating outliers
def outlier_cap(x):
    x = x.clip(lower=x.quantile(0.05))  # method assigns values outside boundary to boundary values
    x = x.clip(upper=x.quantile(
        0.95))  # The quantile() method calculates the quantile of the values in a given axis. Default axis is row.
    return (x)


for col in df_num.columns:
    df_num[col] = outlier_cap(df_num[col])

""" **Creating Dummies** """
Y = df_cat["ProdTaken"]
df_cat = df_cat.drop("ProdTaken", axis=1)

# creating features with n-1 variables
df_dummies = pd.get_dummies(df_cat, drop_first=True)
df_dummies.head()

"""## **Selecting KBest**"""
from sklearn.feature_selection import SelectKBest, chi2

selector = SelectKBest(chi2, k=20)
selector.fit_transform(df_dummies, Y)

# Get columns to keep and create a new dataframe with selected features
cols = selector.get_support(indices=True)
selected_features_df_char = df_dummies.iloc[:, cols]

X = pd.concat([selected_features_df_char, df_num], axis=1, join='inner')
X.head()

"""## **Splitting Data**"""

from sklearn.model_selection import train_test_split

# splitting data into training and test set, use stratify to maintain the original distribution of Dependent variable as of original set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=25, stratify=Y)

# creating a list of column names
feature_names = X_train.columns.to_list()

"""## **Model Building**"""

# ML libraries
from sklearn.tree import DecisionTreeClassifier

"""### **Decision Tree Classifier**"""

# DecistionTreeClassifier with gini and class_weight for appropriate importance
dtc = DecisionTreeClassifier(criterion="gini", class_weight={0: 0.15, 1: 0.85}, random_state=1)
# fit the model on training dataset
dtc.fit(X_train, Y_train)

"""## **Storing model in pkl file**"""

import pickle

pickle.dump(dtc, open("model.pkl", "wb"))
model = pickle.load(open("model.pkl", "rb"))

pickle.dump(dtc, open("model.pkl", "wb"))
model = pickle.load(open("model.pkl", "rb"))
