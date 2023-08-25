import pandas as pd

df = pd.read_csv("Datasets/Tourist_Dataset.csv")

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

""" # filling the median value for numerical features"""
for col in df_num.columns:
    df[col] = df_num[col].fillna(df[col].median())

"""# filling up mode in the categorical columns"""
for col in df_cat.columns:
    df[col] = df_cat[col].fillna(df[col].value_counts().index[0])

df_num = df.select_dtypes(['int64', 'float64'])
df_num = df_num.drop('CustomerID', axis=1)

"""### **Cleaning data**"""
df_cat.Gender = df.Gender.replace("Fe Male", "Female")

"""# Treating outliers"""
def outlier_cap(x):
    x = x.clip(lower=x.quantile(0.05))  # method assigns values outside boundary to boundary values
    x = x.clip(upper=x.quantile(
        0.95))  # The quantile() method calculates the quantile of the values in a given axis. Default axis is row.
    return (x)


for col in df_num.columns:
    df_num[col] = outlier_cap(df_num[col])

"""Concatenating df_num and df_cat"""
cleaned_data = pd.concat([df_cat, df_num], axis=1, join='inner')

"""# Saving the cleaned data into a csv"""
cleaned_data.to_csv('Datasets/Cleaned_Tourist_Dataset.csv')