import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error, precision_score, recall_score, f1_score
from time import time
import numpy as np
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('Final Data.csv')

df_melt = df.melt(id_vars='Name Of District', var_name='Year-Gender-Class', value_name='Value')

df_melt[['Year', 'Gender', 'Class']] = df_melt['Year-Gender-Class'].str.split('-', expand=True)

df_melt['Year'] = df_melt['Year'].astype(int)

X = df_melt[['Year', 'Gender', 'Class', 'Name Of District']]
y = df_melt['Value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

le_gender = LabelEncoder()
le_district = LabelEncoder()

X_train['Gender'] = le_gender.fit_transform(X_train['Gender'])

X_train['Name Of District'] = le_district.fit_transform(X_train['Name Of District'])

X_test['Gender'] = le_gender.transform(X_test['Gender'])

X_test['Name Of District'] = le_district.transform(X_test['Name Of District'])

models = [

    ("Random Forest", RandomForestRegressor(), 33),
]

results = []

for name, model, params in models:
    start_time = time()
    model.fit(X_train, y_train)
    training_time = round(time() - start_time, 2)
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    results.append({
        "Model": name,
        "Accuracy": model.score(X_test, y_test),
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R-SQ": r2_score(y_test, y_pred),
        "MedAE": median_absolute_error(y_test, y_pred),
        "Number of Parameters": params
    })

results_df = pd.DataFrame(results)
print(results_df)

new_data = pd.DataFrame({
    'Year': [2026],
    'Gender': ['Boys'],
    'Class': ['9'],
    'Name Of District': ['Ahmadabad']
})

new_data['Gender'] = le_gender.transform(new_data['Gender'])
new_data['Name Of District'] = le_district.transform(new_data['Name Of District'])

for name, model, params in models:
    prediction = model.predict(new_data)
    print(f"The prediction for 2020 Boys class 9 in Ahmedabad using {name} is {prediction}")



