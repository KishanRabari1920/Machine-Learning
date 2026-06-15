import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
import matplotlib.pyplot as plt



df = pd.read_csv('3_multiple_linear_regression_crop_yield.csv')
print(df.head())


x = df[[
    'temperature_celsius',
    'rainfall_mm',
    'soil_type',
    'crop_variety',
    'season',
    'irrigation_type',
    'region',
    'fertilizer_kg_per_acre',
    'sunlight_hours']]
y = df['crop_yield_ton_per_acre']


ct = ColumnTransformer(
    transformers=[('encoder',OneHotEncoder(),['soil_type',
    'crop_variety',
    'season',
    'irrigation_type',
    'region'])],
    remainder= "passthrough"
)

Encoded_x = ct.fit_transform(x)



x_train,x_test,y_train,y_test = train_test_split(
    Encoded_x,
    y,
    test_size=0.3,
    random_state=42
)

model = LinearRegression()
model.fit(x_train,y_train)

y_pred  = model.predict(x_test)
print("y_pred : \n",y_pred)

r2score = r2_score(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
print("R2score : ",r2score)
print("Mean Squared Error : ",mse)
print("Coefficient : ",model.coef_)
print("Intercept : ",model.intercept_)



new_data = pd.DataFrame({
    'temperature_celsius' : [45],
    'rainfall_mm' : [700],
    'soil_type' : ['Silty'],
    'crop_variety' : ['Maize'],
    'season' : ['Zaid'],
    'irrigation_type' : ['Sprinkler'],
    'region' : ['West'],
    'fertilizer_kg_per_acre' : [180],
    'sunlight_hours' : [5]
})

new_data_final = ct.transform(new_data)

new_data_pred = model.predict(new_data_final)
print(f"new_data : {new_data}\n crop_yield_ton_per_acre : {new_data_pred}")


plt.scatter(y_test,y_pred)
plt.xlabel("Actual_crop_yield_ton_per_acre")
plt.ylabel("Predicted_crop_yield_ton_per_acre")
plt.title("Actual Vs Predicted crop_yield_ton_per_acre")
plt.show()