import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
import matplotlib.pyplot as plt



df = pd.read_csv('polynomial_regression_drug_dosage.csv')
print(df.head())


x = df[['drug_dosage_mg']]
y = df['recovery_days']


x_train,x_test,y_train,y_test = train_test_split(
    x,
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
print("\n\n")

poly = PolynomialFeatures(degree=3)
poly_x  = poly.fit_transform(x)


x_train_poly,x_test_poly,y_train_poly,y_test_poly = train_test_split(
    poly_x,
    y,
    test_size=0.3,
    random_state=42
)

model.fit(x_train_poly,y_train_poly)

y_pred_poly  = model.predict(x_test_poly)
print("y_pred : \n",y_pred_poly)

r2score_poly = r2_score(y_test_poly,y_pred_poly)
mse_poly = mean_squared_error(y_test_poly,y_pred_poly)
print("R2score : ",r2score_poly)
print("Mean Squared Error : ",mse_poly)
print("Coefficient_poly : ",model.coef_)
print("Intercept_poly : ",model.intercept_)

drug_dosage_mg_temp = [[50]]
drug_dosage_mg_temp_poly = poly.transform(drug_dosage_mg_temp)
new_data_pred = model.predict(drug_dosage_mg_temp_poly)
print(f"drug_dosage_mg : {drug_dosage_mg_temp}\nRecovery_days : {new_data_pred}")


plt.scatter(y_test_poly,y_pred_poly)
plt.xlabel("Actual_Recovery_days")
plt.ylabel("Predicted_Recovery_days")
plt.title("Actual Vs Predicted Recovery_days ")
plt.show()