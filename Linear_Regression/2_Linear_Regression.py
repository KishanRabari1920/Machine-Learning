import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
import matplotlib.pyplot as plt



df = pd.read_csv('2_linear_regression_student_exam.csv')
print(df.head())


x = df[['daily_study_hours']]
y = df['exam_score']


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

new_study_Hours = [[5]]
new_data_pred = model.predict(new_study_Hours)
print(f"Study_Hours :  {new_study_Hours}\n Marks : {new_data_pred}")

plt.scatter(y_test,y_pred)
plt.xlabel("Actual_Score")
plt.ylabel("Predicted_Score")
plt.title("Actual_Score Vs Predicted Score")
plt.show()