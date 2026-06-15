import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

#Load Dataset
df = pd.read_csv('1_linear_regression_movies.csv')
# print(f"dataframe: {df.head()}")

# print(df.info())

#features and target
x = df[['budget_usd', 'marketing_spend_usd']]
y = df['box_office_revenue_usd']
# print(f"Features (X): {x.head()}")
# print(f"Target (Y): {y.head()}")

# Split data into training and testing
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.3,
    random_state=42
)
# print(f"Training Features (X_train): {x_train.head()}")
# print(f"Testing Features (X_test): {x_test.head()}")
# print(f"Training Target (Y_train): {y_train.head()}")
# print(f"Testing Target (Y_test): {y_test.head()}")

# Create model
model = LinearRegression()
model.fit(x_train, y_train)

# Make predictions
y_pred = model.predict(x_test)
# print(f"Predictions: {y_pred}")

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R-squared: {r2}")

#sample prediction
sample_data = pd.DataFrame({
    'budget_usd': [10000000],
    'marketing_spend_usd': [5000000]
})
sample_prediction = model.predict(sample_data)
print(f"Sample Prediction: {sample_prediction[0]}")


