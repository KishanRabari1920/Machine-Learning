import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
df = pd.read_csv('1_titanic.csv')
# print(df.head())

# Handle missing values
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Fare'] = df['Fare'].fillna(df['Fare'].median())  

#feature and target
x = df[['Sex', 'Age', 'Fare', 'Embarked']]
y = df['Survived']
# print(f"x: {x.head()}")
# print(f"y: {y.head()}")


# Encoding
ct = ColumnTransformer(
    transformers=[
        ('encoder', OneHotEncoder(), ['Sex', 'Embarked'])
    ],
    remainder='passthrough'
)
encoded_X = ct.fit_transform(x)
# print(f"encoded_X : {encoded_X}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    encoded_X,
    y,
    test_size=0.2,
    random_state=42
)
# print(f"X_train: {X_train}")
# print(f"X_test: {X_test}")
# print(f"y_train: {y_train}")
# print(f"y_test: {y_test}")

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
# print(f"y_pred :",y_pred)

# Metrics
print(f'Accuracy:  {accuracy_score(y_test, y_pred)}')
print(f'Precision: {precision_score(y_test, y_pred)}')
print(f'Recall:    {recall_score(y_test, y_pred)}')
print(f'F1 Score:  {f1_score(y_test, y_pred)}')