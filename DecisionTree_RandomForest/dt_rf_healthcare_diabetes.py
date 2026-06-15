import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#Load Dataset
df = pd.read_csv('healthcare_diabetes.csv')
# print(df.head())

#Check Shape
# print(f"Dataset Shape : {df.shape}")

#Check Missing
print(df.isna().sum())

#Handle Missing Values
df['BMI'] = df['BMI'].fillna(df['BMI'].median())
df['Cholesterol'] = df['Cholesterol'].fillna(df['Cholesterol'].median())
df['FastingGlucose'] = df['FastingGlucose'].fillna(df['FastingGlucose'].median())
df['AlcoholConsumption'] = df['AlcoholConsumption'].fillna(df['AlcoholConsumption'].mode()[0])

#Features & Target
X = df[[
    'Age',
    'Gender',
    'BMI',
    'BloodPressure_Sys',
    'BloodPressure_Dia',
    'Cholesterol',
    'FastingGlucose',
    'HbA1c',
    'Insulin',
    'SmokingStatus',
    'PhysicalActivity',
    'FamilyHistoryDiabetes',
    'AlcoholConsumption'
]]

y = df['Diabetes']

#Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

#Encoding Data
ct = ColumnTransformer(
    transformers=[
        (
            'encoder',
            OneHotEncoder(sparse_output=False),
            [
                'Gender',
                'SmokingStatus',
                'PhysicalActivity',
                'AlcoholConsumption'
            ]
        )
    ],
    remainder='passthrough'
)

X_train = ct.fit_transform(X_train)
X_test = ct.transform(X_test)

# print(f"X_train : {X_train}")

# Train Decision Tree model
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

#Train Random Forest model
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

#Predictions
dt_train_pred = dt.predict(X_train)
dt_test_pred = dt.predict(X_test)

rf_train_pred = rf.predict(X_train)
rf_test_pred = rf.predict(X_test)

#Compare Accuracy
dt_train_acc = accuracy_score(y_train, dt_train_pred)
dt_test_acc = accuracy_score(y_test, dt_test_pred)

rf_train_acc = accuracy_score(y_train, rf_train_pred)
rf_test_acc = accuracy_score(y_test, rf_test_pred)

print(f"Metrics")
print(f"Train Accuracy : ( dt_train_acc: {dt_train_acc},rf_train_acc: {rf_train_acc} )")
print(f"Test Accuracy : (dt_test_acc: {dt_test_acc}, rf_test_acc: {rf_test_acc})")
print(f"Variance : (dt_train_acc-dt_test_acc: {dt_train_acc-dt_test_acc},\nrf_train_acc-rf_test_acc: {rf_train_acc-rf_test_acc})")

#Precision / Recall / F1
dt_precision = precision_score(y_test, dt_test_pred)
dt_recall = recall_score(y_test, dt_test_pred)
dt_f1 = f1_score(y_test, dt_test_pred)

rf_precision = precision_score(y_test, rf_test_pred)
rf_recall = recall_score(y_test, rf_test_pred)
rf_f1 = f1_score(y_test, rf_test_pred)

print(f"Precision : ( dt : {dt_precision}, rf : {rf_precision} )")
print(f"Recall : ( dt : {dt_recall}, rf : {rf_recall} )")
print(f"F1 Score : ( dt : {dt_f1}, rf : {rf_f1} )")

#Feature Importance
print("\nFeature Importance :")

feature_names = ct.get_feature_names_out()

importance_pairs = sorted(
    zip(
        feature_names,
        dt.feature_importances_,
        rf.feature_importances_
    ),
    key=lambda x: x[2],
    reverse=True
)
# print(f"importance_pairs : \n{importance_pairs}")

for feat, dt_imp, rf_imp in importance_pairs:

    print(f"{feat} - [ dt: {dt_imp}, rf: {rf_imp} ]")