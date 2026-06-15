import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


#Load Breast Cancer Dataset
data = load_breast_cancer()
# print(f"Data : {data}")

#feature and target
X = data.data
y = data.target
# print(f"X : {X}")
# print(f"y : {y}")


#Dataset Shape
print("Original Shape of Data:")
# print(X.shape)


#Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
# print(f"X_train : {X_train}")
# print(f"X_test : {X_test}")
# print(f"y_train : {y_train}")
# print(f"y_test : {y_test}")

#Scale Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# print(f"X_train_scaled : {X_train_scaled}")
# print(f"X_test_scaled : {X_test_scaled}")


#Train Model Before PCA
model_before = LogisticRegression()
model_before.fit(X_train_scaled, y_train)

#Prediction Before PCA
y_pred_before = model_before.predict(X_test_scaled)
# print(f"y_pred_before :\n{y_pred_before}")

#Accuracy Before PCA
accuracy_before = accuracy_score(y_test,y_pred_before)
print("Accuracy Before PCA:", accuracy_before)

#Apply PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
# print(f"X_train_pca :\n{X_train_pca}")
# print(f"X_test_pca : \n{X_test_pca}")

#Check New Shape
print("\nShape After PCA:")
# print(X_train_pca.shape)


#Visualize PCA Data
plt.figure(figsize=(8,6))
plt.scatter(
    X_train_pca[:,0],
    X_train_pca[:,1],
    c=y_train
)
plt.title("PCA Reduced Data")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()


#Train Model After PCA
model_after = LogisticRegression()
model_after.fit(X_train_pca, y_train)

#Predictions After PCA
y_pred_after = model_after.predict(X_test_pca)
# print(f"y_pred_after :\n{y_pred_after}")

#Accuracy After PCA
accuracy_after = accuracy_score(y_test,y_pred_after)
print("Accuracy After PCA:", accuracy_after)