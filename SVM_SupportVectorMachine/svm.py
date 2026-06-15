import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


#Load Dataset
df = pd.read_csv("svm_social_media_sentiment.csv")
# print(df.head())

# feature and target
X = df[['likes', 'shares', 'comments', 'post_length_chars', 'hashtag_count', 'mention_count', 'follower_count', 'verified_account', 'hour_of_day', 'day_of_week', 'engagement_rate', 'positive_word_count', 'negative_word_count', 'neutral_word_count', 'emoji_count']]
y = df['sentiment']
# print(f"Features : \n{X.head()}")
# print(f"Target : \n{y.head()}")


# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
# print(f"Training Features (X_train): {x_train.head()}")
# print(f"Testing Features (X_test): {x_test.head()}")
# print(f"Training Target (Y_train): {y_train.head()}")
# print(f"Testing Target (Y_test): {y_test.head()}")


# Scale feature data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# print(f"X_train_scaled : {X_train_scaled}")
# print(f"X_test_scaled : {X_test_scaled}")

# Train and evaluate linear kernel SVM model
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train_scaled, y_train)

#rbf predictions
y_pred = svm_model.predict(X_test_scaled)
print(f"y_pred : \n{y_pred}")

linear_acc = accuracy_score(y_test, y_pred)

# Train and evaluate RBF kernel SVM model
svm_rbf = SVC(kernel='rbf', random_state=42)
svm_rbf.fit(X_train_scaled, y_train)

#rbf predictions
y_pred_rbf = svm_rbf.predict(X_test_scaled)
print(f"y_pred_rbf : \n{y_pred_rbf}")

rbf_acc = accuracy_score(y_test, y_pred_rbf)

# Compare accuracy
print(f"Linear Kernel SVM Accuracy: {linear_acc}")
print(f"RBF Kernel SVM Accuracy: {rbf_acc}")
