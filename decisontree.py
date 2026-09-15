import pandas as pd
import numpy as np 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

df = pd.read_csv("heart.csv")


X= df.drop(columns=['HeartDisease'])
y= df['HeartDisease']


le_features = LabelEncoder()
X['Sex'] = le_features.fit_transform(X['Sex'])
X['ChestPainType'] = le_features.fit_transform(X['ChestPainType'])
X['RestingECG'] = le_features.fit_transform(X['RestingECG'])
X['ExerciseAngina'] = le_features.fit_transform(X['ExerciseAngina'])
X['ST_Slope'] = le_features.fit_transform(X['ST_Slope'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = DecisionTreeClassifier()
y_pred = model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
# print(y_pred)
print("Accuracy: ", accuracy)

