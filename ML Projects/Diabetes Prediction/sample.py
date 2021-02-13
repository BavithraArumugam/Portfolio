import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.preprocessing  import StandardScaler

data = pd.read_csv('diabetes.csv')
print(data.info())

print(data.describe())
print(data.isnull().sum())
print(data.head())

data = data.copy(deep = True)
data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

data['Glucose'].fillna(data['Glucose'].mean(), inplace = True)
data['BloodPressure'].fillna(data['BloodPressure'].mean(), inplace = True)
data['SkinThickness'].fillna(data['SkinThickness'].median(), inplace = True)
data['Insulin'].fillna(data['Insulin'].median(), inplace = True)
data['BMI'].fillna(data['BMI'].median(), inplace = True)


X=data.iloc[:,:-1]
Y=data.iloc[:,[-1]]

print(X.head())
print(Y.head())

X_train, X_test, Y_train, Y_test= train_test_split(X,Y, test_size = 0.2 , random_state =0)
#614+154 = 768

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, Y_train.values.ravel())

Y_pred = knn.predict(X_test)
print(Y_pred)
print("Accuracy:" , metrics.accuracy_score(Y_test, Y_pred))