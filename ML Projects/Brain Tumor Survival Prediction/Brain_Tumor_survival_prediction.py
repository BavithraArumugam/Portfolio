import pandas as pd
import io
import os.path
import glob
import numpy as np
import keras
import keras_resnet
import keras_resnet.models
import glob
from sklearn.model_selection import KFold
import sklearn
from keras_applications.resnet import ResNet50
from sklearn.model_selection import StratifiedKFold
from keras.optimizers import adam
from sklearn.metrics import accuracy_score
#Read the Survival Data file
data = pd.read_csv('survival_data - survival_data.csv')
print(data.tail())

#Calculate Median Survival time from Survival Data file
med= data['Survival'].median()

label = []
for row in data['Survival']:
    if row >= med:      
        label.append('1')
    else: 
        label.append('0')
data['Label']=label
print('\n\nThe median Survival Time is \n\n' + str(med))

current_dir="MICCAI_BraTS_2018/MICCAI_BraTS_2018_Data_Training/"

drop_rows = []
train_data, train_label = [],[]
for i in range(len(data)):
    file1 = data['BraTS18ID'][i]
    filename= current_dir +file1 + '/' + file1 + '.npy'
    if not os.path.exists(filename):
        drop_rows.append(i)
    else:
        # valid file 
        train_data.append(np.load(filename))
        train_label.append(data['Label'][i])
data.drop(drop_rows)

print(np.shape(train_data), np.shape(train_label))
print(np.shape(train_data[1]))


X_train=np.array(train_data[1]).reshape(-1,240,240,155)

x=keras.layers.Input(np.shape(train_data[1]))
print(x.shape)

model=keras_resnet.models.ResNet18(x, classes=2)
model.compile("adam", "categorical_crossentropy", ["accuracy"])
print(model.summary())

training_y=keras.utils.np_utils.to_categorical(train_label)
X_train=np.array(train_data).reshape(-1,240,240,155)


X_train=X_train.astype('int32')
training_y=training_y.astype('int32')
y=[]
for item in train_label:
	y.append(int(item))
y=np.asarray(y)

cv=StratifiedKFold(n_splits=5)
acc_score =0.0
def five_fold(model, xtrain, xtest, ytrain, ytest):
	model.fit(xtrain, ytrain, epochs=80, batch_size=4)
	ypred=model.predict(xtest)
	pred_labels=np.argmax(ypred, axis=1)
	print(pred_labels)
	test_labels=np.argmax(ytest, axis=1)
	accuracy = accuracy_score(test_labels, pred_labels)
#	acc_score=acc_score+accuracy
	print(accuracy)
	print(accuracy.dtype)
	print(ytest)
	print(ypred)
	return accuracy
for train_index, test_index in cv.split(X_train, y):
	print("TRAIN:", train_index, "TEST:", test_index)
	x_train, X_test = X_train[train_index], X_train[test_index]
	y_train, y_test = y[train_index], y[test_index]
	y_train=keras.utils.np_utils.to_categorical(y_train)
	y_test=keras.utils.np_utils.to_categorical(y_test)
	acc=five_fold(model,x_train,X_test, y_train, y_test)
	acc_score=acc_score+acc

average_test_accuracy = acc_score/5
print("Average Test Accuracy from 5 Fold CV is \n\n")
print(average_test_accuracy)














