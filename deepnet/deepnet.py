import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, Flatten, Dropout, MaxPooling2D
from keras.optimizers import SGD
import tensorflow as tf

import pandas_ml as pdml
import imblearn
from pandas_ml import ConfusionMatrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

df = pd.read_excel('knndata.xlsx')

#df.drop(['quality'],1,inplace=True)
df.drop(['sec'],1,inplace=True)
df.drop(['age'],1,inplace=True)
df.drop(['failprob'],1,inplace=True)
#df.drop(['output(ab)normal'],1,inplace=True)
num_instances, num_features = df.shape

X = df.iloc[:,:-1]
y = df['class']
print(df.head())

anomalies = df.loc[df['class'] == 1]
non_anomalies = df.loc[df['class'] == 0]
print("We have", len(anomalies), "fraud data points and", len(non_anomalies), "regular data points.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

model = Sequential()
model.add(Dense(num_features-1, input_dim=num_features-1, activation='relu'))     # kernel_initializer='normal'
model.add(Dense(1, activation='sigmoid'))                 # kernel_initializer='normal'
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(X_train.as_matrix(), y_train, epochs=1)

print("Loss: ", model.evaluate(X_test.as_matrix(), y_test, verbose=0))

y_predicted = model.predict(X_test.as_matrix()).T[0].astype(int)

y_right = np.array(y_test)
confusion_matrix = ConfusionMatrix(y_right, y_predicted)
print("Confusion matrix:\n%s" % confusion_matrix)
confusion_matrix.plot(normalized=True)
plt.show()

confusion_matrix.print_stats()

#example_measures = np.array([[4,1],[1.34,0.605],[1.7877,0.3696],[0.23,0.01],[4.98,0.1234],[0.34,0.7]])
#prediction = model.predict(example_measures).T[0].astype(int)
#print(prediction)

df2 = pdml.ModelFrame(X_train, target=y_train)
sampler = df2.imbalance.over_sampling.SMOTE()
oversampled = df2.fit_sample(sampler)
X2, y2 = oversampled.iloc[:,:-1], oversampled['class']

data = scale(X2)
pca = PCA(n_components=num_features-1)
X2 = pca.fit_transform(data)

model2 = Sequential()
model2.add(Dense(num_features-1, input_dim=num_features-1, activation='relu')) 
model2.add(Dense(27, activation='relu'))
model2.add(Dense(20, activation='relu'))
model2.add(Dense(15, activation='relu'))
model2.add(Dense(1, activation='sigmoid'))
model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model2.summary()

X2_test = pca.fit_transform(X_test)
h = model2.fit(X2, y2, epochs=2, validation_data=(X2_test, y_test))

y2_predicted = np.round(model2.predict(X2_test)).T[0].astype(int)
y2_correct = np.array(y_test)

example_measures= pd.read_excel('knndatatest2.xlsx')
example_measures.drop(['sec'],1,inplace=True)
example_measures.drop(['age'],1,inplace=True)
example_measures.drop(['failprob'],1,inplace=True)
example_measures.drop(['class'],1,inplace=True)
prediction2 = np.round(model2.predict(example_measures)).T[0].astype(int)
print(prediction2)

confusion_matrix2 = ConfusionMatrix(y2_correct, y2_predicted)
print("Confusion matrix:\n%s" % confusion_matrix2)
confusion_matrix2.plot(normalized=True)
plt.show()