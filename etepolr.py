from importlib.metadata import files
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
import seaborn as sns
import pickle
# %matplotlib inline

boston=load_boston()

# data exploring

# print(cancer.keys())
# print(cancer.DESCR)
# print(boston.data)
# print(boston.feature_names)
# print(cancer.target)
boston_dataset=pd.DataFrame(boston.data)
boston_dataset.columns=boston.feature_names
print(boston_dataset[ : 1])
boston_dataset["rate"]=boston.target
# print(cancer_dataset[cancer_dataset.columns[-1]].head())
# print(boston_dataset.info())
# print(boston_dataset.describe())
# print(boston_dataset.isnull().sum())                                                                                                             

#EDA

# print(boston_dataset.corr())
# print(sns.pairplot(boston_dataset))
# print(plt.scatter(boston_dataset['INDUS'],boston_dataset['rate']))

# independent and dependent features 
print(boston_dataset)
x=boston_dataset.iloc[ : ,:-1]
y=boston_dataset.iloc[:,-1]
# print(y)
# Train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
# print(x_train)
# print(x_test)
# Standardising the dataset to acheive the global mimuma in the internl (Gradient descent)
from sklearn.preprocessing import StandardScaler
# <<<<<<< HEAD
# =======
# import pickle 
# >>>>>>> 7b101c193260e4195a25e33693203661ee1ae176

scaler=StandardScaler()

x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
print(x_train)
print(x_test)
# filename1='standard_scaler'
pickle.dump(scaler,open('scaling.pkl','wb'))
# pickle_scaler=pickle.load(open('scaling.pkl','rb'))

### Model training

from sklearn.linear_model import LinearRegression
regression=LinearRegression()
regression.fit(x_train,y_train)
# print(regression.fit(x_train,y_train))

### intercepts and coffiecients

print(regression.coef_)
print(regression.intercept_)

print(regression.get_params())   #####  which params we use in to train the model

### Prediction of a Model

y_predict=regression.predict(x_test)
print(y_predict)

#### ASUMPTIONS ON SOME OF THE SITUATIOS

plt.scatter(y_test,y_predict)

### residuals(error from actual data and predicted data)

residuals=y_test-y_predict
# print(residuals)
sns.displot(residuals,kind="kde")

#####scatter plot with residuals and predicted values\

plt.scatter(residuals,y_predict)

####Performance Matrix
from sklearn.metrics import mean_squared_error,mean_absolute_error
print(mean_absolute_error(y_predict,y_test))
print(mean_squared_error(y_predict,y_test))
print(np.sqrt(mean_squared_error(y_predict,y_test)))

##### R2  Sqaure and Adjusted R2 Sqaure method to measure the performance of the model
##Adjusted  R2 value is always less than  R2 Square
from sklearn.metrics import r2_score
score=r2_score(y_predict,y_test)
print(score) #### more the value towards the one more will be the accuracy of the model
#### formula for Adjusted R2 square is
### Adjusted R2=1- (1-r2_score)*(len(y_test)-1)/(len(y_test)-x_test.shape[1]-1))
print(1- (1-score)*(len(y_test)-1)/(len(y_test)-x_test.shape[1]-1))

#### New Data Predction
print(boston.data[1].reshape(-1,1).shape) ### here we need to reshape the data because
#  while we are training the model, 
#### we gave 2 dimensional data like rows and coloums but while we are predicting the 
# new data we only giving without columns without the count of number of rows..
# scaler.transform(boston.data[0].reshape(1,-1))
print(regression.predict(boston.data[1].reshape(1,-1)))### here we got negative prediction
#  because we missed standardaistaion of the data 

#### PICKLING THE MODEL

# import pickle
filename="regression_model"
pickle.dump(regression,open('filename','wb'))
pickle_model=pickle.load(open('filename','rb'))
print(pickle_model.predict(scaler.transform(boston.data[1].reshape(1,-1))))
