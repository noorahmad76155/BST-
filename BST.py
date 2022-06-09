#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Finger
# importing required libraries
import pandas as pd
import matplotlib.pyplot as plot
import numpy as np
# read the train and test dataset
df = pd.read_csv('Earloobe data.csv')
df= df.drop([22])
df=df.drop(columns='Unnamed: 0')
# Importing the dependancies
from sklearn import metrics
# Predicted values
#values of blood sugar test using finger
y_pred =df.iloc[:,3]
y_pred=y_pred.astype(str).astype(float)
# Actual values
#values of blood sugar test using Arm(Base)
y_act =df.iloc[:,2]
y_act= y_act.astype(str).astype(float)
print(y_pred)
#Evaluating Regression Models
'''
firstly
Accuracy (e.g. classification accuracy) is a measure for classification, not regression.

We cannot calculate accuracy for a regression model.
The skill or performance of a regression model must be reported as an error in those predictions.
There are three error metrics that are commonly used for evaluating and reporting the performance of a regression model; they are:

Mean Squared Error (MSE).
Root Mean Squared Error (RMSE).
Mean Absolute Error (MAE)
R2 score 

'''

#using scikit learning API 
#import sklearn library 
import sklearn.metrics as sm
#Metrics for Regression
print("Mean absolute error =",(sm.mean_absolute_error(y_act, y_pred), 2)) 
print("Mean squared error =", (sm.mean_squared_error(y_act, y_pred), 2)) 
print("Median absolute error =", (sm.median_absolute_error(y_act, y_pred), 2)) 
print("Explain variance score =", (sm.explained_variance_score(y_act, y_pred), 2)) 
print("R2 score =",(sm.r2_score(y_act, y_pred), 2))
# Draw the scatter plot
import numpy as np
import matplotlib.pyplot as plot
# Draw the scatter plot
plot.scatter(y_pred, y_act)
plot.title('the correctness of blood sugar test between Finger and Arm')
plot.xlabel('Finger')
plot.ylabel('Arm')

plot.show()


# In[17]:


#Earlob
# importing required libraries
import pandas as pd
import numpy as np
# read the train and test dataset
df = pd.read_csv('Earloobe data.csv')
df= df.drop([22])
df=df.drop(columns='Unnamed: 0')
# Importing the dependancies
from sklearn import metrics
# Predicted values
##values of blood sugar test using Earlob
y_pred =df.iloc[:,4]
y_pred=y_pred.astype(str).astype(float)
# Actual values
#values of blood sugar test using Arm(Base)
y_act =df.iloc[:,2]
y_act= y_act.astype(str).astype(float)
#print(y_act)
#print(y_pred)
'''
We cannot calculate accuracy for a regression model.
The skill or performance of a regression model must be reported as an error in those predictions.
There are three error metrics that are commonly used for evaluating and reporting the performance of a regression model; they are:

Mean Squared Error (MSE).
Root Mean Squared Error (RMSE).
Mean Absolute Error (MAE)
R2 score 


'''
import sklearn.metrics as sm
print("Mean absolute error =", (sm.mean_absolute_error(y_act, y_pred), 2)) 
print("Mean squared error =", (sm.mean_squared_error(y_act, y_pred), 2)) 
print("Median absolute error =", (sm.median_absolute_error(y_act, y_pred), 2)) 
print("Explain variance score =", (sm.explained_variance_score(y_act, y_pred), 2)) 
print("R2 score =", (sm.r2_score(y_act, y_pred), 2))
# Draw the scatter plot
N =50
colors = np.random.rand(N)
plot.scatter(y_pred, y_act, c=colors)
plot.title('the correctness of blood sugar test between Earlob and Arm')
plot.xlabel('Earlob')
plot.ylabel('Arm')
plot.show()

