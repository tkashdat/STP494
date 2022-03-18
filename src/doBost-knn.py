##################################################
### import

### basic 
import matplotlib.pyplot as plt
import numpy as np
import scipy
import pandas as pd
import math

import seaborn as sns; sns.set()
#%matplotlib inline


##sklearn learners
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor

##sklearn metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix

##sklearn model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV

##################################################
### read in boston data
bd = pd.read_csv("http://www.rob-mcculloch.org/data/Boston.csv")

#pull off y=medv and x = lstat
y = bd['medv']
X = bd['lstat'].values[:,np.newaxis]

#plot x vs. y
plt.scatter(X,y)
plt.xlabel('lstat')
plt.ylabel('mdev')

##################################################
### fit one knn

# create model object setting the hyperparameter n_neighbors to 50
knnmod = KNeighborsRegressor(n_neighbors=50)

# fit with training data
knnmod.fit(X,y)

#predict on sorted x training values
Xtest = np.sort(X[:,0])[:,np.newaxis]
yhat = knnmod.predict(Xtest)

#plot fit
plt.scatter(X,y)
plt.plot(Xtest,yhat,c='red')
plt.xlabel('lstat')
plt.ylabel('medv')

##################################################
### train/test split

#train/test split
rng = np.random.RandomState(34)
Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,random_state=rng, test_size=.2)

# model object
kmod = KNeighborsRegressor(n_neighbors=50)

# fit on train
kmod.fit(Xtrain,ytrain)

# predict on test
yhat = kmod.predict(Xtest)

#plot to check predictions
plt.scatter(ytest,yhat)
plt.plot(yhat,yhat,c='red')

#rmse
k50mse = mean_squared_error(ytest,yhat)

 
#check rmse
check  = np.sum((yhat-ytest)**2)/len(ytest)
print('val from fun:',k50mse,' and check val: ',check)


##################################################
###  loop over k using simple train/test split

rng = np.random.RandomState(34)
Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,random_state=rng, test_size=.2)

kvec = np.arange(348) + 2 #values of k to try
ormsev = np.zeros(len(kvec)) # storage for oos rsmse
irmsev = np.zeros(len(kvec)) # storage for in-sample rsmse

for i in range(len(kvec)):
   print(i)
   tmod = KNeighborsRegressor(n_neighbors=kvec[i])
   tmod.fit(Xtrain,ytrain)
   yhat = tmod.predict(Xtest)
   ormsev[i] = math.sqrt(mean_squared_error(ytest,yhat))
   yhat = tmod.predict(Xtrain)
   irmsev[i] = math.sqrt(mean_squared_error(ytrain,yhat))

# plot rmse vs k
plt.scatter(kvec,ormsev,c='blue')
plt.plot(kvec,irmsev,c='red')


# plot rmse vs model complexity
mcmp = np.log(1/kvec) #model complexity
plt.scatter(mcmp,ormsev,c='blue')
plt.plot(mcmp,irmsev,c='red')
plt.xlabel('model complexity = log(1/k)',size='x-large')
plt.ylabel('rmse',size='x-large')
plt.title('rmse: blue: out of sample, red: in sample',size='x-large')

##################################################
###  cross validation using sklearn cross_val_score

# to see a list of scorers:
# sorted(sklearn.metrics.SCORERS.keys()) 

#model object
tempmod = KNeighborsRegressor(n_neighbors=40) #knn with k=40

## rmse from cross validation
cvres = cross_val_score(tempmod,X,y,cv=5,scoring='neg_mean_squared_error') #cross val with 5 folds

# tranform to rmse
rmse = math.sqrt(np.mean(-cvres)) 
print('the rmse for k=40 based on 5-fold is:', rmse)

## do it again but shuffle the data
np.random.seed(34) 
indices = np.random.choice(X.shape[0],X.shape[0],replace=False)
ys = y[indices]
Xs = X[indices,:]
cvres = cross_val_score(tempmod,Xs,ys,cv=5,scoring='neg_mean_squared_error')
rmse = math.sqrt(np.mean(-cvres))
print('the rmse for k=40 based on 5-fold is:', rmse)

##################################################
###  cross validation on a grid of k values using sklearn validation_curve function

# create the knn model
model = KNeighborsRegressor() # create the knn model

# do cv at every value of k in kvec
trainS, testS = validation_curve(model,X,y,'n_neighbors',kvec,cv=10,scoring='neg_mean_squared_error')


# transform neg_mean_squared_error to rmse
trrmse = np.sqrt(-trainS.mean(axis=1))
termse = np.sqrt(-testS.mean(axis=1))

#plot in and out of sample rmse
plt.scatter(mcmp,termse)
plt.plot(mcmp,trrmse,c='red')

##################################################
### cross val on a grid using sklearn GridSearchCV

# hyperparamter values to try in the gid search
param_grid={'n_neighbors' : kvec} # same as above

# grid  is the grid searh object
grid = GridSearchCV(model,param_grid,cv=10,scoring='neg_mean_squared_error')

# now run the grid search
grid.fit(X,y)

grid.best_params_ #best value from grid
grid.best_index_ # index of best value from grid
#check
print(kvec[grid.best_index_])


temp = grid.cv_results_ # results from the grid search (a dictionary)
print(temp.keys()) # what is in temp
temp['mean_test_score'] # this is the average score over folds at the values in param_grid

#transform to rmse
rmsevals = np.sqrt(-temp['mean_test_score'])

# plot
plt.plot(mcmp,rmsevals) # plot model complexity vs. rmse
plt.xlabel('model complexity = log(1/k)',size='x-large')
plt.ylabel('rmse',size='x-large')
plt.title('rmse from GridSearch')

