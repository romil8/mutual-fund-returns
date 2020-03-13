# --------------
# import libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


# Code starts here
data= pd.read_csv(path)
data.shape
data.describe()
data.drop(columns='Serial Number',inplace= True)
data.head()
# code ends here




# --------------
from scipy.stats import chi2_contingency
import scipy.stats as stats

#Critical value 
critical_value = stats.chi2.ppf(q = 0.95, # Find the critical value for 95% confidence*
                      df = 11)   # Df = number of variable categories(in purpose) - 1

#Code starts here


# Subsetting the dataframe
return_rating = data.morningstar_return_rating.value_counts()
risk_rating = data.morningstar_risk_rating.value_counts()

#Concating yes and no into a single dataframe
observed=pd.concat([return_rating.transpose(),risk_rating.transpose()], axis = 1,keys=['return','risk'])

chi2, p, dof, ex = chi2_contingency(observed)


print("p value")
print(p)


print("Chi Statistic")
print(chi2)



# --------------
# Code starts here
data.corr()
correlation = data.corr().abs()

# unstack and sort the values
us_correlation = correlation.unstack()
us_correlation = us_correlation.sort_values(ascending=False)

# max correlated features between 0.75 to 1
max_correlated = us_correlation[(us_correlation>0.75) & (us_correlation<1)]

# drop highly correlated features
data.drop(columns=['morningstar_rating','portfolio_stocks','category_12','sharpe_ratio_3y'],inplace=True)





# code ends here


# --------------
# Code starts here


#Setting up the subplots
fig, (ax_1, ax_2) = plt.subplots(1,2, figsize=(20,8))

#Plotting box plot
ax_1.boxplot(data['price_earning'])

#Setting the subplot axis title
ax_1.set(title='price_earning')

#Plotting box plot
ax_2.boxplot(data['net_annual_expenses_ratio'])

#Setting the subplot axis title
ax_2.set(title='net_annual_expenses_ratio')
#Code ends here   


# --------------
# import libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
from math import sqrt
# Code starts here
X= data.drop(['bonds_aaa'],1)
y= data['bonds_aaa']
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.3,random_state= 3)
lr= LinearRegression()
lr.fit(X_train,y_train)
y_pred= lr.predict(X_test)
rmse= sqrt(mean_squared_error(y_test, y_pred))
print(rmse)
# Code ends here


# --------------
# import libraries
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import Ridge,Lasso

# regularization parameters for grid search
ridge_lambdas = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60]
lasso_lambdas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1]

# Code starts here
ridge_model= Ridge()
ridge_grid= GridSearchCV(estimator=ridge_model, param_grid=dict(alpha=ridge_lambdas))
ridge_grid.fit(X_train,y_train)
y_pred= ridge_grid.predict(X_test)
ridge_rmse= sqrt(mean_squared_error(y_test,y_pred))
print(ridge_rmse)

lasso_model= Lasso()
lasso_grid= GridSearchCV(estimator=lasso_model, param_grid=dict(alpha=lasso_lambdas))
lasso_grid.fit(X_train,y_train)
y_pred= lasso_grid.predict(X_test)
lasso_rmse= sqrt(mean_squared_error(y_test,y_pred))
print(lasso_rmse)




# Code ends here


