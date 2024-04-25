import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression


def generate_data(n ,p , true_beta): 
    X = np.random.normal(loc= 0 , scale = 1 , size= (n, p))
    true_beta = np.reshape(true_beta , (p, 1) )

    noise = np.random.normal(loc = 0 , scale = 1 , size= (n,1))
    y = np.dot(X, true_beta ) + noise
    return X, y 

n = 40
p =  5 
true_beta = [-1.0 , 0.0, 1.0, 1.5, 2.0]
X,y = generate_data(n, p , true_beta)

# Exercies 3: regression with sklearn 

reg  = LinearRegression()

reg.fit(X,y)

print(reg.coef_)

# Exercise 4:Regression without sklearn 
def linear_regression(X, y):
        X = np.column_stack((np.ones(len(X)), X))
        theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
                            
        return theta.T


print(linear_regression(X,y )) 