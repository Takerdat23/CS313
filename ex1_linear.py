import numpy as np 
from sklearn.linear_model import LinearRegression 
import matplotlib.pyplot as plt
np.random.seed(40)

n = 40 

beta_0 =1 
beta_1 = 2 



x= np.random.normal(loc = 0  , scale = 1 , size = n)
eps = np.random.normal(loc= 0 , scale = 1 ,size = n )

y = beta_0 + beta_1 * x + eps


x= x.reshape((n, 1))

y = y.reshape((n, 1)) 




# plt.scatter(x, y)
# plt.show()

reg= LinearRegression()

reg.fit(x, y.flatten())

print("intercept: ", reg.intercept_ , "slop: ", reg.coef_[0])


# incoperate the intercept and coef 

reg1= LinearRegression(fit_intercept= False)

# stack thêm cột 1 vào trước để intercept không chú ý tới nó , mà slope sẽ có 2 phần tử

# X =  np.hstack((np.ones((n, 1)), x))

# reg1.fit(X, y.flatten())


# print("intercept: ", reg1.intercept_ , "slop: ", reg1.coef_)


#exercise 2 : compute beta_0 , beta_1 without sklearn 

def mean_squared_error(y_true, y_pred):
    n = len(y_true)
    squared_errors = (y_true - y_pred) ** 2
    mse = np.sum(squared_errors) / n
    return mse


# y_pred = reg.predict(x)

# Compute MSE
# mse = mean_squared_error(y.flatten(), y_pred)
# print("Mean Squared Error:", mse)


y_bar = np.mean(y)
x_bar = np.mean(x)


numerator = 0 
deniminator = 0 

for i in range(n): 
    numerator = numerator + x[i][0] * (y_bar - y[i][0])
    deniminator = deniminator + x[i][0] * (x_bar - x[i][0])

estimated_beta_1 = numerator/ deniminator
estimated_beta_0 = y_bar - estimated_beta_1 * x_bar 


print(estimated_beta_0 , estimated_beta_1)



# beta_0 = 1
# beta_1 = 2
# learning_rate = 0.01
# epochs = 10

# # Gradient descent
# for epoch in range(epochs):
#     # Calculate predictions
#     predictions = beta_0 + beta_1 * x.flatten()
    
#     # Calculate errors
#     mse = mean_squared_error(y.flatten(),predictions)

#     print(mse)
    
#     # Compute gradients
#     gradient_beta_0 = 2 * np.mean(mse)
    
#     # Update beta_0
#     beta_0 -= learning_rate * gradient_beta_0
    
#     # Print progress
#     if epoch % 100 == 0:
#         print(f'Epoch {epoch}: beta_0 = {beta_0}')