import numpy as np
import matplotlib.pyplot as plt



def prediction(theta0, theta1, x):
    return theta0 + (theta1 * x)

def error_func(theta0, theta1, x, y):
    return prediction(theta0, theta1, x) - y

def renew_theta(theta0, theta1, x, y, alpha, n):
    for _ in range(n):
        error = error_func(theta0, theta1, x, y)
        
        theta0 = theta0 - (alpha * error.mean())
        theta1 = theta1 - (alpha * error.mean()) * x
    return theta0, theta1

def renew_theta(theta0, theta1, x, y, alpha, n):
    for i in range(n):
        error = error_func(theta0, theta1, x, y)
        
        theta0 = theta0 - alpha * (error.mean())
        theta1 = theta1 - (alpha * (error * x).mean())
        
        # 처음의 가설함수와 세타를 100번 갱신한 후의 가설함수를 출력
        
        if(i % 99 == 0):
            plt.scatter(x,y)
            plt.plot(x, prediction(theta0, theta1, x), color = 'red')
            plt.show()
    
    return theta0, theta1


x = np.array([0.4, 1.4, 2, 2.5, 3.1, 3.5, 3.6, 4.2, 4.6, 5.5, 6.7, 7.3])
y = np.array([0.2, 0.3, 0.5, 1.1, 1.45, 2.1, 2.3, 2.65, 2.8, 2.9, 3.1, 3.5])

theta0 = 2
theta1 = 0
alpha = 0.1

renew_theta(theta0, theta1, x, y, alpha, 100)