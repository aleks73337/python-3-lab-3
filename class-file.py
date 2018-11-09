import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import  PolynomialFeatures
import matplotlib.pyplot as plt

n = 20
x = np.zeros((n,))
y = np.zeros((n, ))
x = np.random.random(size=(n, )) * 6 - 3
x = np.sort(x, axis=0)

y = 3-3.5*x - 4.2*x**2+ 3.67*x**3 - 2.11*x**4 + 1.8*x**5 + np.random.normal(0,50, n);
x=x.reshape(-1, 1)
plt.plot(x, y, 'o', markersize=3)
#plt.show()

deg = 15
pol = PolynomialFeatures(degree=deg)
x_pol = pol.fit_transform(X=x)

model = LinearRegression()
model.fit(x_pol,y)

regression_x = np.linspace(x.min(), x.max(), 1001)
regression_x = regression_x.reshape(-1, 1)
regression_x_pol = pol.transform(X=regression_x)
regression_y = model.predict(regression_x_pol)

plt.plot(regression_x, regression_y)
print(model.coef_, model.intercept_)
plt.ylim(-500, 200 )
plt.show()