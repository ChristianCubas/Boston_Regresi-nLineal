import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import datasets
from sklearn.model_selection import train_test_split

boston = datasets.load_boston()
print(boston)
x= boston.data[:,np.newaxis,5]
y=boston.target

plt.scatter(x,y,color="blue")
plt.legend()
plt.show()

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
lr = linear_model.LinearRegression()
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)

plt.scatter(x_test,y_test)
plt.plot(x_test,y_pred,linewidth=3,color="red")
plt.legend()
plt.show()

a = lr.coef_
b = lr.intercept_

print(f'La ecuacion es  y = {a}x {b}')
print(f'Porcentaje de efectividad: ',lr.score(x_train, y_train))