import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# create sample data
x = np.linspace(0, 1, 100).reshape(-1, 1)
y = [i*np.random.uniform(0.5, 0.7)] for i in np.linspace(0, 1, 100)]
y = np.array(y)

model = LinearRegression()
model.fit(x, y)
y_pred = model.predict(x)

plt.plot(x, y)
plt.plot(x, y_pred)
plt.show()

with open('model.pickle', 'wb') as file:
    pickle.dump(model, file)