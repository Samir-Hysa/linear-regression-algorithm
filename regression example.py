import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

data = pd.read_csv("student-mat.csv", sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
predict = "G3"
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

"""high_acc = 0
for _ in range(2000):

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
    Linear = linear_model.LinearRegression()
    Linear.fit(x_train, y_train)
    acc = Linear.score(x_test, y_test)
    print(acc)
    if acc > high_acc:
        high_acc = acc
        with open("student_model.pickle", "wb")as f:
            pickle.dump(Linear, f)"""

pickle_open = open("student_model.pickle", "rb")
Linear = pickle.load(pickle_open)

print("coefficents: ", Linear.coef_)
print("intercept: ", Linear.intercept_)

predictions = Linear.predict(x_test)
for i in range(len(predictions)):
    print(predictions[i], x_test[i], y_test[i])

p = "G1"
style.use("ggplot")
plt.scatter(data[p], data["G3"])
plt.xlabel(p)
plt.ylabel("Final Grade")
plt.show()

