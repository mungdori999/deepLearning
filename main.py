import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier

cancer = load_breast_cancer()
x = cancer.data
y = cancer.target
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2, random_state=42)

sgd = SGDClassifier(loss='hinge', max_iter=100, tol=1e-3, random_state=42)

sgd.fit(x_train, y_train)
score = sgd.score(x_test, y_test)
predict = sgd.predict(x_test[0:10])
print(score)
print(x_test[0:10])
print(predict)