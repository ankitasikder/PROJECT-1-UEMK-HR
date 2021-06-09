import matplotlib
import matplotlib.pyplot as plt
#display mnist data digits
"""
ANKITA SIKDER

STUDENT OF BTECH, IN UEMK

CONTACT NO.: 8583939774

EMAIL ID: ankita.sikder14@gmail.com
"""
from sklearn.datasets import fetch_openml
mnist=fetch_openml('mnist_784')
print(mnist)
x,y=mnist['data'],mnist['target']
print(x.shape)
print(y.shape)
digit=x[36001]
digit=digit.reshape(28,28)
plt.imshow(digit,cmap=matplotlib.cm.binary,interpolation="nearest")
plt.show()
print(y[36000])"""
#Using Keras
from keras.datasets import mnist
(x_train, y_train),(x_test, y_test)=mnist.load_data()
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
plt.imshow(x_train[36000])
plt.show()

