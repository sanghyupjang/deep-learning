import tensorflow as tf

# mnist 데이터셋
mnist = tf.keras.datasets.mnist

# 데이터 준비
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#데이터 전처리

x_train = x_train / x_train.max()
x_test = x_test / x_test.max()