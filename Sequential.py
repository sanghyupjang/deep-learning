import tensorflow as tf

# mnist 데이터셋
mnist = tf.keras.datasets.mnist

# 데이터 준비
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#데이터 전처리

x_train = x_train / x_train.max()
x_test = x_test / x_test.max()

#모델 생성

model = tf.kras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))