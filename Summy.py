import tensorflow as tf

# mnist 데이터셋
mnist = tf.keras.datasets.mnist

# 데이터 준비
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#데이터 전처리

x_train = x_train / x_train.max()
x_test = x_test / x_test.max()

# 모델 요약

print(model.summary())

""

summary(
    line_length=None, positions=None, print_fn=None
)

""