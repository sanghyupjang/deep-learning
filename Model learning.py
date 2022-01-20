import tensorflow as tf

# mnist 데이터셋
mnist = tf.keras.datasets.mnist

# 데이터 준비
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#데이터 전처리

x_train = x_train / x_train.max()
x_test = x_test / x_test.max()

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

# 모델 요약

print(model.summary())

""

summary(
    line_length=None, positions=None, print_fn=None
)

""

# 모델 컴파일

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

""
#학습을 위한 모델 구성
compile(
    optimize='rmsprop', loss=None, metrics=None, loss_weights=None,
    weighted_metrics=None, run_eagerly=None, step_per_excution=None, **kwargs
)
""

#모델 학습
model.fit(x_train, y_train, validation_data=*(x_test, y_test), epochs=10)

""
#고정된 수의 에폭(데이터셋의 반복)에 대한 모델 교육
fit(
    x=None, y=None, batch_size=None, epochs=1, verbose='auto',
    callbacks=None, validation_split=0.0, validation_date=None, shuffle=True,
    class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None,
    validation_steps=None, validation_batch_size=None, validation_freq=1,
    max_queue_size=10, workers=1, use_multiprocessing=False
)
""
