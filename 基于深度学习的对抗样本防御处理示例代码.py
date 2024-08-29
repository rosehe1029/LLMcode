import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
# 加载MNIST数据集
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
# 归一化像素值到[0, 1]
X_train, X_test = X_train / 255.0, X_test / 255.0
# 定义模型
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
# 编译模型
model.compile(optimizer=Adam(),
              loss=SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
# 训练模型
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
# 对抗样本生成函数
def generate_adversarial_examples(model, X, y, epsilon):
    # 计算损失函数关于输入的梯度
    with tf.GradientTape() as tape:
        tape.watch(X)
        loss = SparseCategoricalCrossentropy(from_logits=True)(y, model(X))
    gradients = tape.gradient(loss, X)
    
    # 计算对抗样本
    X_adv = X + epsilon * tf.sign(gradients)
    X_adv = tf.clip_by_value(X_adv, 0, 1)
    
    return X_adv
# 生成对抗样本
X_test_adv = generate_adversarial_examples(model, X_test, y_test, 0.1)
# 评估模型在对抗样本上的性能
model.evaluate(X_test_adv, y_test)
# 重新训练模型，使用对抗样本进行训练
model.fit(X_train, y_train, epochs=5, validation_data=(X_test_adv, y_test))
# 再次评估模型在对抗样本上的性能
model.evaluate(X_test_adv, y_test)