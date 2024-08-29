import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape, Flatten
from tensorflow.keras.optimizers import Adam
# 加载MNIST数据集
(X_train, _), (_, _) = mnist.load_data()
# 归一化像素值到[-1, 1]
X_train = (X_train.astype(np.float32) - 127.5) / 127.5
X_train = np.expand_dims(X_train, axis=3)
# 定义生成器模型
generator = Sequential()
generator.add(Dense(256, input_dim=100))
generator.add(LeakyReLU(0.2))
generator.add(BatchNormalization())
generator.add(Dense(512))
generator.add(LeakyReLU(0.2))
generator.add(BatchNormalization())
generator.add(Dense(784, activation='tanh'))
generator.add(Reshape((28, 28, 1)))
# 定义判别器模型
discriminator = Sequential()
discriminator.add(Flatten(input_shape=(28, 28, 1)))
discriminator.add(Dense(512))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dense(256))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dense(1, activation='sigmoid'))
# 编译判别器模型
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5), metrics=['accuracy'])
# 设置判别器不可训练
discriminator.trainable = False
# 定义GAN模型
gan = Sequential()
gan.add(generator)
gan.add(discriminator)
# 编译GAN模型
gan.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))
# 定义训练函数
def train_gan(epochs, batch_size, sample_interval):
    # 计算训练的批次数
    num_batches = X_train.shape[0] // batch_size
    
    for epoch in range(epochs):
        for batch in range(num_batches):
            # 随机选择真实图像
            real_images = X_train[np.random.randint(0, X_train.shape[0], batch_size)]
            
            # 生成噪声作为输入
            noise = np.random.normal(0, 1, (batch_size, 100))
            
            # 使用生成器生成假图像
            fake_images = generator.predict(noise)
            
            # 创建一个包含真实图像和假图像的训练集
            X = np.concatenate((real_images, fake_images))
            
            # 创建一个包含真实标签和假标签的目标值
            y = np.concatenate((np.ones((batch_size, 1)), np.zeros((batch_size, 1))))
            
            # 训练判别器
            discriminator_loss = discriminator.train_on_batch(X, y)
            
            # 重新生成噪声作为输入
            noise = np.random.normal(0, 1, (batch_size, 100))
            
            # 创建目标值为真实标签的训练集
            y = np.ones((batch_size, 1))
            
            # 训练生成器
            generator_loss = gan.train_on_batch(noise, y)
            
            # 每隔一段时间打印损失信息
            if batch % sample_interval == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch}/{num_batches}, D loss: {discriminator_loss[0]}, G loss: {generator_loss}")
        
        # 每个epoch结束后生成并保存一张生成的图像
        if epoch % sample_interval == 0:
            generate_and_save_images(generator, epoch)
    
# 定义生成和保存图像函数
def generate_and_save_images(model, epoch):
    # 生成噪声作为输入
    noise = np.random.normal(0, 1, (100, 100))
    
    # 使用生成器生成图像
    generated_images = generator.predict(noise)
    
    # 可视化生成的图像
    fig, axs = plt.subplots(10, 10, figsize=(10, 10), sharex=True, sharey=True)
    cnt = 0
    for i in range(10):
        for j in range(10):
            axs[i, j].imshow(generated_images[cnt, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
    plt.savefig(f"generated_images_epoch_{epoch}.png")
    plt.close()
# 训练GAN模型
train_gan(epochs=200, batch_size=128, sample_interval=20)