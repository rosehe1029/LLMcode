import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.models import Model

# 设置路径
BASE_DIR = '/path/to/your/dataset'
TRAIN_IMAGES_DIR = os.path.join(BASE_DIR, 'train_images')
TRAIN_MASKS_DIR = os.path.join(BASE_DIR, 'train_masks')
VALIDATION_IMAGES_DIR = os.path.join(BASE_DIR, 'validation_images')
VALIDATION_MASKS_DIR = os.path.join(BASE_DIR, 'validation_masks')

# 数据集参数
IMAGE_SIZE = (512, 512)
BATCH_SIZE = 4
NUM_CLASSES = 21  # 包括背景类
EPOCHS = 100

# 构建 DeepLab v3+ 模型
def build_deeplabv3_plus(input_shape, num_classes):
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet', alpha=0.5)

    x = base_model.output
    x = Conv2D(256, (1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = UpSampling2D(size=(4, 4))(x)

    # 附加分支
    additional_branch = Conv2D(48, (1, 1), padding='same', use_bias=False)(base_model.get_layer('block_1_expand_relu').output)
    additional_branch = BatchNormalization()(additional_branch)

    # 合并
    x = Concatenate()([x, additional_branch])
    x = Conv2D(256, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    # 输出层
    x = Conv2D(num_classes, (1, 1), padding='same')(x)
    x = UpSampling2D(size=(4, 4), interpolation='bilinear')(x)

    model = Model(inputs=base_model.input, outputs=x)
    return model

# 创建模型
model = build_deeplabv3_plus(input_shape=(*IMAGE_SIZE, 3), num_classes=NUM_CLASSES)

# 编译模型
model.compile(optimizer=Adam(learning_rate=0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# 数据生成器
def data_generator(image_dir, mask_dir, batch_size, image_size):
    image_filenames = os.listdir(image_dir)
    while True:
        for i in range(0, len(image_filenames), batch_size):
            images = []
            masks = []
            for filename in image_filenames[i:i + batch_size]:
                img = tf.keras.preprocessing.image.load_img(os.path.join(image_dir, filename), target_size=image_size)
                img = tf.keras.preprocessing.image.img_to_array(img)
                img = img / 255.0
                images.append(img)

                mask_filename = os.path.splitext(filename)[0] + '_mask.png'  # 假设掩码文件名与图像相同，只是后缀不同
                mask = tf.keras.preprocessing.image.load_img(os.path.join(mask_dir, mask_filename), target_size=image_size, color_mode="grayscale")
                mask = tf.keras.preprocessing.image.img_to_array(mask)
                mask = mask / 255.0  # 假设类别标签为0-255
                masks.append(mask)

            yield (np.array(images), np.expand_dims(np.array(masks), axis=-1))

# 训练数据生成器
train_data_gen = data_generator(TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR, BATCH_SIZE, IMAGE_SIZE)

# 验证数据生成器
validation_data_gen = data_generator(VALIDATION_IMAGES_DIR, VALIDATION_MASKS_DIR, BATCH_SIZE, IMAGE_SIZE)

# 回调函数
checkpoint = ModelCheckpoint(filepath='deeplabv3plus.h5', monitor='val_accuracy', save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
tensorboard = TensorBoard(log_dir='./logs')

# 训练模型
history = model.fit(
    train_data_gen,
    steps_per_epoch=len(os.listdir(TRAIN_IMAGES_DIR)) // BATCH_SIZE,
    validation_data=validation_data_gen,
    validation_steps=len(os.listdir(VALIDATION_IMAGES_DIR)) // BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stopping, tensorboard]
)