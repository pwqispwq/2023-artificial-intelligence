import tensorflow as tf
import matplotlib.pyplot as plt
plt.rcParams ["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams ["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题

# 数据集下载地址：https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
# 数据加载，按照8:2的比例加载花卉数据
def data_load(data_dir, img_height, img_width, batch_size):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode='categorical',
        validation_split=0.2,  # 数据加载，按照8:2的比例加载花卉数据
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    # 加载验证数据集
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode='categorical',
        validation_split=0.2,  # 按照8:2的比例加载花卉数据
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    class_names = train_ds.class_names
    return train_ds, val_ds, class_names  # 训练数据集 ,验证数据集 ,数据集中的类别名称


# 模型加载，指定图片处理的大小和是否进行迁移学习
def model_load(IMG_SHAPE=(224, 224, 3), is_transfer=False):
    if is_transfer:        # 微调的过程中不需要进行归一化的处理
        # 使用迁移学习的 MobileNetV2 模型
        base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                          include_top=False,
                                          weights='imagenet')
        base_model.trainable = False
        model = tf.keras.models.Sequential([
            tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset=-1, input_shape=IMG_SHAPE),
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(5, activation='softmax')
        ])

    else:
# 使用自定义的 CNN 模型
        model = tf.keras.models.Sequential([
            tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=IMG_SHAPE),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            # Add another convolution
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(5, activation='softmax')
        ])

    model.summary()
    # 模型训练
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# 展示训练过程的曲线
def show_loss_acc(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='训练准确率')
    plt.plot(val_acc, label='验证准确率')
    plt.legend(loc='lower right')
    plt.ylabel('准确率')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('训练和验证准确率')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='训练损失')
    plt.plot(val_loss, label='验证损失')
    plt.legend(loc='upper right')
    plt.ylabel('交叉熵')
    plt.ylim([0, 1.0])
    plt.title('训练和验证损失')
    plt.xlabel('轮次')
    plt.show()


def train(epochs, is_transfer=False):
    train_ds, val_ds, class_names = data_load("../data/flower_photos", 224, 224, 4)
    model = model_load(is_transfer=is_transfer)
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    if is_transfer:
        # model.evaluate(val_ds)
        model.save("models/mobilenet_flower.h5")
    else:
        model.save("models/cnn_flower.h5")
    show_loss_acc(history)


if __name__ == '__main__':
    # 使用迁移学习的 MobileNetV2 进行训练
    #  train(epochs=10, is_transfer=True)
    # 使用自定义的 CNN 进行训练
    train(epochs=5, is_transfer=False)


