import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
plt.rcParams ["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams ["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题
plt.rcParams['font.family'] = ['sans-serif']


# 数据加载，按照8:2的比例加载花卉数据
def data_load(data_dir, img_height, img_width, batch_size):
    # 从目录加载数据集并进行拆分
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode='categorical',
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode='categorical',
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    class_names = train_ds.class_names

    return train_ds, val_ds, class_names   # 训练数据集 ,验证数据集 ,数据集中的类别名称


def test(is_transfer=True):
    # 加载测试数据集和模型
    train_ds, val_ds, class_names = data_load("../data/flower_photos", 224, 224, 4)
    if is_transfer:
        model = tf.keras.models.load_model("models/mobilenet_flower.h5")
    else:
        model = tf.keras.models.load_model("models/cnn_flower.h5")
    model.summary()
    # 评估模型在测试数据集上的准确率
    loss, accuracy = model.evaluate(val_ds)
    print('Test accuracy :', accuracy)

#热力图
    if is_transfer:
        test_real_labels = []
        test_pre_labels = []
        for test_batch_images, test_batch_labels in train_ds:
            test_batch_labels = test_batch_labels.numpy()
            test_batch_pres = model.predict(test_batch_images)
            test_batch_labels_max = np.argmax(test_batch_labels, axis=1)
            test_batch_pres_max = np.argmax(test_batch_pres, axis=1)
            # 将推理对应的标签取出
            for i in test_batch_labels_max:
                test_real_labels.append(i)

            for i in test_batch_pres_max:
                test_pre_labels.append(i)

        class_names_length = len(class_names)
        heat_maps = np.zeros((class_names_length, class_names_length))
        for test_real_label, test_pre_label in zip(test_real_labels, test_pre_labels):
            heat_maps[test_real_label][test_pre_label] = heat_maps[test_real_label][test_pre_label] + 1

        print(heat_maps)
        heat_maps_sum = np.sum(heat_maps, axis=1).reshape(-1, 1)
        print()
        heat_maps_float = heat_maps / heat_maps_sum
        print(heat_maps_float)
        show_heatmaps(title="heatmap", x_labels=class_names, y_labels=class_names, harvest=heat_maps_float,
                      save_name="result/heatmap_mobilenet.png")
    else:
        # 对模型分开进行推理
        test_real_labels = []
        test_pre_labels = []
        for test_batch_images, test_batch_labels in train_ds:
            test_batch_labels = test_batch_labels.numpy()
            test_batch_pres = model.predict(test_batch_images)

            test_batch_labels_max = np.argmax(test_batch_labels, axis=1)
            test_batch_pres_max = np.argmax(test_batch_pres, axis=1)
            # 将推理对应的标签取出
            for i in test_batch_labels_max:
                test_real_labels.append(i)

            for i in test_batch_pres_max:
                test_pre_labels.append(i)

        class_names_length = len(class_names)
        heat_maps = np.zeros((class_names_length, class_names_length))
        for test_real_label, test_pre_label in zip(test_real_labels, test_pre_labels):
            heat_maps[test_real_label][test_pre_label] = heat_maps[test_real_label][test_pre_label] + 1

        print(heat_maps)
        heat_maps_sum = np.sum(heat_maps, axis=1).reshape(-1, 1)
        print()
        heat_maps_float = heat_maps / heat_maps_sum
        print(heat_maps_float)
        show_heatmaps(title="heatmap", x_labels=class_names, y_labels=class_names, harvest=heat_maps_float,
                      save_name="result/heatmap_cnn.png")


def show_heatmaps(title, x_labels, y_labels, harvest, save_name):
    # 这里是创建一个画布
    fig, ax = plt.subplots()
    im = ax.imshow(harvest, cmap="OrRd")
    # 这里是修改标签
    ax.set_xticks(np.arange(len(y_labels)))
    ax.set_yticks(np.arange(len(x_labels)))
    ax.set_xticklabels(y_labels)
    ax.set_yticklabels(x_labels)

    # 因为x轴的标签太长了，需要旋转一下，更加好看
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # 添加每个热力块的具体数值
    for i in range(len(x_labels)):
        for j in range(len(y_labels)):
            text = ax.text(j, i, round(harvest[i, j], 2),
                           ha="center", va="center", color="black")
    ax.set_xlabel("预测标签")
    ax.set_ylabel("实际标签")
    ax.set_title(title)
    fig.tight_layout()
    plt.colorbar(im)
    plt.savefig(save_name, dpi=100)
    plt.show()


if __name__ == '__main__':
    test(True)
    test(False)
