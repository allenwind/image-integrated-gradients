import warnings
warnings.filterwarnings("ignore")
import glob
import numpy as np
import tensorflow as tf

_CD = "/home/zhiwen/workspace/dataset/fastai-datasets-cats-vs-dogs/"
def load_cats_vs_dogs(file=_CD, batch_size=32, image_shape=[256, 256]):
    train_dogs_pattern = file + "train/dogs/*.jpg"
    train_cats_pattern = file + "train/cats/*.jpg"
    valid_dogs_pattern = file + "valid/dogs/*.jpg"
    valid_cats_pattern = file + "valid/cats/*.jpg"

    def load_files(dogs_dir, cats_dir):
        dog_files = glob.glob(dogs_dir) # or use tf.data.Dataset.list_files
        cat_files = glob.glob(cats_dir)
        files = np.array(dog_files + cat_files)
        labels = np.array([0] * len(dog_files) + [1] * len(cat_files))
        np.random.seed(8899)
        np.random.shuffle(files)
        np.random.seed(8899) # 保证标签对齐
        np.random.shuffle(labels)
        labels = tf.keras.utils.to_categorical(labels)
        return files, labels

    def fn(filename, label):
        image_string = tf.io.read_file(filename)            # 读取原始文件
        image_decoded = tf.image.decode_jpeg(image_string)  # 解码JPEG图片
        image_resized = tf.image.resize(image_decoded, image_shape) / 255.0
        return image_resized, label

    files, labels = load_files(train_dogs_pattern, train_cats_pattern)
    train_dataset = tf.data.Dataset.from_tensor_slices((files, labels)) \
                             .map(fn, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
                             .shuffle(buffer_size=256) \
                             .batch(batch_size) \
                             .prefetch(tf.data.experimental.AUTOTUNE)

    # 验证集
    files, labels = load_files(valid_dogs_pattern, valid_cats_pattern)
    valid_dataset = tf.data.Dataset.from_tensor_slices((files, labels)) \
                                   .map(fn, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
                                   .batch(batch_size)
    return train_dataset, valid_dataset
