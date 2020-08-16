import os
import functools
import tensorflow as tf


def _get_label(file_path, class_names):
    sub_folder = tf.strings.split(file_path, os.path.sep)[-2]
    label = tf.strings.split(sub_folder, '_')[-1]
    one_hot = tf.cast(label == class_names, tf.int32)
    return tf.argmax(one_hot)


def _decode_img(img, size):
    img = tf.image.decode_jpeg(img, channels=3)
    return tf.image.resize(img, [size, size])


def _process_path(file_path, img_size, class_names):
    label = _get_label(file_path, class_names)
    img = _decode_img(tf.io.read_file(file_path), img_size)
    return img, label


def load_dataset(file_pattern, img_size, class_names, seed=0):
    list_ds = tf.data.Dataset.list_files(file_pattern, shuffle=True, seed=seed)

    process_path = functools.partial(_process_path, img_size=img_size, class_names=class_names)
    list_ds = list_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return list_ds


def split_dataset(dataset, split=0.2):
    count = tf.data.experimental.cardinality(dataset).numpy()

    val_size = int(count * 0.2)
    train_ds = dataset.skip(val_size)
    val_ds = dataset.take(val_size)
    return train_ds, val_ds


def batch_dataset(dataset, batch_size=32, buffer_size=1000):
    dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset
