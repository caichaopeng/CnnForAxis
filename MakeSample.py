# coding=utf8
# __author__='caichaopeng'

import tensorflow as tf
import preprocess
import os
from PIL import Image



def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def create_record(img_files_path, classes, TFRecord_file_path, image_size):

    writer = tf.python_io.TFRecordWriter(TFRecord_file_path)  # 要生成的文件

    for index, name in enumerate(classes):
        class_path = img_files_path + '/' + name
        for img_name in os.listdir(class_path):
            img_path = class_path + '/' + img_name  # 每一个图片的地址
            img = Image.open(img_path)
            img = img.resize((image_size, image_size))
            img_raw = img.tobytes()  # 将图片转化为二进制格式
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': _int64_feature(index),
                'img_raw': _bytes_feature(img_raw),
            }))  # example对象对label和image数据进行封装
            writer.write(example.SerializeToString())  # 序列化为字符串
    writer.close()


def read_and_decode(TFRecord_file_path, image_size):
    filename_queue = tf.train.string_input_producer([TFRecord_file_path])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'img_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })
    label = features['label']
    img = features['img_raw']
    img = tf.decode_raw(img, tf.uint8)
    label = tf.cast(label, tf.int32)
    '''图像预处理'''
    img = tf.reshape(img, [image_size, image_size, 1])
    #img = preprocess.preprocess_for_train(img, image_size, image_size, None)
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    return img, label

