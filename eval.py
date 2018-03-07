#coding=utf8
#__author__='caichaopeng'

import time
import tensorflow as tf
import makeSample
import inference
import train

#每10s加载一次最新模型，并在测试数据上测试最新模型的正确率
EVAL_INTERVAL_SECS = 10
img_files_path = './evalFace'
classes = ['caichaopeng', 'jiangenyong','xiaweihai','huangcheng']  # 人为 设定4类
TFRecord_file_path = './tfrecords/face_eval.tfrecords'
image_size = 299
eval_num = 80


def evaluate(img_files_path, classes, TFRecord_file_path, image_size,eval_num):
    with tf.Graph().as_default() as g:
        makeSample.create_record(img_files_path, classes, TFRecord_file_path, image_size)
        filename_queue = tf.train.string_input_producer([TFRecord_file_path])
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read_up_to(filename_queue, eval_num)
        features = tf.parse_example(
            serialized_example,
            features={
                'img_raw': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64),
            })
        label = features['label']
        label = tf.cast(label, tf.int32)
        img = features['img_raw']
        img = tf.decode_raw(img, tf.uint8)
        img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
        img = tf.reshape(img, [eval_num, image_size, image_size, 1])

        # 直接通过调用封装好的函数来计算前向传播的结果。
        # 因为测试时不关注正则损失的值，所以这里用于计算正则化损失的函数被设置为None。
        y = inference.inference(img, False, None)
        # 使用前向传播的结果计算正确率。
        # 如果需要对未知的样例进行分类，那么使用tf.argmax(y, 1)就可以得到输入样例的预测类别了。
        correct_prediction = tf.equal(tf.cast(tf.argmax(y, 1),tf.int32),label)
        a = tf.argmax(y, 1)
        b = label
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # 通过变量重命名的方式来加载模型，这样在前向传播的过程中就不需要调用求滑动平均的函数来获取平局值了。
        # 这样就可以完全共用mnist_inference.py中定义的前向传播过程
        variable_averages = tf.train.ExponentialMovingAverage(train.MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)

        #每隔EVAL_INTERVAL_SECS秒调用一次计算正确率的过程以检测训练过程中正确率的变化
        while True:
            with tf.Session() as sess:
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                # tf.train.get_checkpoint_state函数会通过checkpoint文件自动找到目录中最新模型的文件名
                ckpt = tf.train.get_checkpoint_state(train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    # 加载模型
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    # 通过文件名得到模型保存时迭代的轮数
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy)
                    print(sess.run(a),sess.run(b))
                    print("After %s training step(s), validation accuracy = %f" % (global_step, accuracy_score))
                else:
                    print("No checkpoint file found")
                    return
                coord.request_stop()
                coord.join(threads)

            time.sleep(EVAL_INTERVAL_SECS)


def main(argv = None):
    evaluate(img_files_path, classes, TFRecord_file_path, image_size,eval_num)

if __name__ == '__main__':
    tf.app.run()

