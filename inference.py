#coding=utf8
#__author__='caichaopeng'

#coding=utf8
#__author__='caichaopeng'

import tensorflow as tf

#配置神经网络参数
INPUT_NODE = 299
OUTPUT_NODE = 4

IMAGE_SIZE = 299
NUM_CHANNELS = 1
NUM_LABELS = 4

#第一层卷积层的尺寸和深度
CONV1_DEEP = 32
CONV1_SIZE = 5
#第二层卷积层的尺寸和深度
CONV2_DEEP = 64
CONV2_SIZE = 5

#全连接层的节点个数
FC_SIZE = 512

#定义卷积神经网络的前向传播过程。这里添加了一个新的参数train，用于区别训练过程和测试过程
#在这个程序中将用到dropout方法，dropout可以进一步提升模型的可靠性，防止过拟合，dropout过程只在训练时使用
def inference(input_tensor,train,regularizer):
    #声明第一层卷积层的变量并实现前向传播过程。
    #通过使用不同的命名空间来隔离不同层的变量，可以让每一层中的变量命名只需要考虑在当前层中的作用，而不担心重命名问题
    #和标准LeNet-5模型不大一样，这里定义的卷积层输入为28x28x1的原始MNIST图片像素。因为卷积层用了全0填充，
    #所以输出为28x28x32的矩阵
    with tf.variable_scope('layer1-conv1'):
        conv1_weight = tf.get_variable('weight',[CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_DEEP],
                                       initializer = tf.truncated_normal_initializer(stddev = 0.1))
        conv1_biases = tf.get_variable('bias',[CONV1_DEEP],initializer = tf.constant_initializer(0.0))

        #使用边长为5，深度为32的过滤器，过滤器移动步长为1，且为全0填充
        conv1 = tf.nn.conv2d(input_tensor,conv1_weight,strides = [1,1,1,1],padding = 'SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))

    #实现第二层池化层的前向传播过程。这里选用最大池化层，池化层过滤器的边长为2，
    #使用全0填充且移动步长为2.这一层的输入是上一层的输出，也就是28x28x32的矩阵。
    #输出14x14x32的矩阵
    with tf.variable_scope('layer2-poll1'):
        pool1 = tf.nn.max_pool(relu1,ksize = [1,2,2,1],strides = [1,2,2,1],padding = 'SAME')
    #声明第三层卷积层的变量实现前向传播过程，这一层的输入为14x14x32的矩阵，输出为14x14x64的矩阵
    with tf.variable_scope('layer3-conv2'):
        conv2_weight = tf.get_variable('weight',[CONV2_SIZE,CONV2_SIZE,CONV1_DEEP,CONV2_DEEP],
                                       initializer = tf.truncated_normal_initializer(stddev = 0.1))
        conv2_biases = tf.get_variable('bias',[CONV2_DEEP],
                                       initializer = tf.constant_initializer(0.0))
        #使用边长为5，深度为64的过滤器，过滤器的移动步长为1，且全0连接
        conv2 = tf.nn.conv2d(pool1,conv2_weight,strides = [1,1,1,1],padding = 'SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2,conv2_biases))

    #实现第四层池化层的前向传播过程，这一层和第二层的结构是一样的。这一层的输入为14x14x16的矩阵，输出为
    #7x7x64的矩阵
    with tf.variable_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(relu2,ksize = [1,2,2,1],strides = [1,2,2,1],padding = 'SAME')

    #将第四层池化层的输出转化为第五层全连接层的输入格式。第四层的输出为7x7x64的矩阵，
    #然而第五层全连接层需要输入格式为向量，所以在这里需要将这个7x7x64的矩阵拉直成一个向量。
    #pool2.get_shape函数可以得到第四层输出矩阵的维度而不需要手工计算，注意因为每一层神经网络
    #的输入输出都为一个batch的矩阵，所以这里得到的维度也包含了batch中数据的个数。
    pool_shape=pool2.get_shape().as_list()
    #计算将矩阵拉直成向量之后的长度，这个长度就是矩阵长宽及深度的乘积
    #注意这里pool_shape[0]为一个batch中数据的个数
    nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]

    #通过tf.reshape函数将第四层的输出变成一个batch的向量
    reshaped = tf.reshape(pool2,[pool_shape[0],nodes])

    #声明第五层全连接层的变量并实现前向传播过程。这一层的输入是拉直后的一组向量，向量的长度为3136，
    #输出是一组长度为512的向量。引入dropout,dropout在训练时会随机将部分节点的输出改为0.drop可以避免过拟合的
    #问题，从而使模型在测试数据上的效果更好。
    #dropout一般只在全连接层而不是卷积层或者池化层使用
    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable('weight',[nodes,FC_SIZE],
                                      initializer = tf.truncated_normal_initializer(stddev =0.1))
        #只有全连接层的权重需要加入正则化
        if regularizer != None:
            tf.add_to_collection('losses',regularizer(fc1_weights))
        fc1_biases = tf.get_variable('bias',[FC_SIZE],initializer = tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped,fc1_weights)+fc1_biases)
        if train:
            fc1 = tf.nn.dropout(fc1,0.5)

    #声明第六层全连接的变量并实现前向传播过程。这一层的输入为一组长度为512的向量。
    #输出为一组长度为10的向量。这一层输出通过softmax之后就得到了最后的分类结果
    with tf.variable_scope('layer6-fc2'):
        fc2_weights = tf.get_variable('weight',[FC_SIZE,NUM_LABELS],
                                      initializer = tf.truncated_normal_initializer(stddev = 0.1))
        if regularizer != None:
            tf.add_to_collection('losses',regularizer(fc2_weights))
        fc2_biases = tf.get_variable('bias',[NUM_LABELS],initializer = tf.constant_initializer(0.1))
        logit = tf.matmul(fc1,fc2_weights)+fc2_biases

    #返回第六层的输出
    return logit