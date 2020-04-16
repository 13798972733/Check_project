#- * -coding:utf8 -
###########################################################################
#                        人脸识别算法                                       #
###########################################################################
from PIL import Image, ImageDraw, ImageFont
import numpy as numpy
import cv2
import numpy as np
import tensorflow as tf
import os
from .facenet.understand_facenet.align import detect_face
from .settings import FACE_DISTANCE_THRESHOLD
tf.reset_default_graph()
class FCNN(object):
    def __init__(self):
        #tf.reset_default_graph()
        self.reslut = 0
        self.drop_rate = 0.7                         # 丢弃率
        self.learning_rate = 0.0001                 # 学习率
        self.output_size = 32                     # 模型输出节点数
        self.batch_size = 100                   # 每次取n张图片                    # 模型输入图片大小
        self.model_save = "/home/itc/ai_resource/class_classifier"                  # 模型保存路径包括文件名
        self.data = []
        self.data1 = []# 图像矩阵和标签
        #self.create_folder(self.model_save)                 # 创建文件夹
        self.graph = tf.Graph()
        self.distance_threshold = FACE_DISTANCE_THRESHOLD
        self.all_model_map = {}
        self.minsize = 20
        self.threshold = [0.6, 0.7, 0.7]
        self.factor = 0.709
        graph = tf.Graph()
        self.class_student_map = {}
        self.CLASS_RESOURSE_PATH = "/home/itc/ai_resource/class_data/15"
        self.CLASS_RESOURSE_PATH_ = "/home/itc/ai_resource/class_data/15"
        self.CLASS_RESOURSE_OUTPATH = "/home/itc/ai_resource/class_data/11"
        self.CLASS_RESOURSE_READPATH = "/home/itc/ai_resource/class_data/00.png"
        self.size = 128
        self.path = os.path.join(self.model_save, 'model.ckpt')
        self.prediction = []
        self.train_step = 1
    def create_folder(self, model_save):
        path = model_save[:self.find_last(model_save, "/")]
        if not os.path.exists(path):
            os.makedirs(path)
        pass
    # 使用一个线程源源不断的将硬盘中的图片数据读入到一个内存队列中，
    # 另一个线程负责计算任务，所需数据直接从内存队列中获取不经过CPU
    def get_Batch(self, data, label, batch_size):
        #处理的是来源tensor的数据，创建tf的文件名队列形成引索
        input_queue = tf.train.slice_input_producer([data, label], num_epochs=self.train_step, shuffle=True, capacity=32)
        #调用tf.train.start_queue_runners 函数来启动执行文件名队列填充的线程
        x_batch, y_batch = tf.train.shuffle_batch(input_queue, batch_size=batch_size, num_threads=1, capacity=64, min_after_dequeue=1)
        return x_batch, y_batch

    def find_last(self, strings, sting):
        last_position = -1
        while True:
            position = strings.find(sting, last_position + 1)
            if position == -1:
                return last_position
            last_position = position
        pass

    """设置数据集"""

    def set_data(self):
        #读取文件目录下目录为列表
        for class_id in os.listdir(self.CLASS_RESOURSE_PATH):
            #路径拼接加重新读取次级目录
            for stu_id in os.listdir(os.path.join(self.CLASS_RESOURSE_PATH, class_id)):
                #迭代拼接出路径
                path1 = os.path.join(self.CLASS_RESOURSE_PATH, class_id)
                #按路径打开图片
                img = Image.open(os.path.join(path1, stu_id))
                #图片转入并转成灰度数组
                region = np.asarray(img)
                region = cv2.resize(region, (128, 128), interpolation=cv2.INTER_AREA)
                region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                region = self.to_rgb1(region)
                region = np.asarray(region)
                #标签转为矩阵
                p = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0]
                for q in range(self.output_size):
                    if q == int(int(class_id)):
                        p[q] = 1
                p = np.asarray(p)
                #打入成为全局的数据列表
                self.data1.append(p)
                self.data.append(region)
        print(len(self.data))
        print("数据加载成功...")
        pass
    #测试数据的加载
    def set_test_data(self):
        for class_id in os.listdir(self.CLASS_RESOURSE_PATH):
            for stu_id in os.listdir(os.path.join(self.CLASS_RESOURSE_PATH, class_id)):
                path1 = os.path.join(self.CLASS_RESOURSE_PATH, class_id)
                img = Image.open(os.path.join(path1, stu_id))
                region = np.asarray(img)
                region = cv2.resize(region, (128, 128), interpolation=cv2.INTER_AREA)
                region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                region = self.to_rgb1(region)
                region = np.asarray(region)
                p = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0]
                for q in range(self.output_size):
                    if q == int(int(class_id)):
                        p[q] = 1
                p = np.asarray(p)
                self.data1.append(p)
                self.data.append(region)
        print(len(self.data))
        print("数据加载成功...")
        pass
    #将图片读入为可用的数组方法和上面差不多
    def red_data(self, img):
        region = np.asarray(img)
        region = cv2.resize(region, (128, 128), interpolation=cv2.INTER_AREA)
        region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        region = self.to_rgb1(region)
        region = np.asarray(region)
        self.data.append(region)
    #转灰度为RGB
    def to_rgb(self, img):
        w, h = img.shape
        ret = np.empty((w, h, 3), dtype=np.uint8)
        ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
        return ret

    def to_rgb1(self, img):
        w, h = img.shape
        pet = []
        for i in range(w):
            ret = (img[i] * 255).astype(np.float)
            pet.append(ret)
        return pet
    #将图片返回为回归框
    def load_and_align_data(self, image, image_size, margin, pnet, rnet, onet):
        # img_size = np.asarray(image.shape)[0:2]
        image = np.asarray(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = self.to_rgb(image)
        bounding_boxes, _ = detect_face.detect_face(image, self.minsize, pnet, rnet, onet,
                                                        self.threshold, self.factor)
        return bounding_boxes

    """按长宽中的最大值获得上下左右需要填充的大小"""

    def get_padding_size(self, image):
        height, width = image.shape[:2]  # 同时给长宽赋不同值
        top, bottom, left, right = (0, 0, 0, 0)  # 同时给上下左右赋不同值
        longest = max(height, width)
        if width < longest:
            tmp = longest - width
            left = tmp // 2
            right = tmp - left
        if height < longest:
            tmp = longest - height
            top = tmp // 2
            bottom = tmp - top
        return top, bottom, left, right
        pass

    """根据下标设置标签"""

    def set_label(self, index):
        label = []
        for i in range(self.output_size):
            if i == index:
                label.append(1)
            else:
                label.append(0)
        return label
        pass

    """打乱数据集"""

    def shuffle_data(self, data):
        data = numpy.array(data)
        data_num = len(data) // 2  # 得到样本数
        index = numpy.arange(data_num)  # 生成下标数组
        index = index * 2
        numpy.random.shuffle(index)  # 打乱下标数组
        for i in range(data_num):
            self.data1.append(data[index[i]])
            self.data1.append(data[index[i] + 1])
        pass

    """卷积核"""

    def weight_variable(self, shape):
        # 产生随机变量取值范围[-2*stddev，2*stddev]即[-0.2, 0.2]
        # initial = tf.random.truncated_normal(shape, stddev=0.1)
        initial = tf.random_normal(shape, stddev=0.05)
        return tf.Variable(initial)

    """偏置"""

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)
    """卷积"""
    def conv2d(self, inputs, weight):
        # stride = [1,水平移动步长,竖直移动步长,1]
        return tf.nn.conv2d(inputs, weight, strides=[1, 1, 1, 1], padding='SAME')

    """池化"""
    def pool(self, image):
        # stride = [1,水平移动步长,竖直移动步长,1]
        return tf.nn.max_pool(image, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    """训练模型"""
    def training(self):
        # 模型的默认参数，防止参数的服用出错，最好加再开头
        tf.reset_default_graph()
        #申请将输入的变量用作后期传参
        inputs = tf.placeholder(tf.float32)  # 神经网络输入数据(1*4096)
        inputs_reshape = tf.reshape(inputs, [-1, 128, 128, 1])  # 改变经网络输入数据形状(1个64*64的矩阵)
        labels = tf.placeholder(tf.float32)  # 神经网络输出数据的形状(1*3)
        # 第1次卷积和池化：输入图片128*128*1，输出图片64*64*16
        c1_weight = self.weight_variable([3, 3, 1, 16])  # 卷积核 大小3*3  输入通道1  输出通道32输出通道就是个数
        c1_bias = self.bias_variable([16])
        c1_relu = tf.nn.relu(self.conv2d(inputs_reshape, c1_weight) + c1_bias)  # 激活(卷积后+偏置)
        c1_pool = self.pool(c1_relu)
        # 第1次卷积和池化：输入图片64*64*1，输出图片*32*32
        c2_weight = self.weight_variable([3, 3, 16, 32])  # 卷积核 大小3*3  输入通道1  输出通道32输出通道就是个数
        c2_bias = self.bias_variable([32])
        c2_relu = tf.nn.relu(self.conv2d(c1_pool, c2_weight) + c2_bias)  # 激活(卷积后+偏置)
        c2_pool = self.pool(c2_relu)
        # 第1次卷积和池化：输入图片32*32*1，输出图片16*16*32
        c3_weight = self.weight_variable([3, 3, 32, 64])  # 卷积核 大小3*3  输入通道1  输出通道32输出通道就是个数
        c3_bias = self.bias_variable([64])
        # 偏置=卷积核输出通道,卷积时做了补0，所以规格不变，这是个可选参数
        c3_relu = tf.nn.relu(self.conv2d(c2_pool, c3_weight) + c3_bias)  # 激活(卷积后+偏置)
        c3_pool = self.pool(c3_relu)  # 最大池化后 32*32*32
        # 第2次卷积和池化：输入图片16*16*64，输出图片8*8*128
        c4_weight = self.weight_variable([3, 3, 64, 128])  # 卷积核 大小3*3  输入通道32  输出通道64
        c4_bias = self.bias_variable([128])  # 偏置=卷积核输出通道
        c4_relu = tf.nn.relu(self.conv2d(c3_pool, c4_weight) + c4_bias)  # 激活(卷积后+偏置)
        c4_pool = self.pool(c4_relu)  # 最大池化后 16*16*64
        # 第3次卷积和池化：输入图片8*8*128，输出图片4*4*256
        c5_weight = self.weight_variable([3, 3, 128, 256])  # 卷积核 大小3*3  输入通道64  输出通道64
        c5_bias = self.bias_variable([256])  # 偏置=卷积核输出通道
        c5_relu = tf.nn.relu(self.conv2d(c4_pool, c5_weight) + c5_bias)  # 激活(卷积后+偏置)
        c5_pool = self.pool(c5_relu)  # 最大池化后 8*8*64=4096
        c5_pool_reshape = tf.reshape(c5_pool, [-1, 4 * 4 * 256])  # 改变池化层的形状(1*4096)
        # 全连接层：将图片的卷积输出压扁成一个一维向量
        f4_weight = self.weight_variable([4 * 4 * 256, 4 * 256])  # 权重矩阵   输入通道4096  输出通道512
        fn4_bias = self.bias_variable([4 * 256])  # 偏置=卷积核输出通道
        fn4_relu = tf.nn.relu(tf.matmul(c5_pool_reshape, f4_weight) + fn4_bias)  # [1,4096]*[4096,512]=[1,512]
        fn4_drop = tf.nn.dropout(fn4_relu, keep_prob=self.drop_rate)  # 防止过拟合：丢弃率为rate
        # 第二层(全连接层)
        f5_weight = self.weight_variable([4 * 256, 2 * 128])  # 权重矩阵   输入通道4096  输出通道512
        fn5_bias = self.bias_variable([2 * 128])  # 偏置=卷积核输出通道
        fn5_relu = tf.nn.relu(tf.matmul(fn4_drop, f5_weight) + fn5_bias)  # [1,4096]*[4096,512]=[1,512]
        # 第三层（全连接）
        f6_weight = self.weight_variable([2 * 128, 128])  # 权重矩阵   输入通道4096  输出通道512
        fn6_bias = self.bias_variable([128])  # 偏置=卷积核输出通道
        fn6_relu = tf.nn.relu(tf.matmul(fn5_relu, f6_weight) + fn6_bias)
        # 输出层（全连接）
        f7_weight = self.weight_variable([128, self.output_size])  # 输入通道512， 输出通道3
        f7_bias = self.bias_variable([self.output_size])  # 偏置=卷积核输出通道
        prediction = tf.add(tf.matmul(fn6_relu, f7_weight), f7_bias)
        # 损失函数为交叉熵
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
        # 梯度下降法:选用AdamOptimizer优化器
        train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        # 求得准确率，比较标签是否相等，再求的所有数的平均值
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1)), tf.float32))
        self.data = np.asarray(self.data)
        self.data1 = np.asarray(self.data1)
        images1, label1 = self.get_Batch(self.data, self.data1, self.batch_size)
        summary_op = tf.summary.merge_all()
        sess = tf.Session()
        # images2 = images1.eval()
        # label2 = label1.eval()
        train_writer = tf.summary.FileWriter(self.model_save, sess.graph)
        #起动机算
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())  # 就是这一行
        saver = tf.train.Saver()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        epoch = 0
        try:
            while not coord.should_stop():
                #将数据转化为tensor张量
                data, label = sess.run([images1, label1])
                epoch = epoch + 1
                # 启动以下操作节点
                sess.run(train_step, feed_dict={inputs: data, labels: label})
                loss1 = sess.run(loss, feed_dict={inputs: data, labels: label})
                accuracy1 = sess.run(accuracy, feed_dict={inputs: data, labels: label})
                print('损失为' + str(loss1))
                # 每隔50步打印一次当前的loss以及acc，同时记录log，写入writer
                if epoch % ((len(self.data)) // self.batch_size) == 0:
                    print('准确率' + str(accuracy1))
                    saver.save(sess, self.path)
                # 保存最后一次网络参数
        except tf.errors.OutOfRangeError:
            print('Done training')
        finally:
            coord.request_stop()
        coord.join(threads)
        sess.close()

    # 使用模型检测
    def detection(self):  # 检查数据集
        # self.check()
        tf.reset_default_graph()
        inputs = tf.placeholder(tf.float32)  # 神经网络输入数据(1*4096)
        inputs_reshape = tf.reshape(inputs, [-1, 128, 128, 1])  # 改变经网络输入数据形状(1个64*64的矩阵)
        labels = tf.placeholder(tf.float32)  # 神经网络输出数据的形状(1*3)
        # 第1次卷积和池化：输入图片128*128*1，输出图片64*64*16
        c1_weight = self.weight_variable([3, 3, 1, 16])  # 卷积核 大小3*3  输入通道1  输出通道32输出通道就是个数
        c1_bias = self.bias_variable([16])
        c1_relu = tf.nn.relu(self.conv2d(inputs_reshape, c1_weight) + c1_bias)  # 激活(卷积后+偏置)
        c1_pool = self.pool(c1_relu)
        # 第1次卷积和池化：输入图片64*64*1，输出图片*32*32
        c2_weight = self.weight_variable([3, 3, 16, 32])  # 卷积核 大小3*3  输入通道1  输出通道32输出通道就是个数
        c2_bias = self.bias_variable([32])
        c2_relu = tf.nn.relu(self.conv2d(c1_pool, c2_weight) + c2_bias)  # 激活(卷积后+偏置)
        c2_pool = self.pool(c2_relu)
        # 第1次卷积和池化：输入图片32*32*1，输出图片16*16*32
        c3_weight = self.weight_variable([3, 3, 32, 64])  # 卷积核 大小3*3  输入通道1  输出通道32输出通道就是个数
        c3_bias = self.bias_variable([64])
        # 偏置=卷积核输出通道,卷积时做了补0，所以规格不变，这是个可选参数
        c3_relu = tf.nn.relu(self.conv2d(c2_pool, c3_weight) + c3_bias)  # 激活(卷积后+偏置)
        c3_pool = self.pool(c3_relu)  # 最大池化后 32*32*32
        # 第2次卷积和池化：输入图片16*16*64，输出图片8*8*128
        c4_weight = self.weight_variable([3, 3, 64, 128])  # 卷积核 大小3*3  输入通道32  输出通道64
        c4_bias = self.bias_variable([128])  # 偏置=卷积核输出通道
        c4_relu = tf.nn.relu(self.conv2d(c3_pool, c4_weight) + c4_bias)  # 激活(卷积后+偏置)
        c4_pool = self.pool(c4_relu)  # 最大池化后 16*16*64
        # 第3次卷积和池化：输入图片8*8*128，输出图片4*4*256
        c5_weight = self.weight_variable([3, 3, 128, 256])  # 卷积核 大小3*3  输入通道64  输出通道64
        c5_bias = self.bias_variable([256])  # 偏置=卷积核输出通道
        c5_relu = tf.nn.relu(self.conv2d(c4_pool, c5_weight) + c5_bias)  # 激活(卷积后+偏置)
        c5_pool = self.pool(c5_relu)  # 最大池化后 8*8*64=4096
        c5_pool_reshape = tf.reshape(c5_pool, [-1, 4 * 4 * 256])  # 改变池化层的形状(1*4096)
        # 全连接层：将图片的卷积输出压扁成一个一维向量
        f4_weight = self.weight_variable([4 * 4 * 256, 4 * 256])  # 权重矩阵   输入通道4096  输出通道512
        fn4_bias = self.bias_variable([4 * 256])  # 偏置=卷积核输出通道
        fn4_relu = tf.nn.relu(tf.matmul(c5_pool_reshape, f4_weight) + fn4_bias)  # [1,4096]*[4096,512]=[1,512]
        fn4_drop = tf.nn.dropout(fn4_relu, keep_prob=self.drop_rate)  # 防止过拟合：丢弃率为rate
        # 第二层(全连接层)
        f5_weight = self.weight_variable([4 * 256, 2 * 128])  # 权重矩阵   输入通道4096  输出通道512
        fn5_bias = self.bias_variable([2 * 128])  # 偏置=卷积核输出通道
        fn5_relu = tf.nn.relu(tf.matmul(fn4_drop, f5_weight) + fn5_bias)  # [1,4096]*[4096,512]=[1,512]
        # 第三层（全连接）
        f6_weight = self.weight_variable([2 * 128, 128])  # 权重矩阵   输入通道4096  输出通道512
        fn6_bias = self.bias_variable([128])  # 偏置=卷积核输出通道
        fn6_relu = tf.nn.relu(tf.matmul(fn5_relu, f6_weight) + fn6_bias)
        # 输出层（全连接）
        f7_weight = self.weight_variable([128, self.output_size])  # 输入通道512， 输出通道3
        f7_bias = self.bias_variable([self.output_size])  # 偏置=卷积核输出通道
        prediction = tf.add(tf.matmul(fn6_relu, f7_weight), f7_bias)
        # 损失函数为交叉熵
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
        # 梯度下降法:选用AdamOptimizer优化器
        train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        # 求得准确率，比较标签是否相等，再求的所有数的平均值
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1)), tf.float32))

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.07  # 占用GPU90%的显存
        session = tf.Session(config=config)
        sess = session
        sess.run(tf.global_variables_initializer())  # Session对象初始化
        saver = tf.train.Saver()  # 创建Saver对象
        saver.restore(sess, self.path)  # 使用模型文件
        # self.data = self.shuffle_data(self.data)
        self.data = np.asarray(self.data)
        for i in range(len(self.data)):
            prediction2 = sess.run(prediction, feed_dict={inputs: self.data[i]})
            #print(numpy.argmax(prediction2[0]))
            self.prediction.append(numpy.argmax(prediction2[0]))
            self.data = list(self.data)
            self.data = []
    def test(self):
        # self.check()
        tf.reset_default_graph()
        inputs = tf.placeholder(tf.float32)  # 神经网络输入数据(1*4096)
        inputs_reshape = tf.reshape(inputs, [-1, 128, 128, 1])  # 改变经网络输入数据形状(1个64*64的矩阵)
        labels = tf.placeholder(tf.float32)  # 神经网络输出数据的形状(1*3)
        # 第1次卷积和池化：输入图片128*128*1，输出图片64*64*16
        c1_weight = self.weight_variable([3, 3, 1, 16])  # 卷积核 大小3*3  输入通道1  输出通道32输出通道就是个数
        c1_bias = self.bias_variable([16])
        c1_relu = tf.nn.relu(self.conv2d(inputs_reshape, c1_weight) + c1_bias)  # 激活(卷积后+偏置)
        c1_pool = self.pool(c1_relu)
        # 第1次卷积和池化：输入图片64*64*1，输出图片*32*32
        c2_weight = self.weight_variable([3, 3, 16, 32])  # 卷积核 大小3*3  输入通道1  输出通道32输出通道就是个数
        c2_bias = self.bias_variable([32])
        c2_relu = tf.nn.relu(self.conv2d(c1_pool, c2_weight) + c2_bias)  # 激活(卷积后+偏置)
        c2_pool = self.pool(c2_relu)
        # 第1次卷积和池化：输入图片32*32*1，输出图片16*16*32
        c3_weight = self.weight_variable([3, 3, 32, 64])  # 卷积核 大小3*3  输入通道1  输出通道32输出通道就是个数
        c3_bias = self.bias_variable([64])
        # 偏置=卷积核输出通道,卷积时做了补0，所以规格不变，这是个可选参数
        c3_relu = tf.nn.relu(self.conv2d(c2_pool, c3_weight) + c3_bias)  # 激活(卷积后+偏置)
        c3_pool = self.pool(c3_relu)  # 最大池化后 32*32*32
        # 第2次卷积和池化：输入图片16*16*64，输出图片8*8*128
        c4_weight = self.weight_variable([3, 3, 64, 128])  # 卷积核 大小3*3  输入通道32  输出通道64
        c4_bias = self.bias_variable([128])  # 偏置=卷积核输出通道
        c4_relu = tf.nn.relu(self.conv2d(c3_pool, c4_weight) + c4_bias)  # 激活(卷积后+偏置)
        c4_pool = self.pool(c4_relu)  # 最大池化后 16*16*64
        # 第3次卷积和池化：输入图片8*8*128，输出图片4*4*256
        c5_weight = self.weight_variable([3, 3, 128, 256])  # 卷积核 大小3*3  输入通道64  输出通道64
        c5_bias = self.bias_variable([256])  # 偏置=卷积核输出通道
        c5_relu = tf.nn.relu(self.conv2d(c4_pool, c5_weight) + c5_bias)  # 激活(卷积后+偏置)
        c5_pool = self.pool(c5_relu)  # 最大池化后 8*8*64=4096
        c5_pool_reshape = tf.reshape(c5_pool, [-1, 4 * 4 * 256])  # 改变池化层的形状(1*4096)
        # 全连接层：将图片的卷积输出压扁成一个一维向量
        f4_weight = self.weight_variable([4 * 4 * 256, 4 * 256])  # 权重矩阵   输入通道4096  输出通道512
        fn4_bias = self.bias_variable([4 * 256])  # 偏置=卷积核输出通道
        fn4_relu = tf.nn.relu(tf.matmul(c5_pool_reshape, f4_weight) + fn4_bias)  # [1,4096]*[4096,512]=[1,512]
        fn4_drop = tf.nn.dropout(fn4_relu, keep_prob=self.drop_rate)  # 防止过拟合：丢弃率为rate
        # 第二层(全连接层)
        f5_weight = self.weight_variable([4 * 256, 2 * 128])  # 权重矩阵   输入通道4096  输出通道512
        fn5_bias = self.bias_variable([2 * 128])  # 偏置=卷积核输出通道
        fn5_relu = tf.nn.relu(tf.matmul(fn4_drop, f5_weight) + fn5_bias)  # [1,4096]*[4096,512]=[1,512]
        # 第三层（全连接）
        f6_weight = self.weight_variable([2 * 128, 128])  # 权重矩阵   输入通道4096  输出通道512
        fn6_bias = self.bias_variable([128])  # 偏置=卷积核输出通道
        fn6_relu = tf.nn.relu(tf.matmul(fn5_relu, f6_weight) + fn6_bias)
        # 输出层（全连接）
        f7_weight = self.weight_variable([128, self.output_size])  # 输入通道512， 输出通道3
        f7_bias = self.bias_variable([self.output_size])  # 偏置=卷积核输出通道
        prediction = tf.nn.softmax(tf.add(tf.matmul(fn6_relu, f7_weight), f7_bias))
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
        # 梯度下降法:选用AdamOptimizer优化器
        train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        # 求得准确率，比较标签是否相等，再求的所有数的平均值
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1)), tf.float32))

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())  # Session对象初始化
        saver = tf.train.Saver()  # 创建Saver对象
        saver.restore(sess, self.path)  # 使用模型文件

        t = 0
        for i in range(len(self.data)):
            q = 0
            for j in range(self.output_size):
                if self.data1[i][j] != 0:
                    q = j
            prediction2 = sess.run(prediction, feed_dict={inputs: self.data[i]})
            print(prediction2)
            # if numpy.argmax(prediction2[0]) == self.data1[2 * i]:
            if numpy.argmax(prediction2[0]) == q:
                t = t + 1
        print(t / (len(self.data1)))
        self.reslut = (t / (len(self.data1)))
        # print(t/(len(self.data1)//2))

if __name__ == '__main__':
    #神网训练
    #img = Image.open("/home/itc/ai_resource/class_data/00.png")
    fcnn = FCNN()
    fcnn.set_data()
    #fcnn.shuffle_data(fcnn.data)
    #cnn.red_data(img)
    #fcnn.training()
    #神网测试
    #cnn = CNN()
    #fcnn.detection()
    #调用测试
    #cnn = CNN(0.5, None, 3)
    fcnn.test()
    #print(fcnn.data[0])