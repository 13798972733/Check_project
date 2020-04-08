# encoding:utf-8
###########################################################################
#                        行为检测                                          #
###########################################################################
from classes.facenet.understand_facenet.align import detect_face
from classes.facenet.understand_facenet import facenet
from classes.settings import FACE_DISTANCE_THRESHOLD
import tensorflow as tf
import numpy as np
import os
from PIL import Image
import cv2
from classes.settings import CLASS_RESOURSE_PATH, FACE_MODEL_PATH, FACE_CLASSIFIER_MODEL_PATH
from classes.object.modelsmaster.research.my_object_detection import behavior
from classes import face_detector1
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if not os.path.exists(FACE_CLASSIFIER_MODEL_PATH):
    os.makedirs(FACE_CLASSIFIER_MODEL_PATH)
if not os.path.exists(CLASS_RESOURSE_PATH):
    os.makedirs(CLASS_RESOURSE_PATH)
#使用默认模型文件
tf.reset_default_graph()
# 下载下来的模型的目录
class Detect(object):
    def __init__(self):
        self.minsize = 20
        self.factor = 0.709
        self.threshold = [0.6, 0.7, 0.7]
        '''self.MODEL_DIR = '/home/itc/PycharmProjects/untitled/object/modelsmaster/research/my_object_detection/models/ssd_mobilenet_v1_coco_2018_01_28'
        # 下载下来的模型的文件
        self.MODEL_CHECK_FILE = os.path.join(self.MODEL_DIR, 'frozen_inference_graph.pb')
        # 数据集对于的label
        self.MODEL_LABEL_MAP = os.path.join('/home/itc/PycharmProjects/untitled/object/modelsmaster/research/my_object_detection/data', 'mscoco_label_map.pbtxt')
        # 数据集分类数量，可以打开mscoco_label_map.pbtxt文件看看
        self.MODEL_NUM_CLASSES = 90
        # 这里是获取实例图片文件名，将其放到数组中
        self.PATH_TO_TEST_IMAGES_DIR = '/home/itc/PycharmProjects/untitled/object/modelsmaster/research/my_object_detection/images/test_images'
        self.TEST_IMAGES_PATHS = [os.path.join(self.PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 2)]'''
        self.bounding_boxes = np.array([[0, 1, 2, 3, 4]])
        # 输出图像大小，单位是in
        self.IMAGE_SIZE = (12, 8)
        tf.reset_default_graph()
        #创建字典
        self.dit = {0: "听讲", 1: "学习", 2: "活跃", 3: "休息"}
        # 将模型读取到默认的图中
        # 加载COCO数据标签，将mscoco_label_map.pbtxt的内容转换成
        # {1: {'id': 1, 'name': u'person'}...90: {'id': 90, 'name': u'toothbrush'}}格式
        self.boxes1 = []
        self.img = []
        self.graph = tf.Graph()
        self.distance_threshold = FACE_DISTANCE_THRESHOLD
        self.all_model_map = {}
        self.minsize = 20
        self.threshold = [0.6, 0.7, 0.7]
        self.factor = 0.709
        self.graph = tf.Graph()
        with self.graph.as_default():
            config = tf.ConfigProto()
            #配置GPU资源站四分之一
            config.gpu_options.per_process_gpu_memory_fraction = 0.25  # 占用GPU90%的显存
            #启动会话
            session = tf.Session(config=config)
            self.sess = session
            with self.sess.as_default():
                self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(self.sess, None)
                facenet.load_model("/home/itc/20180402-114759/")
                # 返回给定名称的tensor
                self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

    # 将图片转化成numpy数组形式
    def load_image_into_numpy_array(self,image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

    def to_rgb1(self, img):
        w, h = img.shape
        ret = np.empty((w, h, 3), dtype=np.uint8)
        ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
        return ret


    # 在图中开始，根据坐标扣出人脸
    def Calculation(self, image1):
        image = image1
        image = np.asarray(image)
        h, w = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_size = np.asarray(image.shape)[0:2]
        image = self.to_rgb1(image)
        bounding_boxes, _ = detect_face.detect_face(image, self.minsize, self.pnet, self.rnet, self.onet,
                                                          self.threshold, self.factor)
        for i in range(len(bounding_boxes)):
            det = np.squeeze(bounding_boxes[i, 0:4])
            bb = np.zeros(4, dtype=np.int32)
            # x1, y1, x2, y2
            margin =det[2] - det[0]
            margin1 = det[3] - det[1]
            bb[0] = np.maximum(det[0] - margin / 2, 0)
            bb[1] = np.maximum(det[1] - (margin1*2) / 2, 0)
            bb[2] = np.minimum(det[2] + margin / 2, w)
            bb[3] = np.minimum(det[3] + (margin1*2) / 2, h)
            imge = image1.crop([bb[0], bb[1], bb[2], bb[3]])
            self.img.append(imge)
            #self.bounding_boxes = np.row_stack((self.bounding_boxes, bounding_boxes))

'''def add(self,image):
        tf.reset_default_graph()
        (img_width, img_height) = image.size
        self.boxes1 = np.asarray(self.boxes1)
        for i in range(self.boxes1.shape[0]):
            for j in range(self.boxes1.shape[1]):
                imge = image.crop([int((self.boxes1[i][j][1])*(img_width)), int((self.boxes1[i][j][0])*(img_height)), int((self.boxes1[i][j][3])*(img_width)), int((self.boxes1[i][j][2])*(img_height))])
                img = np.asarray(imge)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_size = np.asarray(img.shape)[0:2]
                img = self.to_rgb1(img)
                bounding_boxes, _ = align.detect_face.detect_face(img, self.minsize, self.pnet, self.rnet, self.onet,
                                                                  self.threshold, self.factor)
                if (bounding_boxes.shape)==(1,5):
                    bounding_boxes1 = self.bounding_boxes[:, 0]
                    r = 0
                    for w in bounding_boxes1:
                        if bounding_boxes[0][0] == w:
                            r = r+1
                    if r == 0:
                        self.img.append(imge)
                        self.bounding_boxes = np.row_stack((self.bounding_boxes,bounding_boxes))'''

if __name__ == '__main__':
    dcnn = Detect()
    iamge = Image.open("/home/itc/PycharmProjects/untitled/object/modelsmaster/research/my_object_detection/images/test_images/1.jpg")
    dcnn.Calculation(iamge)
    print(len(dcnn.img))
    '''fcnn = face_detector1.FCNN()
    acnn = behavior.ACNN()
    for img in dcnn.img:
        fcnn.red_data(img)
        fcnn.detection()
        fcnn_reslut = fcnn.prediction
        fcnn.prediction = []
        acnn.red_data(img)
        acnn.detection()
        acnn_reslut = acnn.prediction
        acnn.prediction = []
        #print('此人为'+str(fcnn_reslut)+'此人行为'+str(acnn.dict[acnn_reslut[0]]))
        #img = np.asarray(img)
        #cv2.imshow('image', img)
        #cv2.waitKey(0)'''