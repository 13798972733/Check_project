# encoding:utf-8
import cv2
from PIL import Image, ImageDraw, ImageFont
from sklearn.externals import joblib
from sklearn.svm import SVC
from os.path import join as pjoin
import numpy as np
from scipy import misc
import align.detect_face
import tensorflow as tf
import os
import facenet
from settings import CLASS_RESOURSE_PATH, FACE_MODEL_PATH, FACE_CLASSIFIER_MODEL_PATH
from settings import FACE_DISTANCE_THRESHOLD
import copy
import tensorflow as tf
import numpy as np
import os
from PIL import Image
from object_detection.utils import label_map_util
import cv2
from settings import CLASS_RESOURSE_PATH, FACE_MODEL_PATH, FACE_CLASSIFIER_MODEL_PATH
from understand_facenet import facenet
import align.detect_face
from object.modelsmaster.research.my_object_detection import behavior
from cws_AI.ai_server_image_builder.ai_server_v2.build.python_script.common import face_detector1
if not os.path.exists(FACE_CLASSIFIER_MODEL_PATH):
    os.makedirs(FACE_CLASSIFIER_MODEL_PATH)
if not os.path.exists(CLASS_RESOURSE_PATH):
    os.makedirs(CLASS_RESOURSE_PATH)

tf.reset_default_graph()
# 下载下来的模型的目录
class Detect(object):
    def __init__(self):
        self.minsize = 20
        self.factor = 0.709
        self.threshold = [0.6, 0.7, 0.7]
        self.MODEL_DIR = '/home/itc/PycharmProjects/untitled/object/modelsmaster/research/my_object_detection/models/ssd_mobilenet_v1_coco_2018_01_28'
        # 下载下来的模型的文件
        self.MODEL_CHECK_FILE = os.path.join(self.MODEL_DIR, 'frozen_inference_graph.pb')
        # 数据集对于的label
        self.MODEL_LABEL_MAP = os.path.join('/home/itc/PycharmProjects/untitled/object/modelsmaster/research/my_object_detection/data', 'mscoco_label_map.pbtxt')
        # 数据集分类数量，可以打开mscoco_label_map.pbtxt文件看看
        self.MODEL_NUM_CLASSES = 90
        # 这里是获取实例图片文件名，将其放到数组中
        self.PATH_TO_TEST_IMAGES_DIR = '/home/itc/PycharmProjects/untitled/object/modelsmaster/research/my_object_detection/images/test_images'
        self.TEST_IMAGES_PATHS = [os.path.join(self.PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 2)]
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
            self.sess = tf.Session()
            with self.sess.as_default():
                self.pnet, self.rnet, self.onet = align.detect_face.create_mtcnn(self.sess, None)
                facenet.load_model(FACE_MODEL_PATH)
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


    # 在图中开始
    def Calculation(self, image):
        tf.reset_default_graph()
        with tf.gfile.GFile(self.MODEL_CHECK_FILE, 'rb') as fd:
            _graph = tf.GraphDef()
            _graph.ParseFromString(fd.read())
            tf.import_graph_def(_graph, name='')
        # 加载COCO数据标签，将mscoco_label_map.pbtxt的内容转换成
        # {1: {'id': 1, 'name': u'person'}...90: {'id': 90, 'name': u'toothbrush'}}格式
        label_map = label_map_util.load_labelmap(self.MODEL_LABEL_MAP)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=self.MODEL_NUM_CLASSES)
        category_index = label_map_util.create_category_index(categories)
        detection_graph = tf.get_default_graph()
        with tf.Session(graph=detection_graph) as sess:
            image_np = self.load_image_into_numpy_array(image)
            # 增加一个维度
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # 下面都是获取模型中的变量，直接使用就好了
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # 存放所有检测框
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # 每个检测结果的可信度
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            # 每个框对应的类别
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            # 检测框的个数
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            # 开始计算
            (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections],feed_dict={image_tensor: image_np_expanded})
        boxes2 = []
        classes2 = []
        scores2 = []
        for i in range(boxes.shape[0]):
            boxes1 = []
            classes1 = []
            scores1 = []
            for j in range(boxes.shape[1]):
                if boxes[i][j][0] != 0.0 and classes[i][j] == 1:
                    boxes1.append(boxes[i][j])
                    classes1.append(classes[i][j])
                    scores1.append(scores[i][j])
                    self.boxes1.append(boxes1)
    def add(self,image):
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
                        self.bounding_boxes = np.row_stack((self.bounding_boxes,bounding_boxes))

if __name__ == '__main__':
    dcnn = Detect()
    iamge = Image.open("/home/itc/PycharmProjects/untitled/object/modelsmaster/research/my_object_detection/images/test_images/1.jpg")
    dcnn.Calculation(iamge)
    dcnn.add(iamge)
    print(len(dcnn.img))
    fcnn = face_detector1.FCNN()
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
        print('此人为'+str(fcnn_reslut)+'此人行为'+str(acnn.dict[acnn_reslut[0]]))
        #img = np.asarray(img)
        #cv2.imshow('image', img)
        #cv2.waitKey(0)'''