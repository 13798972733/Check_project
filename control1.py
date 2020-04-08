# encoding:utf-8
###########################################################################
#                        录制分析功能本地总控程序                            #
###########################################################################
from PIL import Image
import cv2
import numpy as np
from multiprocessing import Process
import random
import time
import tensorflow as tf
from classes import face_detector1
from classes.object.modelsmaster.research.my_object_detection import behavior
from classes.object.modelsmaster.research.my_object_detection import detect1
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tf.reset_default_graph()
class Control(object):
    def __init__(self):
        self.seet = 1
        self.people1 = []
        self.behavior1 = []
        self.set = 0
        self.time1 = []
        self.people = []
        self.behavior = []
        self.address = "/home/itc/PycharmProjects/untitled/cws_AI/check/classes/object/modelsmaster/research/my_object_detection/images/test_images"
        self.CLASS_RESOURSE_PATH = "/home/itc/ai_resource/class_data/15"
        self.total = []
        self.image = []
        self.dict = {0:"听讲", 1:"学习", 2:"活跃", 3:"趴卧"}
        self.dict1 = {0: "听讲时间较多", 1: "学习时间较多", 2: "活跃时间较多", 3: "低头时间较多"}
        self.pil_im = []
        self.class_id = []
        self.student = []
        self.reslut = []
        self.reslut1 = []
        self.number = 0
        self.state0 = 0
        self.state1 = 0
        self.state2 = 0
        self.signol = 0
        time2 = time.localtime(time.time())
        name = str(str(time2[0]) + '年' + str(time2[1]) + '月' + str(time2[2]) + '日'
                   + str(time2[3]) + ':' + str(time2[4]) + ':' + str(time2[5]))
        desktop_path = "/home/itc/PycharmProjects/untitled/cws_AI/check/classes/object/modelsmaster/research/my_object_detection/"
        self.full_path = desktop_path + name + '.txt'  # 也可以创建一个.doc的word文档
        file = open(self.full_path, 'w')
        file.close()
        time2 = time.localtime(time.time())
        name1 = str(str(time2[0]) + '年' + str(time2[1]) + '月' + str(time2[2]) + '日'
                   + str(time2[3]) + ':' + str(time2[4]) + ':' + str(time2[5])+"_")
        desktop_path = "/home/itc/PycharmProjects/untitled/cws_AI/check/classes/object/modelsmaster/research/my_object_detection/"
        self.full_path1 = desktop_path + name1 + '.txt'  # 也可以创建一个.doc的word文档
        file = open(self.full_path, 'w')
        file.close()
        for i in os.listdir(self.address):
            image = Image.open(os.path.join(self.address,i))
            self.image.append(image)
        print("init done")
    def get_image(self):
        vc = cv2.VideoCapture("rtsp://172.16.18.32:554/1")  # 读入视频文件，命名cv
        rval, frame = vc.read()
        pil_im = Image.fromarray(np.uint8(frame))
        print("get_image done")
        return pil_im
    def star(self,image):
        fcnn = face_detector1.FCNN()
        acnn = behavior.ACNN()
        dcnn = detect1.Detect()
        dcnn.Calculation(image)
        people = []
        behavior1 = []
        behavior2 = []
        people2 = []
        for img in dcnn.img:
            fcnn.red_data(img)
            fcnn.detection()
            fcnn_reslut = fcnn.prediction
            fcnn.prediction = []
            acnn.red_data(img)
            acnn.detection()
            acnn_reslut = acnn.prediction
            acnn.prediction = []
            people.append(fcnn_reslut[0])
            people.append(self.dict[acnn_reslut[0]])
            behavior2.append(acnn_reslut[0])
            people2.append(fcnn_reslut[0])
        self.people.append(people)
        self.time1 = time.localtime(time.time())
        self.people1.append(people2)
        self.behavior1.append(behavior2)
        self.set = self.set + 1
        print("star done")
    def end(self):
        file = open(self.full_path, 'a')
        file.write('此次数据')
        file.write('\r\n')
        file.write(str(str(self.time1[0]) + '年' + str(self.time1[1]) + '月' + str(self.time1[2]) + '日'
                   + str(self.time1[3]) + ':' + str(self.time1[4]) + ':' + str(self.time1[5])))  # msg也就是下面的Hello world!
        file.write('\r\n')
        file.write(str(self.people))  # msg也就是下面的Hello world!
        file.write('\r\n')
        file1 = open(self.full_path1, 'a')
        file1.write(str(self.people1[0]))  # msg也就是下面的Hello world!
        file1.write('\r\n')
        file1.write(str(self.behavior1[0]))
        file1.write('\r\n')
        file1.close()
        self.time1 = []
        self.people = []
        self.behavior = []
        print("end done")
    def control_all(self):
        image = self.image[random.randint(0,20)]
        #image = self.get_image()
        self.star(image)
        self.end()
        print("control_all done")
    def begin(self):
        while self.seet > 0:
            self.control_all()
    def Settlement(self):
        people = []
        behavior = []
        linestr1 = []
        state0 = 0
        state1 = 0
        state2 = 0
        state3 = 0
        with open(self.full_path1, encoding='utf8') as f:
            for line in f.readlines():
                line = eval(line)#字符列表转列表
                linestr1.append(line)
        for i in range((len(linestr1))//2):
            people.append(linestr1[2*i])
            behavior.append(linestr1[2*i+1])
        print(people)
        print(behavior)
        for class_id in os.listdir(self.CLASS_RESOURSE_PATH):
            self.class_id.append(int(class_id))
            student = []
            self.student.append(student)
        for i in range(len(people)):
            for j in range(len(people[i])):
                for k in range(len(self.class_id)):
                    if int(people[i][j]) == self.class_id[k]:
                        #self.student[k].append(self.people1[i][j])
                        self.student[self.class_id[k]].append(int(behavior[i][j]))
        print(self.student)
        for w in range(len(self.student)):
            a = 0
            if len(self.student[w]) == 0:
                print("学生"+str(w)+"始终未抬头或未到")
                b = str("学生"+str(w)+"始终未抬头或未到")
                self.reslut1.append(b)
            else:
                for l in range(len(self.student[w])):
                    if self.student[w][l] == 0:
                        state0 =state0+1
                    if self.student[w][l] == 1:
                        state1 =state1+1
                    if self.student[w][l] == 2:
                        state2 =state2+1
                    if self.student[w][l] == 3:
                        state3 =state3+1
                print(state0,state1,state2,state3)
                if (state0 >= state2) and (state0 >= state1) and (state0 >= state3):
                    a = 0
                else:
                    if (state1 >= state2) and (state1 > state0) and (state1 >= state3):
                        a = 1
                    else:
                        if (state2 > state0) and (state2 > state1) and (state2 >= state3):
                            a = 2
                        else:
                            if (state3 >= state2) and (state3 >= state1) and (state3 >= state0):
                                a = 3
                state0 = 0
                state1 = 0
                state2 = 0
                state3 = 0
                print("学生" + str(w)+self.dict1[a])
                c = str("学生" + str(w)+self.dict1[a])
                self.reslut1.append(c)
        print("Settlement done")
if __name__ == '__main__':
    ccnn = Control()
    ccnn.begin()
    ccnn.Settlement()
    print(ccnn.reslut1)