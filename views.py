# -*- coding: utf-8 -*-
###########################################################################
#                        控制网络传参调用本地功能模块                         #
###########################################################################
import sys
from django.shortcuts import render
from PIL import Image
from . import face_detector1
from .object.modelsmaster.research.my_object_detection import behavior
from .object.modelsmaster.research.my_object_detection import detect1
from .object.modelsmaster.research.my_object_detection import control1
from django.http import HttpResponseRedirect
# Create your views here.
ccnn = control1.Control()
b = 0
#标准调用格式返回网页
def homepage(request):
    return render(request, 'homepage.html')
def toface(request):
    return render(request, 'facepafe.html')
def tobehavior(request):
    return render(request, 'behaviorpage.html')
def tonanlysis(request):
    return render(request, 'analysis.html')
def tofacetrain(request):

    request.encoding = 'utf-8'
    #post方法传送前段数据回来
    a = request.POST.getlist("_province")
    a1 = float(a[0])
    b = request.POST.getlist("_province1")
    b1 = float(b[0])
    c = request.POST.getlist("_province2")
    c1 = float(c[0])
    print(a1,b1,c1)
    #Django以字典的形式传参
    reslut = {}
    #调用功能模块
    fcnn = face_detector1.FCNN()
    fcnn.drop_rate = c1
    fcnn.learning_rate = a1
    fcnn.train_step = b1
    fcnn.set_data()
    fcnn.training()
    fcnn.test()
    #django给前段传参的方式
    reslut['reslut'] = fcnn.reslut
    return  render(request, 'tofacetrain.html',reslut)
def tobehaviortrain(request):
    request.encoding = 'utf-8'
    a = request.POST.getlist("1_province")
    a1 = float(a[0])
    b = request.POST.getlist("2_province1")
    b1 = float(b[0])
    c = request.POST.getlist("3_province2")
    c1 = float(c[0])
    print(a1, b1, c1)
    reslut = {}
    # 调用功能模块
    acnn = behavior.ACNN()
    acnn.drop_rate = c1
    acnn.learning_rate = a1
    acnn.train_step = b1
    acnn.set_data()
    acnn.training()
    acnn.test()
    # django给前段传参的方式
    reslut['reslut'] = acnn.reslut
    return render(request, 'tobehaviortrain.html',reslut)
def begin(request):
    reslut = {}
    c = ""
    request.encoding = 'utf-8'
    a = request.POST.getlist("1")
    b = request.POST.getlist("2")
    if len(a) >0:
        c = a[0]
    else:
        if len(b) >0:
            c = b[0]
    if c == '开始录制':
        ccnn.seet = 1
        ccnn.begin()
        return render(request, 'begin.html')
    if c == '开始分析':
        ccnn.seet = -1
        ccnn.Settlement()
        reslut['reslut'] = str(ccnn.reslut1)
        return render(request, 'begin.html',reslut)

