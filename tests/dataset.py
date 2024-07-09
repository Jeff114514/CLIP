import sys
import os
import os.path as osp
import glob
import re


class MyData():

    def __init__(self, datasetPath = 'D:\code\L-MBN\LMBN\MyData'):
        self.cur_path = datasetPath
        self.data_path = osp.join(self.cur_path, 'satimg_train')
        train_list = os.listdir(self.data_path)
        trainImgList = []
        trainNameList = []
        trainIdList = []
        for lable in train_list:
            #匹配__之前的全部字符和__之后的数字
            match = re.match(r"(.*)__(.*)", lable)
            satName = match.group(1)
            satId = int(match.group(2))
            for img in glob.glob(os.path.join(self.data_path, lable, '*.jpeg')):
                trainImgList.append(img)
                name = "a photo of " + satName
                trainNameList.append(name)
                trainIdList.append(satId)

        self.trainData = {
            "image_path": trainImgList,
            "name": trainNameList,
            "id": trainIdList
        }

        self.query_path = osp.join(self.cur_path, 'satellite')
        query_lables = os.listdir(self.query_path)
        queryImgList = []
        queryNameList = []
        queryIdList = []
        for lable in query_lables:
            match = re.match(r"(.*)__(.*)", lable)
            satName = match.group(1)
            satId = int(match.group(2).split('.')[0])
            queryImgList.append(os.path.join(self.query_path, lable))
            name = "a photo of " + satName
            queryNameList.append(name)
            queryIdList.append(satId)
            
        self.queryData = {
            "image_path": queryImgList,
            "name": queryNameList,
            "id": queryIdList
        }

        # 未准备测试集
        # 去除trainData中不在query中出现的名字
        query_names_set = set(queryNameList)
        filtered_trainImgList = []
        filtered_trainNameList = []
        filtered_trainIdList = []

        for img, name, id in zip(trainImgList, trainNameList, trainIdList):
            if name in query_names_set:
                filtered_trainImgList.append(img)
                filtered_trainNameList.append(name)
                filtered_trainIdList.append(id)

        self.galleryData = {
            "image_path": filtered_trainImgList,
            "name": filtered_trainNameList,
            "id": filtered_trainIdList
        }

    def getTrainData(self):
        return self.trainData
    
    def getQueryData(self):
        return self.queryData
    
    def getGalleryData(self):
        return self.galleryData
