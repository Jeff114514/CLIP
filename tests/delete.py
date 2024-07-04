import torch
import clip
from PIL import Image
from torchvision import transforms
import os
import numpy as np
import re
import glob
import cv2
 
class ClipEmbeding():
    device = "cuda" if torch.cuda.is_available() else "cpu"
 
    def __init__(self, root):
        modelPath = "../model/ViT-L-14-336px.pt"
        modelPath = os.path.join(root, modelPath)
        self.model, self.processor = clip.load(modelPath, device=self.device)
        self.tokenizer = clip.tokenize
 
    def probs(self, image: Image, textList: list):
        process_image = self.processor(image).unsqueeze(0).to(self.device)
        text = self.tokenizer(textList).to(self.device)
 
        with torch.no_grad():
            logits_per_image, logits_per_text = self.model(process_image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
 
        print("Label probs:", probs)
 
 
    def embeding(self, image: Image, text: str):
        process_image = self.processor(image).unsqueeze(0).to(self.device)
        text = self.tokenizer([text]).to(self.device)
 
        image_features = self.model.encode_image(process_image)
        text_features = self.model.encode_text(text)
        return image_features, text_features
    
    def img2fea(self, image1: Image):
        process_image = self.processor(image1).unsqueeze(0).to(self.device)
        image_features = self.model.encode_image(process_image)
        return image_features
    
    def text2fea(self, text: str):
        text = self.tokenizer([text]).to(self.device)
        text_features = self.model.encode_text(text)
        return text_features

    def cos_sim(self, vector1, vector2):
        return torch.nn.functional.cosine_similarity(vector1, vector2, dim=-1)

 
 
if __name__ == "__main__":
    curPath = os.path.dirname(os.path.abspath(__file__))
    dataPath = 'D:\code\L-MBN\LMBN\MyData\satimg_train'
 
    model = ClipEmbeding(curPath)
    
    folders = os.listdir(dataPath)
    train_data = []
    
    for folder in folders:
        pid = int(re.findall(r"__(.*)",folder)[0])
        for img in glob.glob(os.path.join(dataPath,folder,'*.jpeg')):
            train_data.append((img, pid))

    query_data = []
    query_path = 'D:\code\L-MBN\LMBN\MyData\satellite'
    query_lables = os.listdir(query_path)
    query_ids = [re.findall(r"__(.*)", lable)[0] for lable in query_lables]
    query_ids = [int(i.split('.')[0]) for i in query_ids]
    for i, name in enumerate(glob.glob(os.path.join(query_path, '*.jpg'))):
        query_data.append((name, query_ids[i]))

    for gt, pid in query_data:
        img1 = Image.open(gt)
        fea1 = model.img2fea(img1)
        img2 = None
        path = None
        for train_img, train_pid in train_data:
            if train_pid == pid:
                img2 = Image.open(train_img)
                path = train_img
                break
        if img2 is None:
            print("No match found for pid: ", pid)
            continue
        fea2 = model.img2fea(img2)
        sim = model.cos_sim(fea1, fea2)
        if sim.item() > 0.99:
            print("Match found: ", path)
            os.remove(path)


    


