import torch
import clip
from PIL import Image
from torchvision import transforms
import os
import numpy as np
import re
 
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
    # txSat = 'a satellite'
    # txRoc = 'a rocket'
    # txMap = 'a map'
    # txCar = 'a car'
    textList = ['a satellite', 'a rocket', 'a map', 'a car', 'rocket launch', 'a trunk', 'a news report', 'an advertisement']

    txfeaList = [model.text2fea(text) for text in textList]

    folders = os.listdir(dataPath)
    for folder in folders:
        images = os.listdir(os.path.join(dataPath, folder))
        print(folder)
        for image in images:
            img = Image.open(os.path.join(dataPath, folder, image))
            img = img.convert('RGB')
            imgFea = model.img2fea(img)
            
            likeList = [model.cos_sim(imgFea, txFea)[0].item()*100 for txFea in txfeaList]
            likeList = [np.exp(like) for like in likeList]
            likeList = [like/sum(likeList) for like in likeList]

            # model.probs(img, textList)

            if max(likeList) != likeList[0]:
                os.remove(os.path.join(dataPath, folder, image))
            elif not image.endswith('.jpeg'):
                os.remove(os.path.join(dataPath, folder, image))
                image = re.sub(r"\.(.*)", ".jpeg", image)
                img.save(os.path.join(dataPath, folder, image))

