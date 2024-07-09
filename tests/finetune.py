from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import clip
from transformers import CLIPProcessor,CLIPModel
from tqdm import tqdm
import os
from dataset import MyData

root = os.path.dirname(os.path.abspath(__file__))
modelPath = "../model/ViT-L-14-336px.pt"
modelPath = os.path.join(root, modelPath)
device = "cuda" if torch.cuda.is_available() else "cpu" 

model, processor = clip.load(modelPath, device=device)
tokenizer = clip.tokenize

batch_size = 100
test_batch_size = 100
lr = 5e-5
betas = (0.9, 0.98)
eps = 1e-6
weight_decay = 0.2
epochs = 10

# dataset = image_title_dataset(list_image_path, list_txt)
class image_title_dataset():
	
	def __init__(self, list_image_path, list_txt):
		self.image_path = list_image_path
		self.name = clip.tokenize(list_txt)
	
	def __len__(self):
		return len(self.name)
        
	def __getitem__(self, idx):
		image = processor(Image.open(self.image_path[idx]))
		name = self.name[idx] 
		return image, name

data = MyData()
trainData = data.getTrainData()
queryData = data.getQueryData()
galleryData = data.getGalleryData()

trainDataset = image_title_dataset(trainData["image_path"], trainData["name"])
queryDataset = image_title_dataset(queryData["image_path"], queryData["name"])
galleryDataset = image_title_dataset(galleryData["image_path"], galleryData["name"])

train_dataloader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True)
query_dataloader = DataLoader(queryDataset, batch_size=batch_size, shuffle=False)
gallery_dataloader = DataLoader(galleryDataset, batch_size=batch_size, shuffle=False)

optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas, eps=eps ,weight_decay=weight_decay) 

def train():
	model.train()
	for epoch in range(epochs):
		pbar = tqdm(train_dataloader, total=len(train_dataloader), desc=f"Epoch {epoch + 1}/{epochs}")
		for images, titles in pbar:
			images = images.to(device)
			titles = titles.to(device)
			optimizer.zero_grad()
			
			logits_per_image, logits_per_text = model(images, titles)
			gt = torch.arange(len(images), dtype=torch.long, device=device)
            
            # Use both to compute loss
			loss = nn.CrossEntropyLoss()(logits_per_image, gt) + nn.CrossEntropyLoss()(logits_per_text, gt)
			loss.backward()
			optimizer.step()
			pbar.set_postfix({"loss": f"{loss.item():.4f}"})

def eval():
    model.eval()
    with torch.no_grad():
        pbar = tqdm(query_dataloader, total=len(query_dataloader), desc="Evaluating")
        for images, titles in pbar:
            images = images.to(device)
            titles = titles.to(device)
            logits_per_image, logits_per_text = model(images, titles)
            gt = torch.arange(len(images), dtype=torch.long, device=device)
            loss = nn.CrossEntropyLoss()(logits_per_image, gt) + nn.CrossEntropyLoss()(logits_per_text, gt)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
			
train()
eval()

model.save_pretrained("../model/clip_model")
processor.save_pretrained("../model/clip_processor")