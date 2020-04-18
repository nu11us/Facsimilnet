import torch
import torch.nn as nn
import torchvision
import pandas as pd
import math
import numpy as np
from ringloss.loss_func import RingLoss
from PIL import Image
import types
from torch.utils.tensorboard import SummaryWriter

ALIGNED_PATH = "WC-MS-Celeb/WC-MS-Celeb.tsv"
CROPPED_PATH = "WC-MS-Celeb/WC-MS-Celeb.tsv"

JSON_PATH = "WC-MS-Celeb/json/"
CLASSES_PATH = "WC-MS-Celeb/data/WDList"
NUM_CLASSES = 77543
NUM_IMAGES = 5429085

BATCH_SIZE = 16
NUM_WORKERS = 4

LEARNING_RATE = 0.0005
WEIGHT_DECAY =  0.0
MOMENTUM = 0.8
LR_STEP = 10

MODE = "pretrained"
OPTIM = "adam"

def get_num_classes():
    s = set()
    with open(ALIGNED_PATH) as a:
        for line in a:
            s.add(line.split('\t')[0])
    return len(s)

def get_shapes():
    with open(ALIGNED_PATH) as a:
        for line in a:
            out = io.BytesIO(base64.b64decode(line.split('\t')[-1]))
            img = Image.open(out)
            print(img)
            input()

def classes():
    lst = []
    idx = {}
    with open(CLASSES_PATH) as classes:
        for line in classes:
            lst.append(line.rstrip())
            idx[lst[-1]] = len(lst) - 1
    return lst, idx


class AlignedWC(torch.utils.data.Dataset):
    def __init__(self, tsv, transform = None):
        super(AlignedWC).__init__() 
        self.df = pd.read_csv(tsv, names=['tag', 'image'],  sep='\t')
        self.classes, self.class_idx = classes()
        self.tf = transform

    def transform(self, image):
        if self.tf == None:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize([224,224]),
                torchvision.transforms.ToTensor()
            ])
        else:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize([224,224]),
                self.tf,
                torchvision.transforms.ToTensor()
            ])
        return transform(image)
        

    def loader(self, image):
        return torchvision.datasets.folder.default_loader('aligned/'+image)

    def __getitem__(self, index):
        return (
            self.class_idx[self.df.iloc[index]['tag']],
            self.transform(self.loader(self.df.iloc[index]['image']))
        )

    def __len__(self):
        return len(self.df.index)

def train(trainloader, num_epochs = 10):
    # partially from pytorch docs
    for epoch in range(num_epochs):
        run_loss = 0.0
        run_ring = 0.0
        for i, data in enumerate(trainloader, 0):
            optimizer.zero_grad()
            inputs, labels = data[1].to(device), data[0].to(device) 
            outputs = model(inputs)
            softmax_val = loss(softmax(outputs),labels)
            ring_val = ringloss(outputs)
            loss_val = softmax_val + ring_val
            loss_val.backward()
            optimizer.step()   
            
            run_loss += softmax_val.item()
            run_ring += ring_val.item()
            if i % 50 == 49:    # print every 10 mini-batches
                correct  = (outputs.data.max(1)[1] == labels).sum().item()
                '''
                print('[%d, %5d, %.3f%%] acc: %d/%d, speed: %.6fs, loss: %.3f' %
                     (
                        epoch + 1,
                        BATCH_SIZE*(i + 1),
                        100*BATCH_SIZE*(i+1)/(NUM_IMAGES*0.8), 
                        correct, 
                        BATCH_SIZE, 
                        (time.time()-start)/(BATCH_SIZE*50), 
                        run_loss / 50
                    )
                )
                '''
                writer.add_scalar('Loss/train', (run_loss+run_ring)/50, BATCH_SIZE*(i+1))
                writer.add_scalar('Softmax/train', run_loss/50, BATCH_SIZE*(i+1))
                writer.add_scalar('Ring/train', run_ring/50, BATCH_SIZE*(i+1))
                writer.add_scalar('Accuracy/train', correct/(BATCH_SIZE), BATCH_SIZE*(i+1))
                run_loss = 0.0
                run_ring = 0.0
        scheduler.step()

def test(testloader):
    # partially from pytorch docs
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[1], data[0]
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted==labels).sum().item()
    print("correct {}/{}".format(correct, total))

writer = SummaryWriter()

data = AlignedWC(ALIGNED_PATH)

train_data, test_data = torch.utils.data.random_split(data, [int(.8*len(data)), int(.2*len(data))])

train_loader = torch.utils.data.DataLoader(train_data, batch_size = BATCH_SIZE, num_workers = NUM_WORKERS, pin_memory=True, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size = BATCH_SIZE, num_workers = NUM_WORKERS, pin_memory=True, shuffle=True)

if MODE == "pretrained":
    model = torchvision.models.resnet50(pretrained=True)
    model.fc = nn.Linear(2048, NUM_CLASSES)
elif MODE == "default":
    model = torchvision.models.resnet50(pretrained=False, num_classes=NUM_CLASSES)

softmax = nn.LogSoftmax(dim=1)
loss = nn.NLLLoss()
ringloss = RingLoss(type='auto', loss_weight=1.0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model.to(device)
softmax.to(device)
loss.to(device)
ringloss.to(device)

if OPTIM == "adam":
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
elif OPTIM == "sgd":
    optimizer = torch.optim.SGD(params=model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP, gamma=0.1)
            
train(train_loader, num_epochs=1)
test(test_loader)
torch.save(model.state_dict(), "epoch1.pth")
