from scipy.io import loadmat
import os
import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from transformers import AutoModel, BertTokenizerFast, AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch import optim
import time
import pandas as pd

device = torch.device("cuda")
label = loadmat('e:\\imagelabels')
label = label['labels'][0]
print(label.min())

'''
for id, i in enumerate(label):
    if i == 100:
        print(id)
'''


imgpath = 'C:\\Users\\Lam\\OneDrive - HKUST Connect\\Desktop\\Lecture Note\\CSIT5910\\Project\\data\\jpg'
imgsam = []
os.chdir(imgpath)
convert_tensor = transforms.ToTensor()
for _, _, filesname in os.walk(imgpath):
    for file in filesname:
        img = Image.open(file)
        imgsam.extend([convert_tensor(img)])

txtpath = 'C:\\Users\\Lam\\OneDrive - HKUST Connect\\Desktop\\Lecture Note\\CSIT5910\\Project\\text'

os.chdir(txtpath)
txtdes = []
for _, _, filesname in os.walk(txtpath):
    for file in filesname:
        with open(file, 'r') as f:
            txtdes.extend([f.read().replace('\n',' ')])

train_txt, temp_txt, train_img, temp_img = train_test_split(txtdes, imgsam, test_size=0.5, random_state=2022)
val_txt, test_txt, val_img, test_img = train_test_split(temp_txt, temp_img, train_size=0.05, test_size=0.05, random_state=2022)

max_seq_len = 150


'''
Loading the BERT Model for the text embedding...

'''

# import BERT-base pretrained model
bert = AutoModel.from_pretrained('bert-base-uncased')

# Load the BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')


# tokenize and encode sequences in the training set
tokens_train = tokenizer.batch_encode_plus(
    train_txt,
    max_length = max_seq_len,
    pad_to_max_length=True,
    truncation=True,
    return_token_type_ids=False
)

# tokenize and encode sequences in the validation set
tokens_val = tokenizer.batch_encode_plus(
    val_txt,
    max_length = max_seq_len,
    pad_to_max_length=True,
    truncation=True,
    return_token_type_ids=False
)

# tokenize and encode sequences in the test set
tokens_test = tokenizer.batch_encode_plus(
    test_txt,
    max_length = max_seq_len,
    pad_to_max_length=True,
    truncation=True,
    return_token_type_ids=False
)


# for train set
train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])

# for validation set
val_seq = torch.tensor(tokens_val['input_ids'])
val_mask = torch.tensor(tokens_val['attention_mask'])

# for test set
test_seq = torch.tensor(tokens_test['input_ids'])
test_mask = torch.tensor(tokens_test['attention_mask'])

# Stack for image Tensor
train_img = torch.stack(train_img)
val_img = torch.stack(val_img)
test_img = torch.stack(test_img)

train_data = TensorDataset(train_seq, train_mask, train_img)
val_data = TensorDataset(val_seq, val_mask, val_img)
test_data = TensorDataset(test_seq, test_mask, test_img)

batchsize = 20

train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batchsize,drop_last=True)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batchsize)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batchsize)

class BERT_Arch(nn.Module):
    def __init__(self, bert, nooffeatures):
        super(BERT_Arch, self).__init__()
        self.bert = bert
        # dropout layer
        self.dropout = nn.Dropout(0.1)
        # relu activation function
        self.relu = nn.ReLU()
        # dense layer 1
        self.fc1 = nn.Linear(768,512)
        # dense layer 2 (Output layer)
        self.fc2 = nn.Linear(512,nooffeatures)
        self.softmax = nn.LogSoftmax(dim=1)

    #define the forward pass
    def forward(self, sent_id, mask):
        #pass the inputs to the model
        _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)
        x = self.fc1(cls_hs)
        #x dim 512
        x = self.relu(x)
        x = self.dropout(x)
        # output layer
        x = self.fc2(x)
        return x

noofoutfeature = 1000
imgmodel = models.resnet18(weights='DEFAULT')
infeature_ResNet = imgmodel.fc.in_features
imgmodel.fc = nn.Linear(infeature_ResNet, noofoutfeature)
imgmodel = imgmodel.cuda()
txtmodel = BERT_Arch(bert, noofoutfeature)
txtmodel = txtmodel.cuda()

learning_rate = 1e-5

optimizer = optim.AdamW([{'params': txtmodel.parameters()},
                {'params': imgmodel.parameters()}], lr=learning_rate)

Epochs = 20
val_loss = []

for param in bert.parameters():
    param.requires_grad = True


pdist = nn.PairwiseDistance(p=2)

def loss_func(txt, img):

    samepairdist=torch.tensor(0)

    diffpairdist=torch.tensor(1e-10)

    truthcounter=0
    falsecounter=0

    for titem, tsample in enumerate(txt):
        for iitem, isample in enumerate(img):
            if titem == iitem:
                samepairdist = samepairdist+pdist(tsample, isample).sum()
                truthcounter+=1
            else:
                diffpairdist = diffpairdist+pdist(tsample, isample).sum()
                falsecounter+=1
    final_loss = samepairdist * falsecounter / diffpairdist / truthcounter

    return final_loss


def train():
    imgmodel.train()
    txtmodel.train()

    total_loss = 0
    for step, batch in enumerate(train_dataloader):

        batch = [r.to(device) for r in batch]
        seq, mask, image_tensor = batch

        imgmodel.zero_grad()
        txtmodel.zero_grad()

        txtout = txtmodel(seq, mask)
        imgout = imgmodel(image_tensor)


        loss = loss_func(txtout, imgout)
        loss = loss.sum()


        del txtout
        del imgout

        loss.backward()
        optimizer.step()

        optimizer.zero_grad()

        total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)

    return avg_loss

def evaluate():
    imgmodel.eval()
    txtmodel.eval()
    total_loss = 0
    for step, batch in enumerate(val_dataloader):
        batch = [t.to(device) for t in batch]
        seq, mask, image_tensor = batch

        with torch.no_grad():
            txtout = txtmodel(seq, mask)
            imgout = imgmodel(image_tensor)

            loss = loss_func(txtout, imgout)
            loss = loss.sum()
            total_loss = total_loss+loss.item()

            del txtout,imgout

    avg_loss = total_loss / len(val_dataloader)

    return avg_loss

def test():
    imgmodel.eval()
    txtmodel.eval()

    for step, batch in enumerate(test_dataloader):
        batch = [t.to(device) for t in batch]
        seq, mask, image_tensor = batch

        with torch.no_grad():
            txtout = txtmodel(seq, mask)
            imgout = imgmodel(image_tensor)

        txtout.detach().cpu()
        imgout.detach().cpu()

        if step == 0:
            outtxt = txtout
            outimg = imgout
        else:
            outtxt = torch.cat((outtxt, txtout))
            outimg = torch.cat((outimg, imgout))

        print(outtxt.sum().item())

        check = []
        for txtitem, txtsample in enumerate(outtxt):
            smallestdist = float('inf')
            for imgitem, imgsample in enumerate(outimg):
                if imgitem == txtitem:
                    truthdist = pdist(imgsample, txtsample).sum().item()
                if pdist(txtsample, imgsample).sum().item() < smallestdist:
                    imgcheck = imgitem
                    smallestdist = pdist(txtsample, imgsample).sum().item()
            check.append([txtitem, imgcheck, True if txtitem == imgcheck else False, truthdist, smallestdist])
    return check

best_val_loss = float('inf')

modelpath = 'C:\\Users\\Lam\\OneDrive - HKUST Connect\\Desktop\\Lecture Note\\CSIT5910\\Project'
os.chdir(modelpath)

elapsedtime = time.time()
for epoch in range(Epochs):

    print('Epoch %d' % (epoch+1))
    train_loss = train()
    eval_loss = evaluate()

    if eval_loss<best_val_loss:
        best_val_loss = eval_loss
        print('Saving Model')
        torch.save(txtmodel, 'Text_Model.pt')
        torch.save(imgmodel, 'Image_Model.pt')

    print('Training Loss: %.5f' % train_loss)
    print('Validation Loss: %.5f' % eval_loss)

elapsedtime = time.time()-elapsedtime
print('Running Time: %d min %d sec' % (elapsedtime//60, elapsedtime % 60))

check = test()
df = pd.DataFrame(check)
df.to_csv('E:\\predresult.csv')


