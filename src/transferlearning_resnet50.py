import os
import torch
import numpy as np
import torchvision
from torchvision import transforms,utils
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tensorboardX import SummaryWriter
#perpare data set
#train data

writer = SummaryWriter()
my_device="cuda:0"
train_data=torchvision.datasets.ImageFolder('./emotion/train',
            transform=transforms.Compose(
                                        [transforms.Scale(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                        [0.229, 0.224, 0.225])]))
print(len(train_data))
train_loader=DataLoader(train_data,batch_size=128,shuffle=True)
 
#test data
test_data=torchvision.datasets.ImageFolder('./emotion/val',
            transform=transforms.Compose([transforms.Scale(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]))
test_loader=DataLoader(test_data,batch_size=128,shuffle=True)
 
# prepare model
mode1_ft_res50=torchvision.models.resnet50(pretrained=True)
writer.add_graph(mode1_ft_res50,torch.rand([1,3,224,224]))
for param in mode1_ft_res50.parameters():
    param.requires_grad=False
num_fc=mode1_ft_res50.fc.in_features
mode1_ft_res50.fc=torch.nn.Linear(num_fc,7)
mode1_ft_res50 = mode1_ft_res50.to(device=my_device)
#mode1_ft_res50 = mode1_ft_res50.fc() 


#loss function and optimizer
criterion=torch.nn.CrossEntropyLoss()
#parameters only train the last fc layer
optimizer=torch.optim.Adam(mode1_ft_res50.fc.parameters(),lr=0.01，weight_decay=1e-3)
 
#start train
#label  not  one-hot encoder
EPOCH=200
for epoch in range(EPOCH):
    train_loss=0.
    train_acc=0.
    for step,data in enumerate(train_loader):
        batch_x,batch_y=data
        batch_x,batch_y=Variable(batch_x.to(device=my_device)),Variable(batch_y.to(device=my_device))
        #batch_y not one hot
        #out is the probability of eatch class
        # such as one sample[-1.1009  0.1411  0.0320],need to calculate the max index
        # out shape is batch_size * class
        out=mode1_ft_res50(batch_x)
        loss=criterion(out,batch_y)
        train_loss+=loss.item()
        # pred is the expect class
        #batch_y is the true label
        pred=torch.max(out,1)[1]
        train_correct=(pred==batch_y).sum()
        train_acc+=train_correct.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step%14==0:
            print('Epoch: ',epoch,'Step',step,
                  'Train_loss: ',train_loss/((step+1)*128),'Train acc: ',train_acc/((step+1)*128))
        writer.add_scalar('train',train_loss,epoch)
        writer.add_scalar('train',train_acc,epoch)

    #print('Epoch: ', epoch, 'Train_loss: ', train_loss / len(train_data), 'Train acc: ', train_acc / len(train_data))
 
 
# test model
mode1_ft_res50.eval()
eval_loss=0
eval_acc=0
for step ,data in enumerate(test_loader):
    batch_x,batch_y=data
    batch_x,batch_y=Variable(batch_x.to(device=my_device)),Variable(batch_y.to(device=my_device))
    out=mode1_ft_res50(batch_x)
    loss = criterion(out, batch_y)
    eval_loss += loss.item()
    # pred is the expect class
    # batch_y is the true label
    pred = torch.max(out, 1)[1]
    test_correct = (pred == batch_y).sum()
    eval_acc += test_correct.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print( 'Test_loss: ', eval_loss / len(test_data), 'Test acc: ', eval_acc / len(test_data))
    writer.add_scalar('eval',eval_loss,step)
    writer.add_scalar('eval',eval_acc,step)
    for name, param in mode1_ft_res50.named_parameters():
        writer.add_histogram(name,param.clone().data.cpu().data.numpy(),step)

#print(mode1_ft_res50)
torch.save(mode1_ft_res50.state_dict(), 'mode1_ft_res50.pth')
#mode1_ft_res50.load_state_dict(torch.load('mode1_ft_res50.pth'))

writer.export_scalars_to_json("./all_scalars.json")
writer.close()
