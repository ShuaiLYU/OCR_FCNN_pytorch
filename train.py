import torch
import numpy as np
import time
from model_pt import  VGG16_conv
from myDataset import  MyDataset
from torchvision import transforms
import utils_ls
from torch.autograd import Variable
import  os

#
# def main():
#     pass

logger=utils_ls.get_logger()
epoch_n=1000
save_path='./checkpoint_path'
if not os.path.exists(save_path):
    os.mkdir(save_path)
data_transform=transforms.Compose([transforms.Resize([32,256]),
                                    transforms.ToTensor()])
path_train='dataset12/train'
path_valid='dataset12/valid'
item = [('0', 0), ('1', 1), ('2', 2), ('3', 3), ('4', 4), ('5', 5), ('6', 6), ('7', 7), ('8', 8), ('9', 9),
        ('E', 10), ('H', 11)]
dic=dict(item)
dataset_train=MyDataset(path_train,dic=dic,transforms=data_transform)
dataset_valid=MyDataset(path_valid,dic=dic,transforms=data_transform)
dataloader_train = torch.utils.data.DataLoader(dataset=dataset_train,
                                            batch_size=1,   #batch_size 要=1
                                            shuffle=True,
                                            num_workers=0)

#


#x_example,y_example=next(iter(dataset_train))

model=VGG16_conv(13)
loss_f=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.recognition_character.parameters(),lr=0.000001)
Use_gpu=torch.cuda.is_available()
if Use_gpu:
    model=model.cuda()

time_open=time.time()

for epoch in range(epoch_n):

    for batch,data in enumerate(dataloader_train,):
        x,y=data
        y_length=y[0]
        y_labes=y[1]
        #[1,y_length] -> [y_length]
        y_labes=y_labes.squeeze_(0).long()
        if Use_gpu:
            x, y_length,y_labes=Variable(x.cuda()),Variable(y_length.cuda()),Variable(y_labes.cuda())
        else:
            x, y_length, y_labes = Variable(x), Variable(y_length), Variable(y_labes)
        model.dynamic_pooling(y_length)
        pred=model.forward(x,y_length)

        list=[2*i for i in range(y_length)]
        split_labes=torch.index_select(pred,0,torch.tensor(list).cuda())
        _, pred_labels = torch.max(split_labes, 1)
        #split_labes=torch.split(pred,list,dim=0)
        optimizer.zero_grad()
        loss=loss_f(split_labes,y_labes)
        running_loss = loss.item()
        # print(running_loss)
        # print(y_labes)
        # print(pred_labels)

        logger.info('epoch:{},batch:{},loss:{}'.format(epoch,batch,running_loss))
        if batch%10==0:
            logger.info('当前图片label：{}'.format( y_labes))
            logger.info('识别结果label：{}'.format(pred_labels))
        loss.backward()
        optimizer.step()
    #模型持久化
    if epoch%10==1 or epoch==epoch_n-1:
        # state = {'net ':model.state_dict(),
        #          'ptimizer':optimizer.state_dict(),
        #          'epoch ':epoch}
        time_now = time.time()
        logger.info('time:{}saving checkpoint.'.format(time_now))
        torch.save(model.state_dict(),save_path+'/modelpara.pth')