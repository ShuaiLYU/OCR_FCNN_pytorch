import  os
from PIL import Image
from torch.utils import data
import torch
class MyDataset(data.Dataset):
    def __init__(self,root,dic=None,transforms=None,vector_length=17):
        super(MyDataset, self).__init__()
        self.imgs = os.listdir(root)
        self.img_paths = [os.path.join(root,img) for img in self.imgs]
        self.transforms = transforms
        self.vector_length=vector_length #用来输出固定长度的标签（网络识别的字符串长度上限16+1）
        item=[('0',0),('1',1),('2',2),('3',3),('4',4),('5',5),('6',6),('7',7),('8',8),('9',9)]
        if dic is None:
            self.dic=dict(item)
        else:
            self.dic=dic

    def __getitem__(self,index):
        img_path = self.img_paths[index]
        #label = 1 if 'empty' in img_path.split('')[-1] else 0  #定义标签：图片名中有label
        label=self.imgs[index].split('.')[0].split('_')[1:]
        label=self.labelprocress(label)
        data = Image.open(img_path).convert("RGB")
        #data=cv2.imread(img_path,1)
        if self.transforms:
            data = self.transforms(data)
        return data,label

    def __len__(self):
        return len(self.imgs)

    def labelprocress(self,label):
        length =[int(label[0])]
        labels=[int(self.dic[str(x)]) for x in label[1]]
        vector=length+labels
        output=torch.zeros(self.vector_length)
        for i,data in  enumerate(vector):
            output[i]=data
        # vector=torch.ones([label[0]])
        # for i,item in  enumerate(labels):
        #     vector[2 * i] = item
        # label[1]=vector
        return output