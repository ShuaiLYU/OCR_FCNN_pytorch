import torch
import numpy as np
import time
from model_pt import  VGG16_conv
from myDataset import  MyDataset
from torchvision import transforms
import utils_ls
from torch.autograd import Variable
import  os

logger=utils_ls.get_logger()
save_path='./checkpoint_path'
if not os.path.exists(save_path):
	os.mkdir(save_path)
data_transform=transforms.Compose([transforms.Resize([32,256]),
                                    transforms.ToTensor()])
path_test='dataset12/test'
item = [('0', 0), ('1', 1), ('2', 2), ('3', 3), ('4', 4), ('5', 5), ('6', 6), ('7', 7), ('8', 8), ('9', 9),
        ('E', 10), ('H', 11)]
dic=dict(item)
def test():
	###模型初始化
	model = VGG16_conv(13)
	model.load_state_dict(torch.load(save_path + '/modelpara.pth'))
	Use_gpu = torch.cuda.is_available()
	if Use_gpu:
		model = model.cuda()

	###数据初始化
	dataset_test = MyDataset(path_test, dic=dic, transforms=data_transform)

	dataloader_test = torch.utils.data.DataLoader(dataset=dataset_test,
                                                   batch_size=1,  # batch_size 要=1
                                                   shuffle=True,
                                                   num_workers=0)

	time_open = time.time()


	length_acc=0
	count_test=0
	for batch, data in enumerate(dataloader_test, ):
		x, y = data
		y_length = torch.index_select(y, 1, torch.tensor([0])).squeeze_(1).long()
		label_index = [x + 1 for x in range(y_length)]
		y_labes = torch.index_select(y, 1, torch.tensor(label_index)).squeeze_(0).long()
		# [1,y_length] -> [y_length]
		y_labes = y_labes.squeeze_(0).long()
		if Use_gpu:
			x, y_length, y_labes = Variable(x.cuda()), Variable(y_length.cuda()), Variable(y_labes.cuda())
		else:
			x, y_length, y_labes = Variable(x), Variable(y_length), Variable(y_labes)
		model.dynamic_pooling(y_length)
		num_pred ,pred = model.forward(x,)

		list = [2 * i for i in range(num_pred)]
		split_labes = torch.index_select(pred, 0, torch.tensor(list).cuda())
		_, pred_labels = torch.max(split_labes, 1)


		if y_length==num_pred:
			length_acc+=1
			logger.info('当前图片length:{},label：{}'.format(y_length, y_labes))
			logger.info('识别结果length:{},label：{}'.format(num_pred, pred_labels))
		count_test+=1

	time_close = time.time()
	time_=(time_close-time_open)/count_test
	length_acc/=count_test
	logger.info('识别样本数量：{}，字符长度准确率：{},平均时间：{}'.format(count_test,length_acc,time_))

if __name__ == "__main__":
	test()