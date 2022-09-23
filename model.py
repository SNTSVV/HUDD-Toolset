"""
created by: jun wang
the model size is around 2M
TODO: quantize and prune toe model

"""

import torch
from torch import nn, optim
import torch.nn.functional as F

class KPNet(nn.Module):
	def __init__(self):
		super(KPNet, self).__init__()
		in_channel = 1
		out_conv2d_1 = 64+32
		out_conv2d_2 = 64+32
		out_conv2d_3 = 128+64
		out_conv2d_4 = 128+64

		out_conv2d_5 = 32*5+64
		out_conv2d_6 = 32*5+64
		out_conv2d_trans_1 = 128
		n_class = 27 # the number of coordinates

		self.conv2d_1 = nn.Conv2d(in_channels=in_channel, out_channels=out_conv2d_1, kernel_size=3, padding=1)
		self.conv2d_2 = nn.Conv2d(in_channels=out_conv2d_1, out_channels=out_conv2d_2, kernel_size=3, padding=1)
		self.conv2d_3 = nn.Conv2d(in_channels=out_conv2d_2, out_channels=out_conv2d_3, kernel_size=3, padding=1)
		self.conv2d_4 = nn.Conv2d(in_channels=out_conv2d_3, out_channels=out_conv2d_4, kernel_size=3, padding=1)

		self.conv2d_5 = nn.Conv2d(in_channels=out_conv2d_4, out_channels=out_conv2d_5, kernel_size=3, padding=1)
		self.conv2d_6 = nn.Conv2d(in_channels=out_conv2d_5, out_channels=out_conv2d_6, kernel_size=1, padding=0)

		self.conv2d_trans_1 = nn.ConvTranspose2d(in_channels=out_conv2d_6, out_channels=out_conv2d_trans_1, kernel_size=2, stride=2,bias=False)
		self.conv2d_trans_2 = nn.ConvTranspose2d(in_channels=out_conv2d_trans_1, out_channels=n_class, kernel_size=2, stride=2,bias=False)
		return

	def forward(self,x):
		x = F.relu(self.conv2d_1(x))
		x = F.relu(self.conv2d_2(x))
		x = F.max_pool2d(x,kernel_size=2, stride=2)
		x = F.relu(self.conv2d_3(x))
		x = F.relu(self.conv2d_4(x))
		x = F.max_pool2d(x,kernel_size=2, stride=2)
		x = F.relu(self.conv2d_5(x))
		x = F.relu(self.conv2d_6(x))
		x = F.relu(self.conv2d_trans_1(x))
		x = self.conv2d_trans_2(x)
		#print(x.size())
		return x

if __name__ == '__main__':
	model = KPNet().cuda()
	print(model)
	summary(model, (1,128,128))