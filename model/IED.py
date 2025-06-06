import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class IED(nn.Module):

	def __init__(self):
		super(IED, self).__init__()

		self.relu = nn.ReLU(inplace=True)

		self.e_conv1 = nn.Conv2d(3,32,3,1,1,bias=True) 
		self.e_conv2 = nn.Conv2d(32,32,3,1,1,bias=True) 
		self.e_conv3 = nn.Conv2d(32,32,3,1,1,bias=True) 
		self.e_conv4 = nn.Conv2d(32,32,3,1,1,bias=True)#64 32 128 256

		self.e_conv5 = nn.Conv2d(64,32,3,1,1,bias=True) 
		self.e_conv6 = nn.Conv2d(64,32,3,1,1,bias=True) 
		self.e_conv7 = nn.Conv2d(64,27,3,1,1,bias=True)

		self.channels = 3
		kernel = [[0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633],
				[0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
				[0.01330373, 0.11098164, 0.22508352, 0.11098164, 0.01330373],
				[0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
				[0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633]]
		kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
		self.kernel = kernel.repeat(self.channels, 1, 1, 1).cuda()

	def enhance(self, x,x_r):
		
		r1,r2,r3,r4,r5,r6,r7,r8,d1 = torch.split(x_r, 3, dim=1)
		x = x + r1*(torch.pow(x,2)-x)
		x = x + r2*(torch.pow(x,2)-x)
		x = x + r3*(torch.pow(x,2)-x)
		enhance_image_1 = x + r4*(torch.pow(x,2)-x)		
		x = enhance_image_1 + r5*(torch.pow(enhance_image_1,2)-enhance_image_1)		
		x = x + r6*(torch.pow(x,2)-x)	
		x = x + r7*(torch.pow(x,2)-x)
		enhance_image = x + r8*(torch.pow(x,2)-x)

		x = F.conv2d(enhance_image, self.kernel, padding=2, groups=self.channels)
		enhance_image_d = (enhance_image - x) * 2 * torch.tanh(d1) + enhance_image

		return enhance_image,enhance_image_d


	def forward(self, x):

		x1 = self.relu(self.e_conv1(x))
		x2 = self.relu(self.e_conv2(x1))
		x3 = self.relu(self.e_conv3(x2))
		x4 = self.relu(self.e_conv4(x3))
		x5 = self.relu(self.e_conv5(torch.cat([x3,x4],1)))
		x6 = self.relu(self.e_conv6(torch.cat([x2,x5],1)))
		x_r = torch.tanh(self.e_conv7(torch.cat([x1,x6],1)))

		enhance_image,enhance_image_d = self.enhance(x,x_r)

		return enhance_image,enhance_image_d,x_r[:,:24]
