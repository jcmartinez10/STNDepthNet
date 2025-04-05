import itertools
import os
import shutil
import cv2
import os
import json
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision.utils as vutils
from torchvision import transforms, models
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
from matplotlib import colormaps
from PIL import Image
from TinyUNetArch import TinyUNet
import time




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()


class SegmentationDataset(Dataset):
   def __init__(self, data_dir, n_epochs=100, transform=None):

     
      self.data_dir = data_dir
      self.transform = transform
      self.file_names=[]

      self.file_names=os.listdir(self.data_dir)[0:9000:69]
           

##      print(len(self.file_names))

   def __len__(self):
      return len(self.file_names)

   def __getitem__(self, idx):
        
      if torch.is_tensor(idx):
         idx = idx.tolist()

      img_name = self.data_dir+'/'+self.file_names[idx]


      raw_name = img_name.replace('color','sparse').replace('png','npy')

      gt_name = img_name.replace('color','dense').replace('png','npy')


      #image=cv2.imread(img_name)[:,7:1175,:]
      image=cv2.imread(img_name)[:192,264:904,:]
      image=cv2.resize(image, (640, 192))
      image = image/255

      
      sparse_data=np.load(raw_name)
      #sparse_data=sparse_data[:,7:1175]
      sparse_data=sparse_data[:192,264:904]
      sparse_data=cv2.resize(sparse_data, (640, 192))


      gt_tensor=np.load(gt_name)
      #gt_tensor=gt_tensor[:,7:1175]
      gt_tensor=gt_tensor[:192,264:904]
      gt_tensor=cv2.resize(gt_tensor, (640, 192))
      
      image_tensor=torch.from_numpy(image.transpose((2, 0, 1)))
      sparse_tensor=torch.unsqueeze(torch.from_numpy(sparse_data),0)
      gt_tensor=torch.unsqueeze(torch.from_numpy(gt_tensor),0)
      
      sample = {'rgb': image_tensor.to(device),'sparse': sparse_tensor.to(device), 'dense': gt_tensor.to(device)}

      if self.transform:
         sample = self.transform(sample)

      return sample



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Spatial transformer color
        self.rgb_localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        
        # Spatial transformer sparse
        self.sparse_localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(137280, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        self.unet=TinyUNet(4,1)

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, rgb,sparse):
       
        #print(near.size())
        x_rgb=self.rgb_localization(rgb)
        x_sparse=self.sparse_localization(sparse)

        xs = torch.cat((x_rgb,x_sparse),dim=1)
        xs = xs.view(-1, 137280)
        
        theta = self.fc_loc(xs)
##        print('theta',theta.size())
        theta = theta.view(-1, 2, 3)

##        print('rgb',rgb.size())
##        print('theta',theta[0])

        grid = F.affine_grid(theta, rgb.size())
        xt = F.grid_sample(rgb, grid)

        return xt

    def forward(self, rgb, sparse):
        # transform the input
        xh = self.stn(rgb, sparse)
        x=torch.cat((xh,sparse),dim=1)
        x = self.unet(x)
        return x,xh

model = Net().to(device)
model.load_state_dict(torch.load('chimera.mod'))

train_dataset = SegmentationDataset(data_dir=r'D:\KITTI\train\color')
train_loader = DataLoader(train_dataset, batch_size=2,
                        shuffle=True, num_workers=0)




criterion = nn.MSELoss() #Función de pérdida MSE
optimizer = optim.Adam(model.parameters(), lr=1e-5)

epochs=100
train_losses=[]
val_losses=[]
for epoch in range(epochs):
    model.train()
    start_time=time.time()
    for batch_idx, sample_batched in enumerate(train_loader):
        print(batch_idx)
        rgb = sample_batched['rgb'].float()
        sparse=sample_batched['sparse'].float()
        dense=sample_batched['dense'].float()
       

        optimizer.zero_grad()

        recon_batch, delta_batch = model(rgb,sparse)

##        print(recon_batch.size())
##        print(dense.size())

        loss = criterion(recon_batch, dense)#+0.5*criterion(delta_batch, delta_in)

        loss.backward()
        optimizer.step()

##        print('shape',pred.shape)

    train_loss=loss.item()
    train_losses.append(train_loss)

    if (epoch+1) % 10== 0:
        torch.save(model.state_dict(), 'stn_chimera.mod')

    pred = recon_batch[0].cpu().detach().numpy()[0]
    pred[pred < 0] = 0
    pred[pred > 1] = 1
    pred=255*pred
    

    lab = dense[0].cpu().detach().numpy()[0]
    lab[lab < 0] = 0
    lab[lab > 1] = 1

    lab=255*lab

    #print(np.max(pred))
    
    img=255*rgb[0].cpu().detach().numpy().transpose((1, 2, 0))[:,:,:3]
    homography=255*delta_batch[0].cpu().detach().numpy().transpose((1, 2, 0))[:,:,:3]


    
    #print('shape',pred.shape, lab.shape,img.shape)


    if epoch % 1 == 0:
        print("Epoch: ", epoch+1,"Train loss :",train_loss,"Time:",time.time()-start_time)
        cv2.imwrite(os.path.join(os.getcwd(),"ntest/"+str(epoch+1)+"base.png"), img)
        cv2.imwrite(os.path.join(os.getcwd(),"ntest/"+str(epoch+1)+"cbase.png"), homography.astype(np.uint8))
        cv2.imwrite(os.path.join(os.getcwd(),"ntest/"+str(epoch+1)+"guess.png"), pred.astype(np.uint8))
        cv2.imwrite(os.path.join(os.getcwd(),"ntest/"+str(epoch+1)+"true.png"), lab.astype(np.uint8))
        




print("Loss chart:")
e=range(epochs)
#plt.plot(e, val_losses, 'r') # plotting t, a separately
plt.plot(e, train_losses, 'b') # plotting t, b separately
plt.show()
