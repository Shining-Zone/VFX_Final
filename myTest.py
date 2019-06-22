import os
import math
import torch
import argparse
import numpy as np
from torch import nn
import torchvision as tv
import skimage.transform
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from myDataloader import get_data_info , ImageSeqDataset
from myModel import get_mse_weighted_loss , DeepVo ,load_pretrain_weight

class Config(object):

    num_workers = 8
    batch_size  = 1

    test_video = ['04', '06', '07', '10']
    img_root = './KITTI/images'
    gt_root  = './KITTI/pose_GT' 

    cut_size = 7 #不要亂動 7就7
    overlap  = 0
    assert(cut_size>overlap)
    img_new_size = (304, 92)
    img_std  =  (0.2610784009469139, 0.25729316928935814, 0.25163823815039915)
    img_mean = (0.19007764876619865, 0.15170388157131237, 0.10659445665650864)
    minus_point_5 = True

    hidden_size = 1000

    model_dir  = 'model_para'

    model_name = 'DeepVo_Epoch_Last.pth'

def test(opts):
    
    #############################################################################################################################################
    
    # model 創建
    model = DeepVo(opts.img_new_size[0],opts.img_new_size[1],frame=opts.cut_size,hidden_size=opts.hidden_size)

    map_location = lambda storage, loc: storage
    print('===========loading DeepVo Model :', os.path.join(opts.model_dir,opts.model_name) ,'===========')
    model.load_state_dict(torch.load(os.path.join(opts.model_dir,opts.model_name), map_location=map_location))
    model = model.to(device)
    
    #############################################################################################################################################
    
    # Dataloader創建
    transform1 = tv.transforms.Compose([tv.transforms.Resize(opts.img_new_size),tv.transforms.ToTensor()])
    transform2 =  tv.transforms.Compose([tv.transforms.Normalize(mean=opts.img_mean , std=opts.img_std)])

    data_information = get_data_info(opts.img_root,opts.gt_root,opts.test_video,opts.cut_size,opts.overlap,test=True)
    dataset = ImageSeqDataset(data_information,transform1,transform2,minus_point_5=opts.minus_point_5)
    dataloader = DataLoader(dataset,batch_size=opts.batch_size,shuffle=False,num_workers=opts.num_workers,drop_last=False)     #這邊要注意!!! batch_size 為1
    
    #############################################################################################################################################

    #開啟紀錄檔案
    f = open(os.path.join(opts.model_dir,'record.txt'),'w')

    print('='*50)
    model.eval()
    total_loss = 0.0
    for i ,(_,img,gt) in enumerate(dataloader):
        print('{} / {}'.format(i, len(dataloader)), end='\r', flush=True)
        with torch.no_grad():
            img , gt = img.to(device) , gt.to(device)

            output = model(img)
            
            loss = get_mse_weighted_loss(output,gt)

            total_loss += loss.item()
    print('Average_Loss: {:.6f}'.format(total_loss/len(dataloader)))   
        
    #############################################################################################################################################   
    #紀錄檔案  
    f.write('='*50 +'\n')
    f.write('Valid Average_Loss: {}\n'.format(total_loss/len(dataloader)))
    f.write('='*50+'\n')
    f.close()
    #############################################################################################################################################

if __name__ == '__main__':

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(1214)
    np.random.seed(1214)
    device = torch.device("cuda" if use_cuda else "cpu")
    print('===========Device used :', device,'===========')

    opts = Config()

    test(opts)

