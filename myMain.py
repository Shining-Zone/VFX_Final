import os
import math
import torch
import pickle
import argparse
import numpy as np
from torch import nn
import torchvision as tv
import skimage.transform
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from myDataloader import get_data_info , ImageSeqDataset
from myModel import get_mse_weighted_loss , DeepVo ,load_pretrain_weight

class Config(object):

    num_workers = 8
    epochs      = 250
    batch_size  = 8
    lr = 0.0005

    train_video = ['00', '01', '02', '05', '08', '09']
    valid_video = ['04', '06', '07', '10']
    img_root = './KITTI/images'
    gt_root  = './KITTI/pose_GT'

    cut_size = 7 #不要亂動 7就7
    overlap  = 1
    assert(cut_size>overlap)
    img_new_size = (304, 92)
    img_std  =  (0.2610784009469139, 0.25729316928935814, 0.25163823815039915)
    img_mean = (0.19007764876619865, 0.15170388157131237, 0.10659445665650864)
    minus_point_5 = True

    hidden_size = 1000

    model_dir  = 'model_para'

    pretrain_model_name = './flownets_bn_EPE2.459.pth.tar'  # 'flownets_EPE1.951.pth.tar'

    save_interval = 1

def test(opts):

    #############################################################################################################################################

    # model 創建
    model = DeepVo(opts.img_new_size[0],opts.img_new_size[1],frame=opts.cut_size,hidden_size=1000)
    load_pretrain_weight(model,opts.pretrain_model_name)
    model = model.to(device)

    #############################################################################################################################################

    # Dataloader創建
    transform1 = tv.transforms.Compose([tv.transforms.Resize(opts.img_new_size),tv.transforms.ToTensor()])
    transform2 =  tv.transforms.Compose([tv.transforms.Normalize(mean=opts.img_mean , std=opts.img_std)])

    train_data_information = get_data_info(opts.img_root,opts.gt_root,opts.train_video,opts.cut_size,opts.overlap,test=False)
    train_dataset = ImageSeqDataset(train_data_information,transform1,transform2,minus_point_5=opts.minus_point_5)
    train_dataloader = DataLoader(train_dataset,batch_size=opts.batch_size,shuffle=True,num_workers=opts.num_workers,drop_last=False)

    valid_data_information = get_data_info(opts.img_root,opts.gt_root,opts.valid_video,opts.cut_size,opts.overlap,test=True)
    valid_dataset = ImageSeqDataset(valid_data_information,transform1,transform2,minus_point_5=opts.minus_point_5)
    valid_dataloader = DataLoader(valid_dataset,batch_size=1,shuffle=False,num_workers=opts.num_workers,drop_last=False)     #這邊要注意!!! batch_size 為1

    #############################################################################################################################################

    if not os.path.exists(opts.model_dir):
        os.makedirs(opts.model_dir, exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters(),lr=opts.lr,betas=(0.9, 0.999))

    #開啟紀錄檔案
    f = open(os.path.join(opts.model_dir,'record.txt'),'w')

    for ep in range(opts.epochs):
        print('='*50)
        model.train()
        total_loss = 0.0
        for i ,(_,img,gt) in enumerate(train_dataloader):
            img , gt = img.to(device) , gt.to(device)

            optimizer.zero_grad()
            output = model(img)

            loss = get_mse_weighted_loss(output,gt)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print('Train Epoch : {}\t Average_Loss: {:.6f}'.format(ep,total_loss/len(train_dataloader)))

        model.eval()
        val_loss  = 0.0
        for i ,(_,v_img,v_gt) in enumerate(valid_dataloader):
            with torch.no_grad():
                v_img , v_gt = v_img.to(device) , v_gt.to(device)

                v_output = model(v_img)

                loss = get_mse_weighted_loss(v_output,v_gt)

                val_loss += loss.item()
        print('Valid Epoch : {}\t Average_Loss:  {:.6f}'.format(ep,val_loss/len(valid_dataloader)))

        if (ep+1) % opts.save_interval == 0:
            print('=========save the model at Epoch : ' + str(ep) + ' =========')
            #torch.save(model.state_dict(), os.path.join(opts.model_dir,'DeepVo_Epoch_{}.pth'.format(ep)))
            torch.save(model.state_dict(), os.path.join(opts.model_dir,'DeepVo_Epoch_best.pth'))
    #############################################################################################################################################
    #紀錄檔案
        f.write('='*50+'\n')
        f.write('Epoch {}\tTrain Average_Loss: {} Valid Average_Loss: {}\n'.format(ep, total_loss/len(train_dataloader), val_loss/len(valid_dataloader)))
    f.write('='*50+'\n')
    f.close()
    #############################################################################################################################################
    print('=========save the model at last Epoch   =========')
    torch.save(model.state_dict(), os.path.join(opts.model_dir,'DeepVo_Epoch_Last.pth'))

if __name__ == '__main__':

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(1214)
    np.random.seed(1214)
    device = torch.device("cuda" if use_cuda else "cpu")
    print('===========Device used :', device,'===========')

    opts = Config()

    test(opts)
