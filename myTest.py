import os
import glob
import time
import math
import torch
import pandas as pd
import numpy as np
from torch import nn
from PIL import Image
import torchvision as tv
import skimage.transform
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from myModel import get_mse_weighted_loss , DeepVo ,load_pretrain_weight

from helper import eulerAnglesToRotationMatrix, normalize_angle_delta

class ImageSequenceDataset(Dataset):
    def __init__(self, info_dataframe, resize_mode='crop', new_sizeize=None, img_mean=None, img_std=(1,1,1), minus_point_5=False):
        # Transforms
        transform_ops = []
        if resize_mode == 'crop':
            transform_ops.append(tv.transforms.CenterCrop((new_sizeize[0], new_sizeize[1])))
        elif resize_mode == 'rescale':
            transform_ops.append(tv.transforms.Resize((new_sizeize[0], new_sizeize[1])))
        transform_ops.append(tv.transforms.ToTensor())
        #transform_ops.append(transforms.Normalize(mean=img_mean, std=img_std))
        self.transformer = tv.transforms.Compose(transform_ops)
        self.minus_point_5 = minus_point_5
        self.normalizer = tv.transforms.Normalize(mean=img_mean, std=img_std)
        
        self.data_info = info_dataframe
        self.seq_len_list = list(self.data_info.seq_len)
        self.image_arr = np.asarray(self.data_info.image_path)  # image paths
        self.groundtruth_arr = np.asarray(self.data_info.pose)

    def __getitem__(self, index):
        raw_groundtruth = np.hsplit(self.groundtruth_arr[index], np.array([6]))	
        groundtruth_sequence = raw_groundtruth[0]
        groundtruth_rotation = raw_groundtruth[1][0].reshape((3, 3)).T # opposite rotation of the first frame
        groundtruth_sequence = torch.FloatTensor(groundtruth_sequence)
        # groundtruth_sequence[1:] = groundtruth_sequence[1:] - groundtruth_sequence[0:-1]  # get relative pose w.r.t. previois frame 

        groundtruth_sequence[1:] = groundtruth_sequence[1:] - groundtruth_sequence[0] # get relative pose w.r.t. the first frame in the sequence 
		
        # print('Item before transform: ' + str(index) + '   ' + str(groundtruth_sequence))

        # here we rotate the sequence relative to the first frame
        for gt_seq in groundtruth_sequence[1:]:
            location = torch.FloatTensor(groundtruth_rotation.dot(gt_seq[3:].numpy()))
            gt_seq[3:] = location[:]
            # print(location)

        # get relative pose w.r.t. previous frame
        groundtruth_sequence[2:] = groundtruth_sequence[2:] - groundtruth_sequence[1:-1]

		# here we consider cases when rotation angles over Y axis go through PI -PI discontinuity
        for gt_seq in groundtruth_sequence[1:]:
            gt_seq[0] = normalize_angle_delta(gt_seq[0])
			
        # print('Item after transform: ' + str(index) + '   ' + str(groundtruth_sequence))

        image_path_sequence = self.image_arr[index]
        sequence_len = torch.tensor(self.seq_len_list[index])  #sequence_len = torch.tensor(len(image_path_sequence))
        
        image_sequence = []
        for img_path in image_path_sequence:
            img_as_img = Image.open(img_path)
            img_as_tensor = self.transformer(img_as_img)
            if self.minus_point_5:
                img_as_tensor = img_as_tensor - 0.5  # from [0, 1] -> [-0.5, 0.5]
            img_as_tensor = self.normalizer(img_as_tensor)
            img_as_tensor = img_as_tensor.unsqueeze(0)
            image_sequence.append(img_as_tensor)
        image_sequence = torch.cat(image_sequence, 0)
        return (sequence_len, image_sequence, groundtruth_sequence)

    def __len__(self):
        return len(self.data_info.index)

def get_data_info(img_root,pose_root,folder_list, seq_len_range, overlap, sample_times=1, pad_y=False, shuffle=False, sort=True):
    X_path, Y = [], []
    X_len = []
    for folder in folder_list:
        start_t = time.time()
        poses = np.load('{}.npy'.format(os.path.join(pose_root, folder))) # (n_images, 6)
        fpaths = glob.glob('{}/*.png'.format(os.path.join(img_root, folder)))
        fpaths.sort()
        # Fixed seq_len
        if seq_len_range[0] == seq_len_range[1]:
            if sample_times > 1:
                sample_interval = int(np.ceil(seq_len_range[0] / sample_times))
                start_frames = list(range(0, seq_len_range[0], sample_interval))
                print('Sample start from frame {}'.format(start_frames))
            else:
                start_frames = [0]

            for st in start_frames:
                seq_len = seq_len_range[0]
                n_frames = len(fpaths) - st
                jump = seq_len - overlap
                res = n_frames % seq_len
                if res != 0:
                    n_frames = n_frames - res
                x_segs = [fpaths[i:i+seq_len] for i in range(st, n_frames, jump)]
                y_segs = [poses[i:i+seq_len] for i in range(st, n_frames, jump)]
                Y += y_segs
                X_path += x_segs
                X_len += [len(xs) for xs in x_segs]
        # Random segment to sequences with diff lengths
        else:
            assert(overlap < min(seq_len_range))
            n_frames = len(fpaths)
            min_len, max_len = seq_len_range[0], seq_len_range[1]
            for i in range(sample_times):
                start = 0
                while True:
                    n = np.random.random_integers(min_len, max_len)
                    if start + n < n_frames:
                        x_seg = fpaths[start:start+n] 
                        X_path.append(x_seg)
                        if not pad_y:
                            Y.append(poses[start:start+n])
                        else:
                            pad_zero = np.zeros((max_len-n, 15))
                            padded = np.concatenate((poses[start:start+n], pad_zero))
                            Y.append(padded.tolist())
                    else:
                        print('Last %d frames is not used' %(start+n-n_frames))
                        break
                    start += n - overlap
                    X_len.append(len(x_seg))
        print('Folder {} finish in {} sec'.format(folder, time.time()-start_t))
    
    # Convert to pandas dataframes
    data = {'seq_len': X_len, 'image_path': X_path, 'pose': Y}
    df = pd.DataFrame(data, columns = ['seq_len', 'image_path', 'pose'])
    # Shuffle through all videos
    if shuffle:
        df = df.sample(frac=1)
    # Sort dataframe by seq_len
    if sort:
        df = df.sort_values(by=['seq_len'], ascending=False)
    return df

class Config(object):

    num_workers = 4
    batch_size  = 8

    test_video = ['04', '05', '07', '10', '09'] # ['00', '01', '02'] 
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

    save_dir = 'result/'

    seq_len = (5, 7)

if __name__ == '__main__':    

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(1214)
    np.random.seed(1214)
    device = torch.device("cuda" if use_cuda else "cpu")
    print('===========Device used :', device,'===========')

    opts = Config()

    if not os.path.exists(opts.save_dir):
        os.makedirs(opts.save_dir, exist_ok=True)

    #############################################################################################################################################
    
    # model 創建
    model = DeepVo(opts.img_new_size[0],opts.img_new_size[1],frame=opts.cut_size,hidden_size=opts.hidden_size)

    map_location = lambda storage, loc: storage
    print('===========loading DeepVo Model :', os.path.join(opts.model_dir,opts.model_name) ,'===========')
    model.load_state_dict(torch.load(os.path.join(opts.model_dir,opts.model_name), map_location=map_location))
    model = model.to(device)
    
    #############################################################################################################################################

	# Data
    seq_len = int((opts.seq_len[0]+opts.seq_len[1])/2)
    overlap = seq_len - 1
    print('seq_len = {},  overlap = {}'.format(seq_len, overlap))

    fd=open('test_dump.txt', 'w')
    fd.write('\n'+'='*50 + '\n')

    for test_video in opts.test_video:
        
        df = get_data_info(opts.img_root,opts.gt_root,folder_list=[test_video], seq_len_range=[seq_len, seq_len], overlap=overlap, sample_times=1, shuffle=False, sort=False)
        df = df.loc[df.seq_len == seq_len]  # drop last
        dataset = ImageSequenceDataset(df, 'rescale', opts.img_new_size, opts.img_mean, opts.img_std, opts.minus_point_5)
        df.to_csv('test_df.csv')
        dataloader = DataLoader(dataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.num_workers)
        
        gt_pose = np.load('{}.npy'.format(os.path.join(opts.gt_root, test_video))) # (n_images, 6)

        # Predict
        model.eval()
        has_predict = False
        answer = [[0.0]*6, ]
        st_t = time.time()
        n_batch = len(dataloader)

        for i, batch in enumerate(dataloader):
            with torch.no_grad():
                print('{} / {}'.format(i, n_batch), end='\r', flush=True)
                _, x, y = batch
                if use_cuda:
                    x = x.cuda()
                    y = y.cuda()
                batch_predict_pose = model(x)

            # Record answer
            fd.write('Batch: {}\n'.format(i))
            for seq, predict_pose_seq in enumerate(batch_predict_pose):
                for pose_idx, pose in enumerate(predict_pose_seq):
                    fd.write(' {} {} {}\n'.format(seq, pose_idx, pose))


            batch_predict_pose = batch_predict_pose.data.cpu().numpy()
            if i == 0:
                for pose in batch_predict_pose[0]:
                    # use all predicted pose in the first prediction
                    for i in range(len(pose)):
                        # Convert predicted relative pose to absolute pose by adding last pose
                        pose[i] += answer[-1][i]
                    answer.append(pose.tolist())
                batch_predict_pose = batch_predict_pose[1:]

            # transform from relative to absolute 
            
            for predict_pose_seq in batch_predict_pose:
                # predict_pose_seq[1:] = predict_pose_seq[1:] + predict_pose_seq[0:-1]
                ang = eulerAnglesToRotationMatrix([0, answer[-1][0], 0]) #eulerAnglesToRotationMatrix([answer[-1][1], answer[-1][0], answer[-1][2]])
                location = ang.dot(predict_pose_seq[-1][3:])
                predict_pose_seq[-1][3:] = location[:]

            # use only last predicted pose in the following prediction
                last_pose = predict_pose_seq[-1]
                for i in range(len(last_pose)):
                    last_pose[i] += answer[-1][i]
                # normalize angle to -Pi...Pi over y axis
                last_pose[0] = (last_pose[0] + np.pi) % (2 * np.pi) - np.pi
                answer.append(last_pose.tolist())

        print('len(answer): ', len(answer))
        print('expect len: ', len(glob.glob('{}/*.png'.format(os.path.join(opts.img_root, test_video)))))
        print('Predict use {} sec'.format(time.time() - st_t))


        # Save answer
        with open('{}/out_{}.txt'.format(opts.save_dir, test_video), 'w') as f:
            for pose in answer:
                if type(pose) == list:
                    f.write(', '.join([str(p) for p in pose]))
                else:
                    f.write(str(pose))
                f.write('\n')


        # Calculate loss
        gt_pose = np.load('{}.npy'.format(os.path.join(opts.gt_root, test_video)))  # (n_images, 6)
        loss = 0
        for t in range(len(gt_pose)):
            angle_loss = np.sum((answer[t][:3] - gt_pose[t,:3]) ** 2)
            translation_loss = np.sum((answer[t][3:] - gt_pose[t,3:6]) ** 2)
            loss = (100 * angle_loss + translation_loss)
        loss /= len(gt_pose)
        print('Loss = ', loss)
        print('='*50)
