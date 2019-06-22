import os
import glob
import math
import torch
import numpy as np
from PIL import Image
import torchvision as tv
from torch.utils.data import Dataset , DataLoader
from helper import normalize_angle_delta

def cut_data(v_img_paths,v_gt_poses,slice_num,cut_size,overlap,test=False):
    cut_img_paths = []
    cut_gt_poses  = []
    if test ==True:
        cut_img_paths = np.array_split(v_img_paths,slice_num)
        cut_gt_poses  = np.array_split(v_gt_poses,slice_num)
    else:
        total_frame = v_img_paths.shape[0]
        end         = cut_size
        start       = 0
        while 1 :
            if end > total_frame:
                cut_img_paths.append(v_img_paths[-cut_size:])
                cut_gt_poses.append(v_gt_poses[-cut_size:])
                break
            cut_img_paths.append(v_img_paths[start:end])
            cut_gt_poses.append(v_gt_poses[start:end])
            start = end - overlap
            end   = start + cut_size
    cut_length_labels = [len(frame) for frame in cut_gt_poses]

    return cut_img_paths , cut_gt_poses , cut_length_labels

def get_data_info(img_root,gt_root,video_list,cut_size,overlap,test=False):

    n_img_paths = [np.array(sorted(glob.glob('{}/*.png'.format(os.path.join(img_root,img_kind))))) for img_kind in video_list ] 
    n_gt_poses  = [np.load('{}.npy'.format(os.path.join(gt_root,gt_kind))) for gt_kind in video_list ]

    n_cut_img_paths     = []
    n_cut_gt_poses      = []
    n_cut_length_labels = []
    assert(len(n_img_paths)==len(n_gt_poses)) #確定 poses類別數目與照片類別數目一致
    for kind , (v_img_paths , v_gt_poses) in enumerate(zip(n_img_paths,n_gt_poses)):
        ##################################################################################
        # if kind ==1:
        #     break
        ##################################################################################
        slice_num = math.ceil(v_img_paths.shape[0]/cut_size)
        cut_img_paths , cut_gt_poses , cut_length_labels = cut_data(v_img_paths,v_gt_poses,slice_num,cut_size,overlap,test=test)
        
        n_cut_img_paths     +=     cut_img_paths
        n_cut_gt_poses      +=      cut_gt_poses
        n_cut_length_labels += cut_length_labels

        #print(kind , v_img_paths.shape , v_gt_poses.shape)
    #print(len(n_cut_img_paths) , len(n_cut_gt_poses)  , n_cut_length_labels)

    return (n_cut_img_paths , n_cut_gt_poses  , n_cut_length_labels)

class ImageSeqDataset(Dataset):
    def __init__(self,data_info,transforms1=None,transforms2=None,minus_point_5=False):

        n_cut_img_paths     = data_info[0]
        n_cut_gt_poses      = data_info[1]
        n_cut_length_labels = data_info[2]

        self.image_arr      = np.asarray(n_cut_img_paths)
        self.groundtruth_arr= np.asarray(n_cut_gt_poses)
        self.seq_len_list   = n_cut_length_labels
        self.transforms1     = transforms1
        self.transforms2     = transforms2
        self.minus_point_5 = minus_point_5

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

        image_path_sequence = self.image_arr[index].tolist()
        sequence_len = torch.tensor(self.seq_len_list[index])  #sequence_len = torch.tensor(len(image_path_sequence))

        image_sequence = []
        for img_path in image_path_sequence:
            img_as_img = Image.open(img_path)
            img_as_tensor = self.transforms1(img_as_img)
            if self.minus_point_5:
                img_as_tensor = img_as_tensor - 0.5  # from [0, 1] -> [-0.5, 0.5]
            img_as_tensor = self.transforms2(img_as_tensor)
            img_as_tensor = img_as_tensor.unsqueeze(dim=0)
            image_sequence.append(img_as_tensor)
        image_sequence = torch.cat(image_sequence, dim=0)

        return (sequence_len, image_sequence, groundtruth_sequence)

    def __len__(self):
        return len(self.image_arr)

if __name__ == '__main__':

    train_video = ['00', '01', '02', '05', '08', '09']
    valid_video = ['04', '06', '07', '10']
    img_root = './KITTI/images'
    gt_root  = './KITTI/pose_GT' 

    cut_size = 7
    overlap  = 1
    assert(cut_size>overlap)
    img_new_size = (150, 600)
    img_std=(1,1,1)
    img_mean = (-0.14968217427134656, -0.12941663107068363, -0.1320610301921484)
    minus_point_5 = True
    #tv.transforms.CenterCrop(img_new_size) 可以代替 Resize
    transform1 = tv.transforms.Compose([tv.transforms.Resize(img_new_size),tv.transforms.ToTensor()])
    transform2 =  tv.transforms.Compose([tv.transforms.Normalize(mean=img_mean , std=img_std)])

    data_info = get_data_info(img_root,gt_root,train_video,cut_size,overlap,test=False)

    dataset = ImageSeqDataset(data_info,transform1,transform2,minus_point_5=minus_point_5)
    dataloader = DataLoader(dataset,batch_size=4,shuffle=False,num_workers=0,drop_last=False)

    for batch in dataloader:
        s, x, y = batch
        print('='*50)
        print('len:{}\nx:{}\ny:{}'.format(s, x.shape, y.shape))