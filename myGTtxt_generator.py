import numpy as np
import os

if not os.path.exists("GT_pose_rpyxyz"):
    os.makedirs("GT_pose_rpyxyz", exist_ok=True)

gt_root = './KITTI/pose_GT'
video_list = ["00","01","02","03","04","05","06","07","08","09","10"]
n_gt_poses  = [np.load('{}.npy'.format(os.path.join(gt_root,gt_kind))) for gt_kind in video_list ]
print(len(n_gt_poses))
print(len(n_gt_poses[4]))
print(len(n_gt_poses[4][0]))
print((n_gt_poses[4][0][:6]))

for i, n_gt_pose in enumerate(n_gt_poses):
    with open(os.path.join("GT_pose_rpyxyz","GT_"+video_list[i]+".txt"),'w') as f:
        for gt_post in n_gt_pose:
            #r,p,y,x,y,z
            f.write(str(gt_post[0])+', '+str(gt_post[1])+', '+str(gt_post[2])+', '+str(gt_post[3])+', '+str(gt_post[4])+', '+str(gt_post[5])+'\n')
