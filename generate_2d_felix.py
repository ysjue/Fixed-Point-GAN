#%%
from utils.felix_dataloader import datasets
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import os 
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.data_process import get_ND_bounding_box, resize_ND_volume_to_given_shape

data_root = '/medical_backup/NII_data/felix/'
# txt_root = '/data/shuojue/code/Fixed-Point-GAN/split/valid_set.txt'
txt_root = '/data/shuojue/code/Fixed-Point-GAN/split/valid_set1.txt'
img_pos_file = '/data/shuojue/data/felix/valid/positive_img'
img_neg_file = '/data/shuojue/data/felix/valid/negative_img'
lab_pos_file = '/data/shuojue/data/felix/valid/positive_lab'
lab_neg_file = '/data/shuojue/data/felix/valid/negative_lab'
trainset = datasets(train=True, 
         data_root = data_root,
         txt_path = txt_root,
         crop_size = [None, 54 , 81]
         )
train_dataloader = DataLoader(trainset, batch_size = 1,
                       shuffle = True, num_workers = 8)
missing_data_list = ['FELIX5297_VENOUS.nii.gz', 'FELIX5931_ARTERIAL.nii.gz', 'FELIX5369_ARTERIAL.nii.gz', 
                     'FELIX-CYS-1003_VENOUS.nii.gz', 'FELIX5222_ARTERIAL.nii.gz', 'FELIX5490_ARTERIAL.nii.gz', 
                     'FELIX-Cys-1229_VENOUS.nii.gz', 'FELIX-Cys-1136_VENOUS.nii.gz', 'FELIX5263_ARTERIAL.nii.gz', 
                     'FELIX5224_VENOUS.nii.gz', 'FELIX-CYS-1003_ARTERIAL.nii.gz', 'FELIX-Cys-1432_VENOUS.nii.gz', 
                     'FELIX-CYS-1302_VENOUS.nii.gz', 'FELIX-Cys-1020_ARTERIAL.nii.gz', 'FELIX0442_ARTERIAL.nii.gz', 
                     'FELIX-Cys-1432_ARTERIAL.nii.gz', 'FELIX5145_ARTERIAL.nii.gz', 'FELIX0181_ARTERIAL.nii.gz', 
                     'FELIX-Cys-1222_VENOUS.nii.gz', 'FELIX-Cys-1020_VENOUS.nii.gz', 'FELIX5046_ARTERIAL.nii.gz',
                     'FELIX-Cys-1222_ARTERIAL.nii.gz', 'FELIX-CYS-1289_VENOUS.nii.gz', 'FELIX5366_ARTERIAL.nii.gz',
                     'FELIX5658_VENOUS.nii.gz', 'FELIX5222_VENOUS.nii.gz', 'FELIX5831_ARTERIAL.nii.gz']
for ii,(data, label, imgdir, min_idx, max_idx, data_error) in tqdm(enumerate(train_dataloader)):
    name = imgdir[0].split('/')[-1]
    name = name.split('.nii.gz')[0]
    if data_error:
        continue
    crop_size_2d = [54  , 81]
    resize_size_2d = [128 ,  192]
    data = data[0,0].numpy()
    label = label[0,0].numpy()
    D,H,W = label.shape
    for i in range(D):
        img = data[i]
        lab = label[i]
        new_name = name+'_{0:}.jpg'.format(i)
        tumor_mask = lab >= 3
        pancreas_mask = lab == 1
        # mask = tumor_mask + pancreas_mask
        mask = pancreas_mask
        
        img = resize_ND_volume_to_given_shape(img, resize_size_2d, 3)
        lab = resize_ND_volume_to_given_shape(lab, resize_size_2d, 0)
        img[img > 1] = 1.0
        img[img < 0] = 0.0
        # img = np.repeat(img[:,:, None], 3, axis=0)
        # img = np.transpose(img,[1,0,2])
        # img = np.array(img * (2.0**16-1), np.uint16)
        # cv2.imwrite(os.path.join(img_pos_file, new_name), img)
        
        # plt.show()
        # img = np.array(img*2.0**8, np.float32)
        # img = np.repeat(img[:,:, None] * 255.0, 3, axis=2)
        # # print(img.shape)
        img = np.array(img * 255.0, dtype=np.uint8)
        if np.sum(tumor_mask) > 0:
            img_file_path = img_pos_file
            lab_file_path = lab_pos_file
        else:
            img_file_path = img_neg_file
            lab_file_path = lab_neg_file
        Image.fromarray(img).save(os.path.join(img_file_path, new_name))
        
        Image.fromarray(lab).save(os.path.join(lab_file_path, new_name))
        # img = np.repeat(img[:,:,None], 3, axis=2)
        # min_idx, max_idx = get_ND_bounding_box(mask, 0)
        # min_idx = np.array(min_idx, np.int32)
        # max_idx = np.array(max_idx, np.int32)
        # size = max_idx - min_idx
        
    
    
    

    
# %%
