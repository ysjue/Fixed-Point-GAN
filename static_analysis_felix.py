from statistics import median
from utils.data_process import load_3d_volume_as_array, get_ND_bounding_box
import os
import numpy as np
import tqdm

label_path = '/medical_backup/NII_data/felix/label_6cls'
# label_path = '/data/shuojue/code/Fixed-Point-GAN/images'
dst_path = '/data/shuojue/code/Fixed-Point-GAN/utils/bbox.npy'
files = os.listdir(label_path)
files = [f for f in files if '.nii.gz' in f and 'FELIX7' not in f]


min_idxes = []
max_idxes = []
names = []
for name in tqdm.tqdm(files):
    file_root = os.path.join(label_path, name)
    lab = load_3d_volume_as_array(file_root)
    tumor_mask =  (lab == 1) + (lab >=3)
    if np.sum(tumor_mask) == 0:
        continue
    idx_min, idx_max = get_ND_bounding_box(tumor_mask, 0)
    min_idxes.append(idx_min)
    max_idxes.append(idx_max)
    names.append(name)
# print('min_idxes: ', min_idxes)
# print('max_idxes: ', max_idxes)

min_idxes_npy, max_idxes_npy = np.asarray(min_idxes), np.asarray(max_idxes)
sizes = max_idxes_npy - min_idxes_npy
names_npy = np.asarray(names, dtype=object)
sizes_names_npy = np.concatenate([names_npy[:,None], sizes], axis=1)
print('name - size : ', sizes_names_npy)
np.save(dst_path, sizes_names_npy)
print('mean: ', np.mean(sizes, axis=0))
print('median: ', np.median(sizes, axis=0))
print('std: ', np.std(sizes, axis=0))
largest_size = np.max(sizes, axis=0)
print('largest size: ', largest_size)
print('smallest size: ', np.min(sizes, axis=0))
print('99: ', np.percentile(sizes, 99, axis=0))
print('95: ', np.percentile(sizes, 95, axis=0))
print('90: ', np.percentile(sizes, 90, axis=0))
print('85: ', np.percentile(sizes, 85, axis=0))
print('80: ', np.percentile(sizes, 80, axis=0))
print('70: ', np.percentile(sizes, 70, axis=0))
print('60: ', np.percentile(sizes, 60, axis=0))