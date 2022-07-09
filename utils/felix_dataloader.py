import os
import numpy as np 
from torch.utils import data
# from abdomen_config import opt
from utils.data_process import *
import torch as t
import SimpleITK as sitk
import torchvision.transforms as trans 
# from utils.transformforboth import CenterCrop, RandomCrop,Compose, Z_randomcrop_toTensor, IntensityTransform, RandomCropTransform, SpatialTransform, MirrorTransform, CenterCropTransform, GammaTransform
import itertools
from torch.utils.data.sampler import Sampler
import torch.nn.functional as F



class datasets(data.Dataset):
    def __init__(self  ,train=True ,
                test = False, 
                data_root = '',
                txt_path = '/data/shuojue/code/Fixed-Point-GAN/split/train.txt',
                crop_size = [80, 224,256] ): #80,224,256

        self.resize_size = [None, 256, 256]
        self.train = train
        self.test  = test
        self.data_error = False
    
        self.data_root = data_root

        if self.train:
            self.image_folder =  'img'
            self.label_folder =  'label_6cls'
        if self.test:
            self.image_folder =  'imagesVal'
            self.label_folder =  'labelsVal'
            
        if txt_path == None:
            self.names = os.listdir(os.path.join(data_root,self.image_folder))
        else:
            with open(txt_path) as f:
                content = f.readlines()
            self.names  = [x.strip().split(',')[0] for x in content if 'FELIX7' not in x]
        self.with_ground_truth    = True
        self.label_convert_source = None
        self.label_convert_target = None
        if(self.label_convert_source and self.label_convert_target):
            assert(len(self.label_convert_source) == len(self.label_convert_target))
        
        # self.file_postfix  = opt.file_postfix

        self.crop_size     = crop_size

        self.intensity_normalize = True
        # if self.train:
        #     self.transform = Compose([
        #                 IntensityTransform(shift_range = (-0.1,0.1),scale_range = (0.9,1.1)),
        #                 RandomCropTransform(self.crop_size),
        #                 # Z_randomcrop_toTensor(14),
        #                 # RandomCrop(self.crop_size, mask_label=[i for i in range(17) if i > 0]),
        #                 SpatialTransform(self.crop_size, random_crop = False, do_elastic_deform=True,
        #                                        alpha=(0., 1300.), sigma=(10., 13.),
        #                                        do_rotation=True, angle_x=(-np.pi/8., np.pi/8.), angle_y=(-np.pi/8., np.pi/8.), angle_z=(-np.pi/8., np.pi/8.),
        #                                        do_scale=False, scale=(0.75, 1.25)),
        #                 # MirrorTransform(axes=(1,2)),
        #                 # GammaTransform(gamma_range = (0.8, 1.5))
        #                            ])

        # if self.test:
        #     self.transform = Compose([
        #                 CenterCropTransform(self.crop_size)                
        #                                 ])

                            

    def _find_bounding_box(self, volume, threshold, erosion):
        cut_off_lower = np.percentile(volume.ravel(), threshold)
        # print('cutoff_lower:', cut_off_lower)
        struct = ndimage.generate_binary_structure(3, 2)
        if erosion:
            c = get_largest_one_component(ndimage.morphology.binary_erosion(volume > cut_off_lower, structure=struct, iterations=3))
            idx_min, idx_max = get_ND_bounding_box(c, 10)
        else:
            c = get_largest_one_component(volume > cut_off_lower)
            idx_min, idx_max = get_ND_bounding_box(c, 5)
        return idx_min, idx_max

    def _get_one_sample(self, string):
        name = string
        imagedir = os.path.join(os.path.join(self.data_root, self.image_folder), \
                                name + '.nii.gz')
        labeldir = os.path.join(os.path.join(self.data_root, self.label_folder), \
                                name + '.nii.gz')
        data, label, min_idx, max_idx, data_error = self.load_data(imagedir, labeldir)
        # data_dict = {"data":data, "seg":label}
        # data, label = self.transform(**data_dict)
        # label      = self.class2one_hot(label, 17)
        # if isinstance(data, t.FloatTensor):
        #     return data, label, imagedir
        return data, label, imagedir, min_idx, max_idx,data_error

    def __getitem__(self , index):
        name = self.names[index]
        name = name.split('.nii.gz')[0]
        data, label, imagedir, min_idx, max_idx, data_error = self._get_one_sample(name)
        return data, label, imagedir, min_idx, max_idx,data_error
            
    def __len__(self):
        names = [n for n in self.names ]
        return len(names)

    def cut_off_values_upper_lower_percentile(self, image, mask = None, percentile_lower =0.5, percentile_upper=99.5):
        # cut_off_lower = np.percentile(image[image > 0].ravel(), percentile_lower)
        # cut_off_upper = np.percentile(image[image > 0].ravel(), percentile_upper)
        # cut_off_lower = -650 - 750
        # cut_off_upper = -650 + 750
        cut_off_lower = 40 - 175
        cut_off_upper = 40 + 175
        res = np.copy(image)
        res = np.clip(res, cut_off_lower, cut_off_upper)
        # res[(res < cut_off_lower) & (mask !=0)] = cut_off_lower
        # res[(res > cut_off_upper) & (mask !=0)] = cut_off_upper
        return res
    
    def load_data(self, imagedir, labeldir):
        """
        load one training/testing data
        """
        data_volumes = []
        label_volumes = []
        name = labeldir.split('/')[-1]
                
        label = load_3d_volume_as_array(labeldir)
        volume = load_nifty_volume_as_array(imagedir)
        anomaly_mask = label >= 3
        pancreas_mask = label == 1 
        mask = pancreas_mask + anomaly_mask
        self.data_error = False
        if np.sum(mask) == 0:
            self.data_error = True
            return volume, label, 0, 0, True
        if 'cys' in  name.lower() or \
                    'PDAC' in name or \
                    'FELIX5' in name:
            if np.sum(label >= 3) == 0:    
                self.data_error = True
                return volume, label, 0, 0,True
        else:
            if np.sum(label >= 3) > 0:
                self.data_error = True
                return volume, label, 0, 0, True
            
        min_idx, max_idx = get_ND_bounding_box(mask, [0,5,5])
        min_idx = np.asarray(min_idx, np.int32)
        max_idx = np.asarray(max_idx, np.int32)
        [D, H, W] = label.shape
        center_pt = np.mean(np.stack([min_idx , max_idx]), axis = 0)
        size = max_idx - min_idx
        if size[1] < self.crop_size[1]:
            min_idx[1] = center_pt[1] - self.crop_size[1]/2.0
            max_idx[1] = center_pt[1] + self.crop_size[1]/2.0
        if size[2] < self.crop_size[2]:
            min_idx[2] = center_pt[2] - self.crop_size[2]/2.0
            max_idx[2] = center_pt[2] + self.crop_size[2]/2.0
        
        
        assert np.all(min_idx >= 0)
        assert np.all(max_idx <= np.array([D,H,W]))

        label = crop_ND_volume_with_bounding_box(label, min_idx, max_idx)
        

            
        if self.train or self.test:
            volume = crop_ND_volume_with_bounding_box(volume, min_idx, max_idx)
            max_val = np.max(volume)
            min_val = np.min(volume)
        # if D < self.crop_size[0] or H < self.crop_size[1] or W < self.crop_size[2]:
        #     label = self.pad_nd_image(label, self.crop_size)
        # save_array_as_nifty_volume(volume,name.split('.nii.gz')[0]+'test_without_normalization.nii.gz')
        if(self.intensity_normalize):
            volume = self.cut_off_values_upper_lower_percentile(volume)
            # volume = (volume - min_val)/(max_val*1.0 - min_val*1.0)
            volume = normalize_one_volume(volume)
            # save_array_as_nifty_volume(volume,name.split('.nii.gz')[0]+'test_after_normalization.nii.gz')
        # if  H < self.crop_size[1] or W < self.crop_size[2]:
        #     size = [label.shape[-3]] + self.crop_size[-2:]
        #     label = self.pad_nd_image(label, size)
        label_volumes.append(label)
        if(self.label_convert_source and self.label_convert_target):
            label_volumes[0] = convert_label(label_volumes[0], self.label_convert_source, self.label_convert_target)
        data_volumes.append(volume)
        data   = np.asarray(data_volumes, np.float32)

        label  = np.asarray(label_volumes, np.uint8)
        # label  = label [np.newaxis,:,:,:]
        # data   = data [np.newaxis,:,:,:]
        
        return data, label, min_idx, max_idx, False

    def class2one_hot(self, seg, C):

        k, w, h, l = seg.shape  
        # type: Tuple[int, int, int, int]
        res = np.concatenate([seg == c for c in range(C)], axis=0).astype(np.int32)
        # res[3] = res[3]>0
        # res[1] = (res[1] + res[3])>0
        # res[2] = (res[1] + res[2] + res[3])>0
        assert res.shape == (C, w, h, l)
        return res

    def pad_nd_image(self, image, new_shape=None, mode="constant", kwargs=None, return_slicer=False, shape_must_be_divisible_by=None):
        """
        one padder to pad them all. Documentation? Well okay. A little bit
        :param image: nd image. can be anything
        :param new_shape: what shape do you want? new_shape does not have to have the same dimensionality as image. If
        len(new_shape) < len(image.shape) then the last axes of image will be padded. If new_shape < image.shape in any of
        the axes then we will not pad that axis, but also not crop! (interpret new_shape as new_min_shape)
        Example:
        image.shape = (10, 1, 512, 512); new_shape = (768, 768) -> result: (10, 1, 768, 768). Cool, huh?
        image.shape = (10, 1, 512, 512); new_shape = (364, 768) -> result: (10, 1, 512, 768).
        :param mode: see np.pad for documentation
        :param return_slicer: if True then this function will also return what coords you will need to use when cropping back
        to original shape
        :param shape_must_be_divisible_by: for network prediction. After applying new_shape, make sure the new shape is
        divisibly by that number (can also be a list with an entry for each axis). Whatever is missing to match that will
        be padded (so the result may be larger than new_shape if shape_must_be_divisible_by is not None)
        :param kwargs: see np.pad for documentation
        """
        if kwargs is None:
            kwargs = {'constant_values': 0}
    
        if new_shape is not None:
            old_shape = np.array(image.shape[-len(new_shape):])
        else:
            assert shape_must_be_divisible_by is not None
            assert isinstance(shape_must_be_divisible_by, (list, tuple, np.ndarray))
            new_shape = image.shape[-len(shape_must_be_divisible_by):]
            old_shape = new_shape

        num_axes_nopad = len(image.shape) - len(new_shape)

        new_shape = [max(new_shape[i], old_shape[i]) for i in range(len(new_shape))]

        if not isinstance(new_shape, np.ndarray):
            new_shape = np.array(new_shape)

        if shape_must_be_divisible_by is not None:
            if not isinstance(shape_must_be_divisible_by, (list, tuple, np.ndarray)):
                shape_must_be_divisible_by = [shape_must_be_divisible_by] * len(new_shape)
            else:
                assert len(shape_must_be_divisible_by) == len(new_shape)

            for i in range(len(new_shape)):
                if new_shape[i] % shape_must_be_divisible_by[i] == 0:
                    new_shape[i] -= shape_must_be_divisible_by[i]

            new_shape = np.array([new_shape[i] + shape_must_be_divisible_by[i] - new_shape[i] % shape_must_be_divisible_by[i] for i in range(len(new_shape))])

        difference = new_shape - old_shape
        pad_below = difference // 2
        pad_above = difference // 2 + difference % 2
        pad_list = [[0, 0]]*num_axes_nopad + list([list(i) for i in zip(pad_below, pad_above)])
        res = np.pad(image, pad_list, mode, **kwargs)
        if not return_slicer:
            return res
        else:
            pad_list = np.array(pad_list)
            pad_list[:, 1] = np.array(res.shape) - pad_list[:, 1]
            slicer = list(slice(*i) for i in pad_list)
            return res, slicer



def write(data, fname, root='/data1/shuojue'):
    fname = os.path.join(root, fname)
    with open(fname, 'w') as f:
        f.write('\n'.join(data)) 
        f.close()



# if __name__ == "__main__":
#     main()