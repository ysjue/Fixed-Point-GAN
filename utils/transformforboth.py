import random
import numpy as np
from utils.augmentationforboth import augment_spatial, augment_mirroring, random_crop, center_crop, crop, augment_gamma
from utils.data_process import  get_ND_bounding_box, resize_3D_volume_to_given_shape, crop_ND_volume_with_bounding_box
# import cv2
import copy
import torch as t
from torchvision.transforms import transforms
# from data_aug.gaussian_blur import GaussianBlur

s = 1
size = 224
# color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)


class local_pixel_shuffling(object):
    def __init__(self,prob=0.5):
        self.prob = prob
    def __call__(self,x):
        prob = self.prob
        if random.random() >= prob:
            return x
        image_temp = copy.deepcopy(x)
        orig_image = copy.deepcopy(x)
        _, img_rows, img_cols = x.shape
        num_block = 100
        for _ in range(num_block):
            block_noise_size_x = random.randint(1, img_rows//10)
            block_noise_size_y = random.randint(1, img_cols//10)
        
            noise_x = random.randint(0, img_rows-block_noise_size_x)
            noise_y = random.randint(0, img_cols-block_noise_size_y)
    
            window = orig_image[0, noise_x:noise_x+block_noise_size_x, 
                                noise_y:noise_y+block_noise_size_y, 
                            ]
            window = window.flatten()
            window = np.random.shuffle(window)
            window = window.reshape((block_noise_size_x, 
                                    block_noise_size_y))
            image_temp[0, noise_x:noise_x+block_noise_size_x, 
                        noise_y:noise_y+block_noise_size_y] = window
        local_shuffling_x = image_temp

        return local_shuffling_x

# transform_2d = transforms.Compose([transforms.RandomHorizontalFlip(),
#                                               transforms.RandomApply([local_pixel_shuffling(0.5)], p=0.8),
#                                               transforms.RandomGrayscale(p=0.2),
#                                               GaussianBlur(kernel_size=int(0.1 * size))])
# class Transform_2d(object):
#     def __init__(self, p=0.2, size = 224 ):
#         self.flip = transforms.RandomHorizontalFlip()
#         self.gray = GammaTransform(gamma_range = (0.7, 1.6))
#         self.blur = GaussianBlur(kernel_size=int(0.1*size))
#     def __call__(self, **data_dict):
#         data = data_dict['data']
#         assert len(data.shape) == 4
#         data = t.from_numpy(data)
#         for i in range(data.shape[1]):
#             x = data[:,i]
#             x = self.flip(x)
            
#             x = self.gray(x.numpy())
#             x = self.blur(t.from_numpy(x).float())
#             data[:,i] = x
#         data_dict['data'] = data
#         return data_dict


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms
        self.data_key   = "data"
        self.label_key  = "seg"
    def __call__(self, **data_dict):
        for t in self.transforms:
            data_dict = t(**data_dict)
        data = data_dict.get(self.data_key)
        seg = data_dict.get(self.label_key)
        return data,seg

    def __repr__(self):
        return str(type(self).__name__) + " ( " + repr(self.transforms) + " )"

class GammaTransform(object):
    def __init__(self, gamma_range=(0.5, 2), invert_image=False, per_channel=False, data_key="data", retain_stats=False, p_per_sample=1):
        """
        Augments by changing 'gamma' of the image (same as gamma correction in photos or computer monitors
        :param gamma_range: range to sample gamma from. If one value is smaller than 1 and the other one is
        larger then half the samples will have gamma <1 and the other >1 (in the inverval that was specified).
        Tuple of float. If one value is < 1 and the other > 1 then half the images will be augmented with gamma values
        smaller than 1 and the other half with > 1
        :param invert_image: whether to invert the image before applying gamma augmentation
        :param per_channel:
        :param data_key:
        :param retain_stats: Gamma transformation will alter the mean and std of the data in the patch. If retain_stats=True,
        the data will be transformed to match the mean and standard deviation before gamma augmentation
        :param p_per_sample:
        """
        self.p_per_sample = p_per_sample
        self.retain_stats = retain_stats
        self.per_channel = per_channel
        self.data_key = data_key
        self.gamma_range = gamma_range
        self.invert_image = invert_image

    def __call__(self, data):
        
        if np.random.uniform() < self.p_per_sample:
            data = augment_gamma(data, self.gamma_range, self.invert_image,
                                                        per_channel=self.per_channel, retain_stats=self.retain_stats)
        return data

# class RandomTranslate(object):
#     def __init__(self,  data_key="data", label_key="seg"):
#         # self.shift_range = shift_range
#         self.label_key = label_key
#         self.data_key = data_key

#     def translate_img(self, img, x_shift, y_shift):
    
#         (height, width) = img.shape[:2]
#         # 平移矩阵(浮点数类型)  x_shift +右移 -左移  y_shift -上移 +下移
#         matrix = np.float32([[1,0,x_shift],[0,1,y_shift]])
#         # 平移图像
#         trans_img = cv2.warpAffine(img, matrix, (width, height))
#         return trans_img

#     def __call__(self , **data_dict):
#         image = data_dict.get(self.data_key)
#         label = data_dict.get(self.label_key)
#         assert len(list(image.shape)) == 4
#         if random.random() < 0.5:
#             shift_y = random.randint(-8,8)
#             shift_x = random.randint(-8,8)
#             for i in range(image.shape[-3]):
#                 for c in range(image.shape[0]):
#                     image[c,i] = self.translate_img(image[c,i], shift_x, shift_y)
#                     label[c,i] = self.translate_img(label[c,i], shift_x, shift_y)
#         data_dict[self.data_key] = image
#         data_dict[self.label_key] = label
        
#         return data_dict

class CenterCrop(object):
    def __init__(self, output_size, mask_label,  data_key="data", label_key="seg"):
        self.output_size = output_size
        self.mask_label = mask_label
        self.label_key = label_key
        self.data_key = data_key

    def __call__(self , **data_dict):
        image = data_dict.get(self.data_key)
        label = data_dict.get(self.label_key)
        temp = np.zeros_like(label)
        for lab in self.mask_label:
            temp = np.maximum(temp, label == lab)
        temp = temp[0]
        idxes = np.nonzero(temp)
        List = [i for i in range(len(idxes[0]))]
        # idx = [idxes[random.sample(List)],

#        print(self.mask_label)
#        print(temp.shape)
        idx_min, idx_max = get_ND_bounding_box(temp, margin = 0)
        assert(len(idx_min) == 3)
        center_point = [int(0.5 *(idx_max[i]- idx_min[i])) for i in range(len(idx_min))]
        size = [80, 112, 96]
        output_size = self.output_size
        for i in range(3):
            if(center_point[i] + 0.5*output_size[i] > image.shape[i + 1]):
                center_point[i] = size[i] - 0.5*output_size[i]
            if(center_point[i] - 0.5*output_size[i]<0):
                center_point[i] = int(0.5*output_size[i])
            idx_min[i] = int(center_point[i] - 0.5*output_size[i])
            idx_max[i] = int(center_point[i] + 0.5*output_size[i]) - 1 
        idx_min = [0] + idx_min
        idx_max = [0] + idx_max
        image = crop_ND_volume_with_bounding_box(image, idx_min, idx_max)
        label = crop_ND_volume_with_bounding_box(label, idx_min, idx_max)
       
        data_dict[self.data_key] = image
        data_dict[self.label_key] = label
        
        return data_dict

class RandomCrop(object):
    """Randomly crop the input image (shape [C, D, H, W] or [C, H, W]) 
    Args:
        output_size (tuple or list): Desired output size [D, H, W] or [H, W].
            the output channel is the same as the input channel.
    """

    def __init__(self, output_size, fg_focus = True, fg_ratio = 0.7, mask_label = None,  data_key="data", label_key="seg"):
        assert isinstance(output_size, (list, tuple))
        if(mask_label is not None):
            assert isinstance(mask_label, (list, tuple))
        self.output_size = output_size
        self.fg_focus = fg_focus
        self.fg_ratio = fg_ratio
        self.mask_label = mask_label
        self.label_key = label_key
        self.data_key = data_key
        

    def __call__(self, **data_dict):
        # image = sample['image']
        image = data_dict.get(self.data_key)
        
        input_shape = image.shape
        input_dim   = len(input_shape) - 1
        self.output_size = input_shape
        assert(input_dim == len(self.output_size))
        crop_margin = [input_shape[i + 1] - self.output_size[i]\
            for i in range(input_dim)]
        crop_min = [random.randint(0, item) for item in crop_margin]
        if(self.fg_focus and random.random() < self.fg_ratio):
            label = data_dict.get(self.label_key)
            mask  = np.zeros_like(label)
            for temp_lab in self.mask_label:
                mask = np.maximum(mask, label == temp_lab)
            bb_min, bb_max = get_ND_bounding_box(mask, margin=0)
            crop_min = [random.randint(bb_min[i], bb_max[i]) - int(self.output_size[i]/2) \
                for i in range(input_dim)]
            crop_min = [max(0, item) for item in crop_min]
            crop_min = [min(crop_min[i], input_shape[i+1] - self.output_size[i]) \
                for i in range(input_dim)]

        crop_max = [crop_min[i] + self.output_size[i] - 1 \
            for i in range(input_dim)]
        crop_min = [0] + crop_min
        crop_max = [input_shape[0] - 1] + crop_max
        # print(crop_max)
        # print(image.shape)
        image = crop_ND_volume_with_bounding_box(image, crop_min, crop_max)
       
        data_dict[self.data_key] = image
        if data_dict.get(self.label_key) is not None:
            label = data_dict[self.label_key]
            
            crop_max[0] = label.shape[0] - 1
            label = crop_ND_volume_with_bounding_box(label, crop_min, crop_max)
            data_dict[self.label_key] = label
        
        return data_dict


class IntensityTransform(object):

    def __init__(self, shift_range = (-0.1,0.1),scale_range = (0.9,1.1), data_key="data"):
        
        self.shift_range = shift_range
        self.scale_range = scale_range
        self.data_key = data_key
    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        #seg = data_dict.get(self.label_key)
        
        scale_shift_data = []
        for i in range(len(data)):
            img = data[i]
            min_x, max_x = np.min(img),np.max(img)
            std = np.std(img)
            # shift_channel = np.clip(img + np.random.uniform(self.shift_range[0] * std, self.shift_range[1] * std), min_x,max_x)
            # scale_shift_channel = np.clip(shift_channel * np.random.uniform(self.scale_range[0], self.scale_range[1]), min_x,max_x)
            shift_channel = data[i] + np.random.uniform(self.shift_range[0] * std, self.shift_range[1] * std)
            scale_shift_channel = shift_channel * np.random.uniform(self.scale_range[0], self.scale_range[1])
            scale_shift_data.append(scale_shift_channel)
        scale_shift_data = np.stack(scale_shift_data)
        data_dict[self.data_key] = scale_shift_data
        #print('data:',data_dict['data'].shape)
        return data_dict

class Z_randomcrop_toTensor(object):
    def __init__(self, batch_size, data_key="data", label_key="seg"):
        self.batch_size = batch_size
        self.data_key = data_key
        self.label_key = label_key

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        label = data_dict.get(self.label_key)
        data = t.from_numpy(data)
        label = t.from_numpy(label)
        if random.random() > 0.5:
            start = int(random.random()*(data.shape[-3] - self.batch_size))
        else:
            start = self.posit_sample(label[0], self.batch_size)
        assert data.shape[-3] >= self.batch_size
            
        crop_data = data[:, start : start + self.batch_size]
        crop_label = label[:, start : start + self.batch_size]
        data_dict[self.data_key] = crop_data.float()
        data_dict[self.label_key] = crop_label.float()
        return data_dict

    def posit_sample(self, label, batch_size):
        assert len(list(label.shape)) ==  3
        slic = t.sum(label.contiguous().view(label.shape[-3], -1), dim = 1)
        slic = (slic > 0).int()
        idxes = t.nonzero(slic)
        bmin,_ = t.min(idxes, dim = 0)
        bmax,_ = t.max(idxes, dim = 0)
        zmin = max(0, bmin-batch_size)
        zmax = min(label.shape[-3], bmax + batch_size)
        start = int(zmin+random.random()*(zmax - batch_size - zmin))
        return start

class RandomCropTransform(object):
    """ Randomly crops data and seg (if available)
    Args:
        crop_size (int or tuple of int): Output patch size
        margins (tuple of int): how much distance should the patch border have to the image broder (bilaterally)?
    """

    def __init__(self, crop_size=128, margins=(0, 0, 0), data_key="data", label_key="seg"):
        self.data_key = data_key
        self.label_key = label_key
        self.margins = margins
        self.crop_size = crop_size

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        seg = data_dict.get(self.label_key)
        crop_size = [s for s in self.crop_size]
        if self.crop_size[0] == None:
            crop_size[0] = data.shape[-3]
        else:
            crop_size = self.crop_size

        data, seg = random_crop(data, seg, crop_size, self.margins)

        data_dict[self.data_key] = data
        if seg is not None:
            data_dict[self.label_key] = seg

        return data_dict
class MirrorTransform(object):
    """ Randomly mirrors data along specified axes. Mirroring is evenly distributed. Probability of mirroring along
    each axis is 0.5
    Args:
        axes (tuple of int): axes along which to mirror
    """
    def __init__(self, axes=(0, 1, 2), data_key="data", label_key="seg"):
        self.data_key = data_key
        self.label_key = label_key
        self.axes = axes
        if max(axes) > 2:
            raise ValueError("MirrorTransform now takes the axes as the spatial dimensions. What previously was "
                             "axes=(2, 3, 4) to mirror along all spatial dimensions of a 5d tensor (b, c, x, y, z) "
                             "is now axes=(0, 1, 2). Please adapt your scripts accordingly.")

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        seg = data_dict.get(self.label_key)


        sample_seg = None
        if seg is not None:
            sample_seg = seg
        ret_val = augment_mirroring(data, sample_seg, axes=self.axes)
        data = ret_val[0]
        if seg is not None:
            seg = ret_val[1]

        data_dict[self.data_key] = data
        if seg is not None:
            data_dict[self.label_key] = seg

        return data_dict
class CenterCropTransform(object):
    """ Crops data and seg (if available) in the center
    Args:
        output_size (int or tuple of int): Output patch size
    """

    def __init__(self, crop_size, data_key="data", label_key="seg"):
        self.data_key = data_key
        self.label_key = label_key
        self.crop_size = crop_size

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        seg = data_dict.get(self.label_key)
        idxes = np.nonzero(seg[0])
        idxes = np.array(idxes)
        min_idx = np.min(idxes, 1)
        max_idx = np.max(idxes, 1)
        center_pt = ((max_idx + min_idx)//2).tolist()
        shape = data.shape[-3:]
        for i in range(len(shape)):
            if self.crop_size[i] == None:
                min_idx[i] = max(0, min_idx[i] - 5)
                max_idx[i] = min(shape[i] - 1, max_idx[i]+5)
                continue
            center_pt[i] = min(shape[i] - 1 - self.crop_size[i]//2, center_pt[i])
            center_pt[i] = max(self.crop_size[i]//2, center_pt[i])
            min_idx[i] = center_pt[i] - self.crop_size[i]//2
            max_idx[i] = center_pt[i] + self.crop_size[i]//2 - 1
            
            # print(min_idx[i], max_idx[i] - 1)
        
        datas, segs = [], []
        for i in range(data.shape[0]):
            data = crop_ND_volume_with_bounding_box(data[i], min_idx, max_idx)
            datas.append(data)
            seg = crop_ND_volume_with_bounding_box(seg[i], min_idx, max_idx)
            segs.append(seg)
        data, seg = np.asarray(datas, np.float32), np.asarray(segs, np.uint8)
        # data, seg = center_crop(data, self.crop_size, seg)

        data_dict[self.data_key] = data
        if seg is not None:
            data_dict[self.label_key] = seg

        return data_dict

class SpatialTransform(object):
    """The ultimate spatial transform generator. Rotation, deformation, scaling, cropping: It has all you ever dreamed
    of. Computational time scales only with patch_size, not with input patch size or type of augmentations used.
    Internally, this transform will use a coordinate grid of shape patch_size to which the transformations are
    applied (very fast). Interpolation on the image data will only be done at the very end
    Args:
        patch_size (tuple/list/ndarray of int): Output patch size
        patch_center_dist_from_border (tuple/list/ndarray of int, or int): How far should the center pixel of the
        extracted patch be from the image border? Recommended to use patch_size//2.
        This only applies when random_crop=True
        do_elastic_deform (bool): Whether or not to apply elastic deformation
        alpha (tuple of float): magnitude of the elastic deformation; randomly sampled from interval
        sigma (tuple of float): scale of the elastic deformation (small = local, large = global); randomly sampled
        from interval
        do_rotation (bool): Whether or not to apply rotation
        angle_x, angle_y, angle_z (tuple of float): angle in rad; randomly sampled from interval. Always double check
        whether axes are correct!
        do_scale (bool): Whether or not to apply scaling
        scale (tuple of float): scale range ; scale is randomly sampled from interval
        border_mode_data: How to treat border pixels in data? see scipy.ndimage.map_coordinates
        border_cval_data: If border_mode_data=constant, what value to use?
        order_data: Order of interpolation for data. see scipy.ndimage.map_coordinates
        border_mode_seg: How to treat border pixels in seg? see scipy.ndimage.map_coordinates
        border_cval_seg: If border_mode_seg=constant, what value to use?
        order_seg: Order of interpolation for seg. see scipy.ndimage.map_coordinates. Strongly recommended to use 0!
        If !=0 then you will have to round to int and also beware of interpolation artifacts if you have more then
        labels 0 and 1. (for example if you have [0, 0, 0, 2, 2, 1, 0] the neighboring [0, 0, 2] bay result in [0, 1, 2])
        random_crop: True: do a random crop of size patch_size and minimal distance to border of
        patch_center_dist_from_border. False: do a center crop of size patch_size
    """
    def __init__(self, patch_size, patch_center_dist_from_border=64,
                 do_elastic_deform=True, alpha=(0., 750.), sigma=(10., 13.),
                 do_rotation=False, angle_x=(0, 2 * np.pi), angle_y=(0, 2 * np.pi), angle_z=(0, 2 * np.pi),
                 do_scale=False, scale=(0.85, 1.15), border_mode_data='nearest', border_cval_data=0, order_data=3,
                 border_mode_seg='constant', border_cval_seg=0, order_seg=0, random_crop=True, data_key="data", 
                 label_key="seg", p_el_per_sample=0.3, p_scale_per_sample=0.3, p_rot_per_sample=0.3):
        self.p_rot_per_sample = p_rot_per_sample
        self.p_scale_per_sample = p_scale_per_sample
        self.p_el_per_sample = p_el_per_sample
        self.data_key = data_key
        self.label_key = label_key
        self.patch_size = patch_size
        self.patch_center_dist_from_border = patch_center_dist_from_border
        self.do_elastic_deform = do_elastic_deform
        self.alpha = alpha
        self.sigma = sigma
        self.do_rotation = do_rotation
        self.angle_x = angle_x
        self.angle_y = angle_y
        self.angle_z = angle_z
        self.do_scale = do_scale
        self.scale = scale
        self.border_mode_data = border_mode_data
        self.border_cval_data = border_cval_data
        self.order_data = order_data
        self.border_mode_seg = border_mode_seg
        self.border_cval_seg = border_cval_seg
        self.order_seg = order_seg
        self.random_crop = random_crop

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        seg = data_dict.get(self.label_key)

        if self.patch_size is None:
            if len(data.shape) == 3:
                patch_size = (data.shape[1], data.shape[2])
            elif len(data.shape) == 4:
                patch_size = (data.shape[1], data.shape[2], data.shape[3])
            else:
                raise ValueError("only support 2D/3D batch data.")
        else:
            patch_size = self.patch_size

        ret_val = augment_spatial(data, seg, patch_size=patch_size,
                                  patch_center_dist_from_border=self.patch_center_dist_from_border,
                                  do_elastic_deform=self.do_elastic_deform, alpha=self.alpha, sigma=self.sigma,
                                  do_rotation=self.do_rotation, angle_x=self.angle_x, angle_y=self.angle_y,
                                  angle_z=self.angle_z, do_scale=self.do_scale, scale=self.scale,
                                  border_mode_data=self.border_mode_data,
                                  border_cval_data=self.border_cval_data, order_data=self.order_data,
                                  border_mode_seg=self.border_mode_seg, border_cval_seg=self.border_cval_seg,
                                  order_seg=self.order_seg, random_crop=self.random_crop,
                                  p_el_per_sample=self.p_el_per_sample, p_scale_per_sample=self.p_scale_per_sample,
                                  p_rot_per_sample=self.p_rot_per_sample)

        data_dict[self.data_key] = ret_val[0]#.astype(data.dtype)
        if seg is not None:
            data_dict[self.label_key] = ret_val[1]#.astype(seg.dtype)

        return data_dict