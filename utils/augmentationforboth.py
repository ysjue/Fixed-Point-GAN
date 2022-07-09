import numpy as np
import pandas as pd
#import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt


def augment_mirroring(sample_data, sample_seg=None, axes=(0, 1, 2)):
    if (len(sample_data.shape) != 3) and (len(sample_data.shape) != 4):
        raise Exception(
            "Invalid dimension for sample_data and sample_seg. sample_data and sample_seg should be either "
            "[channels, x, y] or [channels, x, y, z]")
    if 0 in axes and np.random.uniform() < 0.5:
        sample_data[:, :] = sample_data[:, ::-1]
        if sample_seg is not None:
            sample_seg[:, :] = sample_seg[:, ::-1]
    if 1 in axes and np.random.uniform() < 0.5:
        sample_data[:, :, :] = sample_data[:, :, ::-1]
        if sample_seg is not None:
            sample_seg[:, :, :] = sample_seg[:, :, ::-1]
    if 2 in axes and len(sample_data.shape) == 4:
        if np.random.uniform() < 0.5:
            sample_data[:, :, :, :] = sample_data[:, :, :, ::-1]
            if sample_seg is not None:
                sample_seg[:, :, :, :] = sample_seg[:, :, :, ::-1]
    return sample_data, sample_seg

def center_crop(data, crop_size, seg=None):
    return crop(data, seg, crop_size, 0, 'center')


def random_crop(data, seg=None, crop_size=128, margins=[0, 0, 0]):
    return crop(data, seg, crop_size, margins, 'random')


def get_lbs_for_random_crop(crop_size, data_shape, margins):
    """
    :param crop_size:
    :param data_shape: (b,c,x,y(,z)) must be the whole thing!
    :param margins:
    :return:
    """
    lbs = []
    for i in range(len(data_shape) - 1):
        if data_shape[i+1] - crop_size[i] - margins[i] > margins[i]:
            lbs.append(np.random.randint(margins[i], data_shape[i+1] - crop_size[i] - margins[i]))
        else:
            lbs.append((data_shape[i+1] - crop_size[i]) // 2)                                         
    return lbs


def get_lbs_for_center_crop(crop_size, data_shape):
    """
    :param crop_size:
    :param data_shape: (b,c,x,y(,z)) must be the whole thing!
    :return:
    """
    lbs = []
    for i in range(len(data_shape) - 1):
        lbs.append((data_shape[i + 1] - crop_size[i]) // 2)
    return lbs


def crop(data, seg=None, crop_size=128, margins=(0, 0, 0), crop_type="center",
         pad_mode='constant', pad_kwargs={'constant_values': 0},
         pad_mode_seg='constant', pad_kwargs_seg={'constant_values': 0}):
    """
    crops data and seg (seg may be None) to crop_size. Whether this will be achieved via center or random crop is
    determined by crop_type. Margin will be respected only for random_crop and will prevent the crops form being closer
    than margin to the respective image border. crop_size can be larger than data_shape - margin -> data/seg will be
    padded with zeros in that case. margins can be negative -> results in padding of data/seg followed by cropping with
    margin=0 for the appropriate axes
    :param data: b, c, x, y(, z)
    :param seg:
    :param crop_size:
    :param margins: distance from each border, can be int or list/tuple of ints (one element for each dimension).
    Can be negative (data/seg will be padded if needed)
    :param crop_type: random or center
    :return:
    """
    if not isinstance(data, (list, tuple, np.ndarray)):
        raise TypeError("data has to be either a numpy array or a list")

    data_shape = data.shape
    data_dtype = data.dtype
    dim = len(data_shape) - 1
   
    if seg is not None:
        seg_shape = seg.shape
        seg_dtype = seg.dtype

        if not isinstance(seg, (list, tuple, np.ndarray)):
            raise TypeError("data has to be either a numpy array or a list")

        assert all([i == j for i, j in zip(seg_shape[1:], data_shape[1:])]), "data and seg must have the same spatial " \
                                                                             "dimensions. Data: %s, seg: %s" % \
                                                                             (str(data_shape), str(seg_shape))

    if type(crop_size) not in (tuple, list, np.ndarray):
        crop_size = [crop_size] * dim
    else:
        assert len(crop_size) == len(
            data_shape) - 1, "If you provide a list/tuple as center crop make sure it has the same dimension as your " \
                             "data (2d/3d)"

    if not isinstance(margins, (np.ndarray, tuple, list)):
        margins = [margins] * dim

    data_return = np.zeros([data_shape[0]] + list(crop_size), dtype=data_dtype)
    if seg is not None:
        seg_return = np.zeros([seg_shape[0]] + list(crop_size), dtype=seg_dtype)
    else:
        seg_return = None


    if crop_type == "center":
        
        lbs = get_lbs_for_center_crop(crop_size, data_shape)
    elif crop_type == "random":
        lbs = get_lbs_for_random_crop(crop_size, data_shape, margins)
    else:
        raise NotImplementedError("crop_type must be either center or random")

    need_to_pad = [[0, 0]] + [[abs(min(0, lbs[d])),
                                abs(min(0, data_shape[d + 1] - (lbs[d] + crop_size[d])))]
                                for d in range(dim)]


    # we should crop first, then pad -> reduces i/o for memmaps, reduces RAM usage and improves speed
    ubs = [min(lbs[d] + crop_size[d], data_shape[d+1]) for d in range(dim)]
    lbs = [max(0, lbs[d]) for d in range(dim)]

    slicer_data = [slice(0, data_shape[0])] + [slice(lbs[d], ubs[d]) for d in range(dim)]
    data_cropped = data[tuple(slicer_data)]

    if seg_return is not None:
        slicer_seg = [slice(0, seg_shape[0])] + [slice(lbs[d], ubs[d]) for d in range(dim)]
        seg_cropped = seg[tuple(slicer_seg)]

    if any([i > 0 for j in need_to_pad for i in j]):
        data_return = np.pad(data_cropped, need_to_pad, pad_mode, **pad_kwargs)
        out_random = np.random.normal(0, 1, size = data_return.shape)
        data_return[data_return == 0] = out_random[data_return == 0]
        if seg_return is not None:
            seg_return = np.pad(seg_cropped, need_to_pad, pad_mode_seg, **pad_kwargs_seg)
    else:
        data_return = data_cropped
        if seg_return is not None:
            seg_return = seg_cropped

    return data_return, seg_return

    


def Elastic_transform(image, alpha, sigma):
    shape = image.shape
    shape_size = shape[:2]
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]), np.arange(shape[3]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z+dz, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)

'''def Elastic_transform_cood(coords, patch_size, alpha, sigma):
    shape = patch_size
    random_state = np.random.RandomState(None)
    dx = gaussian_filter((random_state.rand(*shape[1:]) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape[1:]) * 2 - 1), sigma) * alpha
    dz = gaussian_filter((random_state.rand(*shape[1:]) * 2 - 1), sigma) * alpha
    indices = np.array([coords[0]+dx, coords[1]+dy, coords[2]+dz])
    return indices'''
def elastic_deform_coordinates(coordinates, alpha, sigma):
    n_dim = len(coordinates)
    offsets = []
    for _ in range(n_dim):
        offsets.append(
            gaussian_filter((np.random.random(coordinates.shape[1:]) * 2 - 1), sigma, mode="constant", cval=0) * alpha)
    offsets = np.array(offsets)
    indices = offsets + coordinates
    return indices

def create_zero_centered_coordinate_mesh(shape):
    tmp = tuple([np.arange(i) for i in shape])
    coords = np.array(np.meshgrid(*tmp, indexing='ij')).astype(float)
    for d in range(len(shape)):
        coords[d] -= ((np.array(shape).astype(float) - 1) / 2.)[d]
    return coords

def create_matrix_rotation_x_3d(angle, matrix=None):
    rotation_x = np.array([[1, 0, 0],
                           [0, np.cos(angle), -np.sin(angle)],
                           [0, np.sin(angle), np.cos(angle)]])
    if matrix is None:
        return rotation_x

    return np.dot(matrix, rotation_x)


def create_matrix_rotation_y_3d(angle, matrix=None):
    rotation_y = np.array([[np.cos(angle), 0, np.sin(angle)],
                           [0, 1, 0],
                           [-np.sin(angle), 0, np.cos(angle)]])
    if matrix is None:
        return rotation_y

    return np.dot(matrix, rotation_y)


def create_matrix_rotation_z_3d(angle, matrix=None):
    rotation_z = np.array([[np.cos(angle), -np.sin(angle), 0],
                           [np.sin(angle), np.cos(angle), 0],
                           [0, 0, 1]])
    if matrix is None:
        return rotation_z

    return np.dot(matrix, rotation_z)


def create_matrix_rotation_2d(angle, matrix=None):
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    if matrix is None:
        return rotation

    return np.dot(matrix, rotation)


def create_random_rotation(angle_x=(0, 2 * np.pi), angle_y=(0, 2 * np.pi), angle_z=(0, 2 * np.pi)):
    return create_matrix_rotation_x_3d(np.random.uniform(*angle_x),
                                       create_matrix_rotation_y_3d(np.random.uniform(*angle_y),
                                                                   create_matrix_rotation_z_3d(
                                                                       np.random.uniform(*angle_z))))


def rotate_coords_3d(coords, angle_x, angle_y, angle_z):
    rot_matrix = np.identity(len(coords))
    rot_matrix = create_matrix_rotation_x_3d(angle_x, rot_matrix)
    rot_matrix = create_matrix_rotation_y_3d(angle_y, rot_matrix)
    rot_matrix = create_matrix_rotation_z_3d(angle_z, rot_matrix)
    coords = np.dot(coords.reshape(len(coords), -1).transpose(), rot_matrix).transpose().reshape(coords.shape)
    return coords


def rotate_coords_2d(coords, angle):
    rot_matrix = create_matrix_rotation_2d(angle)
    coords = np.dot(coords.reshape(len(coords), -1).transpose(), rot_matrix).transpose().reshape(coords.shape)
    return coords


def scale_coords(coords, scale):
    return coords * scale
def interpolate_img(img, coords, order=3, mode='nearest', cval=0.0, is_seg=False):
    if is_seg and order != 0:
        unique_labels = np.unique(img)
        result = np.zeros(coords.shape[1:], img.dtype)
        for i, c in enumerate(unique_labels):
            res_new = map_coordinates((img == c).astype(np.float32), coords, order=order, mode=mode, cval=cval)#关注一下这里的order,原先是order=order
            result[res_new >= 0.5] = c
        return result.astype(img.dtype)
    else:
        return map_coordinates(img.astype(np.float32), coords, order=order, mode=mode, cval=cval).astype(img.dtype)


def augment_spatial(data, seg, patch_size, patch_center_dist_from_border=64,
                    do_elastic_deform=False, alpha=(0., 1000.), sigma=(30, 50),
                    do_rotation=True, angle_x=(0, 2 * np.pi), angle_y=(0, 2 * np.pi), angle_z=(0, 2 * np.pi),
                    do_scale=False, scale=(0.5, 3), border_mode_data='nearest', border_cval_data=0, order_data=3,
                    border_mode_seg='constant', border_cval_seg=0, order_seg=0, random_crop=False, p_el_per_sample=0.5,
                    p_scale_per_sample=0.5, p_rot_per_sample=0.5):
    dim = len(patch_size)
    seg_result = None
    if seg is not None:
        if dim == 2:
            seg_result = np.zeros((seg.shape[0], patch_size[0], patch_size[1]), dtype=np.uint8)
        else:
            seg_result = np.zeros((seg.shape[0], patch_size[0], patch_size[1], patch_size[2]),
                                  dtype=np.uint8)

    if dim == 2:
        data_result = np.zeros((data.shape[0], patch_size[0], patch_size[1]), dtype=np.float32)
    else:
        data_result = np.zeros((data.shape[0], patch_size[0], patch_size[1], patch_size[2]),
                               dtype=np.float32)

    if not isinstance(patch_center_dist_from_border, (list, tuple, np.ndarray)):
        patch_center_dist_from_border = dim * [patch_center_dist_from_border]
    coords = create_zero_centered_coordinate_mesh(patch_size)
    modified_coords = False


    if np.random.uniform() < p_el_per_sample and do_elastic_deform:
        a = np.random.uniform(alpha[0], alpha[1])
        s = np.random.uniform(sigma[0], sigma[1])
        coords = elastic_deform_coordinates(coords, a, s)
        modified_coords = True
 
    if np.random.uniform() < p_rot_per_sample and do_rotation:
        if angle_x[0] == angle_x[1]:
            a_x = angle_x[0]
        else:
            a_x = np.random.uniform(angle_x[0], angle_x[1])
        if dim == 3:
            if angle_y[0] == angle_y[1]:
                a_y = angle_y[0]
            else:
                a_y = np.random.uniform(angle_y[0], angle_y[1])
            if angle_z[0] == angle_z[1]:
                a_z = angle_z[0]
            else:
                a_z = np.random.uniform(angle_z[0], angle_z[1])
            coords = rotate_coords_3d(coords, a_x, a_y, a_z)
        else:
            coords = rotate_coords_2d(coords, a_x)
        modified_coords = True

    if np.random.uniform()  < p_scale_per_sample and do_scale:
        if np.random.random() < 0.5 and scale[0] < 1:
            sc = np.random.uniform(scale[0], 1)
        else:
            sc = np.random.uniform(max(scale[0], 1), scale[1])
        coords = scale_coords(coords, sc)

        modified_coords = True
    # now find a nice center location
    if modified_coords:
        for d in range(dim):
            if random_crop:
                ctr = np.random.uniform(patch_center_dist_from_border[d],
                                        data.shape[d + 1] - patch_center_dist_from_border[d])
            else:
                ctr = int(np.round(data.shape[d + 1] / 2.))#返回四舍五入的zhi, 要不要减去patch_size很困惑
               
            coords[d] += ctr
        for channel_id in range(data.shape[0]):
            data_result[channel_id] = interpolate_img(data[channel_id], coords, order_data,
                                                                    border_mode_data, cval=border_cval_data)
        if seg is not None:
            for channel_id in range(seg.shape[0]):
                seg_result[channel_id] = interpolate_img(seg[channel_id], coords, order_seg,
                                                                    border_mode_seg, cval=border_cval_seg, is_seg=True)
    else:
        if seg is None:
            s = None
        else:
            s = seg
        if random_crop:
            margin = [patch_center_dist_from_border[d] - patch_size[d] // 2 for d in range(dim)]
            d, s = random_crop(data, s, patch_size, margin)
        else:
            d, s = center_crop(data, patch_size, s)
        data_result = d
        if seg is not None:
            seg_result = s
    return data_result, seg_result

def augment_gamma(data_sample, gamma_range=(0.5, 2), invert_image=False, epsilon=1e-7, per_channel=False,
                  retain_stats=False):
    if invert_image:
        data_sample = - data_sample
    if not per_channel:
        if retain_stats:
            mn = data_sample.mean()
            sd = data_sample.std()
        if np.random.random() < 0.5 and gamma_range[0] < 1:
            gamma = np.random.uniform(gamma_range[0], 1)
        else:
            gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])
        minm = data_sample.min()
        rnge = data_sample.max() - minm
        data_sample = np.power(((data_sample - minm) / float(rnge + epsilon)), gamma) * rnge + minm
        if retain_stats:
            data_sample = data_sample - data_sample.mean() + mn
            data_sample = data_sample / (data_sample.std() + 1e-8) * sd
    else:
        for c in range(data_sample.shape[0]):
            if retain_stats:
                mn = data_sample[c].mean()
                sd = data_sample[c].std()
            if np.random.random() < 0.5 and gamma_range[0] < 1:
                gamma = np.random.uniform(gamma_range[0], 1)
            else:
                gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])
            minm = data_sample[c].min()
            rnge = data_sample[c].max() - minm
            data_sample[c] = np.power(((data_sample[c] - minm) / float(rnge + epsilon)), gamma) * float(rnge + epsilon) + minm
            if retain_stats:
                data_sample[c] = data_sample[c] - data_sample[c].mean() + mn
                data_sample[c] = data_sample[c] / (data_sample[c].std() + 1e-8) * sd
    if invert_image:
        data_sample = - data_sample
    return data_sample