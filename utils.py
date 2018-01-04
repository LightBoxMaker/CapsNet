import os
import scipy
import scipy.io
import numpy as np
import tensorflow as tf
import pickle

from config import cfg



    



# Loading affNIST dataset
def load_mnist_affNIST(is_training):
    if is_training:
        dat_path = 'img_data/train_val_rotate_scale.p'
        train_dict = pickle.load(open(dat_path, 'rb'))
        
        ex_imgs = train_dict['ex_imgs'].transpose((3,2,0,1))
        ex_imgs_label = train_dict['ex_imgs_label'].transpose((1,0))
        imgs_scale = train_dict['imgs_scale'].transpose((3,2,0,1))
        imgs_rotated = train_dict['imgs_rotated_full'].transpose((3,2,0,1))
        imgs_label = train_dict['imgs_label'].transpose((1,0))

        return ex_imgs, ex_imgs_label, imgs_scale,imgs_rotated,imgs_label
    else:
        dat_path = 'img_data/test_just_centered.p'
        test_dict = pickle.load(open(dat_path, 'rb'))
        Xmat = test_dict['imgs']
        Ymat = test_dict['labels']
        Ymat = Ymat.reshape(Ymat.shape[0],)
        teCtrl = test_dict['ctrl']

        return Xmat, Ymat, teCtrl








def save_images(imgs, size, path):
    '''
    Args:
        imgs: [batch_size, image_height, image_width]
        size: a list with tow int elements, [image_height, image_width]
        path: the path to save images
    '''
    imgs = (imgs + 1.) / 2  # inverse_transform
    return(scipy.misc.imsave(path, mergeImgs(imgs, size)))


def mergeImgs(images, size):
    h, w = images.shape[1], images.shape[2]
    imgs = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        imgs[j * h:j * h + h, i * w:i * w + w, :] = image

    return imgs


if __name__ == '__main__':
    X, Y = load_mnist(cfg.dataset, cfg.is_training)
    print(X.get_shape())
    print(X.dtype)
