# write a function that adds gaussian noise to the image. 
# write is as a function and use it in the main code

# with this code:
import os
import random
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

def add_gaussian_noise(im, std=100, left_fraction=0.1, right_fraction=0.2, up_fraction=0.5, down_fraction=0.2, return_mask=False):
    # add gaussian noise to the image
    # im: a numpy array
    # noise: gaussian, 
    # the noise is added to the image
    # the image is returned
    
    # get the shape of the image
    shape = im.shape
    # generate the noise
    noise = np.random.normal(0, std, shape) 
    # print("noise:", noise)

    mask = np.zeros_like(im, dtype=np.uint8)
    for i in range(int(up_fraction * shape[0]), int((1 - down_fraction) * shape[0])):
        for j in range(int(left_fraction * shape[1]), int((1 - right_fraction) * shape[1])):
            mask[i, j] = 1
    
    # save the image noise
    # Image.fromarray(np.clip(noise + 100, 0, 255).astype(np.uint8)).save('noise.png')
    
    # add the noise to the image
    # print("im:", im)
    im = im + noise * mask
    # cut into the range [0, 255] and convert to integer
    im[im < 0] = 0
    im[im > 255] = 255
    im = im.astype(np.uint8)
    # print("im:", im)
    if return_mask:
        return im, mask
    return im

def gaussian_blur(im, sigma=1, left_fraction=0.1, right_fraction=0.1, up_fraction=0.1, down_fraction=0.1, return_mask=False):
    # add gaussian blur to the image
    # im: a numpy array
    # sigma: the standard deviation of the gaussian kernel
    # the new image is returned
    
    # get the shape of the image
    shape = im.shape
    # generate the kernel, 2D gaussian kernel
    # 2D array of size 6*sigma+1
    size = int(6 * sigma + 1)
    kernel = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            kernel[i, j] = np.exp(-((i - size // 2)**2 + (j - size // 2)**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    # print("kernel:", kernel)
    
    # blur the image
    im_blur = im.copy()
    mask = np.zeros_like(im, dtype=np.uint8)
    for i in range(int(up_fraction * shape[0]), int((1 - down_fraction) * shape[0])):
        for j in range(int(left_fraction * shape[1]), int((1 - right_fraction) * shape[1])):
            mask[i, j] = 1
            for k in range(3):
                im_blur[i, j] = (im[max(0, i - size // 2):min(shape[0], i + size // 2 + 1), 
                                   max(0, j - size // 2):min(shape[1], j + size // 2 + 1)] * kernel[:min(shape[0], i + size // 2 + 1) - max(0, i - size // 2), 
                                                                                                       :min(shape[1], j + size // 2 + 1) - max(0, j - size // 2)]).sum()
    im_blur = np.clip(im_blur, 0, 255).astype(np.uint8)
    
    if return_mask:
        return im_blur, mask
    return im_blur

def add_duplication(im_ori, im_remaining=None, transparency=0.1, pos_fraction=0.5):
    # add a new fingerprint on one image, to simulate the case where the fingerprint is not clear
    # initial image: im_ori, remaining image: img_remaining. if no remaining image, use im_ori. both: numpy array
    # transparency: now much the remaing fingerprint is visible. 0 is no remaining, 1 is remainingn as the same
    # pos_fraction: how much the remaining image can be moved. 
    # the new image is returned
    
    # get the shape of the image
    shape = im_ori.shape
    # generate the noise as the remaining image
    if im_remaining is None:
        im_remaining = im_ori.copy()
    im_remaining = 255 - im_remaining
    im_remaining = im_remaining * transparency
    # cut the im_remaining into the size of the im_ori, if the im_remaining is larger. if smaller: fill with 0
    # make sure the size of the im_remaining is the same as the im_ori now
    im_remaining_after = np.zeros(shape).astype(np.uint8)
    im_remaining_after[:min(shape[0], im_remaining.shape[0]), :min(shape[1], im_remaining.shape[1])] = im_remaining[:min(shape[0], im_remaining.shape[0]), :min(shape[1], im_remaining.shape[1])]
    
    
    # random position. if pos is (0,0), then do not move the im_remaining, pos can be negative or positive
    pos_range = (int(pos_fraction * shape[0]), int(pos_fraction * shape[1]))
    pos = (random.randint(- pos_range[0], pos_range[0]), random.randint(- pos_range[1], pos_range[1]))
    
    # move the im_remaining to the position
    img = np.zeros(shape)
    img[max(0, pos[0]):min(shape[0], shape[0]+pos[0]), max(0, pos[1]):min(shape[1], shape[1]+pos[1])] = im_remaining_after[max(0, -pos[0]):min(shape[0], shape[0]-pos[0]), max(0, -pos[1]):min(shape[1], shape[1]-pos[1])]
    img = np.clip(img, 0, 255).astype(np.uint8)
    
    # # save the image noise
    # Image.fromarray(img).save('noise.png')
    
    # add the noise to the image
    # use the original image as the background, if minus id negative, set to 0
    im = np.zeros(shape).astype(np.uint8)
    im[im_ori > img] = im_ori[im_ori > img] - img[im_ori > img]
    # print("im", im)
    return im

def darken(im, factor=0.5, r=0.5):
    # darken the fingerprint inside a circle region with radius R = r * min(shape[0], shape[1]) / 2
    # im: numpy array
    # R: radius of the circle
    # factor: how much the image is darkened. 0 is the same, 1 is black (for fingerprint part, not all the region)
    # the new image is returned
    
    # get the shape of the image
    shape = im.shape
    R = r * min(shape[0], shape[1]) / 2
    
    # random position of the circle
    pos = (random.randint(0, shape[0]), random.randint(0, shape[1]))
    
    # darken the image inside the circle
    im[(im < 240) & ((np.arange(shape[0])[:, None] - pos[0])**2 + (np.arange(shape[1]) - pos[1])**2 < R**2)] \
    = (im[(im < 240) & ((np.arange(shape[0])[:, None] - pos[0])**2 + (np.arange(shape[1]) - pos[1])**2 < R**2)] * factor).astype(np.uint8)
                
    return im

def stain(im, left_fraction=0.1, right_fraction=0.1, up_fraction=0.1, down_fraction=0.1, return_mask=False):
    shape = im.shape
    mask = np.zeros_like(im, dtype=np.uint8)
    for i in range(int(up_fraction * shape[0]), int((1 - down_fraction) * shape[0])):
        for j in range(int(left_fraction * shape[1]), int((1 - right_fraction) * shape[1])):
            im[i, j] = 0
            mask[i, j] = 1

    if return_mask:
        return im, mask
    return im


# write: if __name__ == '__main__':

if __name__ == '__main__':
    path = './data/'
    trans = transforms.Compose([
        transforms.Resize(256, transforms.InterpolationMode.BILINEAR),
        transforms.RandomCrop(256),
        # transforms.ToTensor(),
    ])
    # read one picture in ./data_raw/jpng1000, 00002357_J_1000_palm_02.png
    im = Image.open(os.path.join(path, '00002357_J_1000_palm_02.png'))
    # convert to numpy array
    # im = np.array(im, dtype=np.uint8)
    # im = torch.tensor(im, dtype=torch.uint8)
    im = trans(im)
    im = np.array(im)

    
    # add gaussian noise
    noi_im = add_gaussian_noise(im, std=70, left_fraction=0.1, right_fraction=0.2, up_fraction=0.5, down_fraction=0.2)
    
    # # add gaussian blur
    # noi_im = gaussian_blur(im, sigma=3, left_fraction=0.1, right_fraction=0.2, up_fraction=0.5, down_fraction=0.2)
    
    # # add duplication
    # noi_im = add_duplication(im_ori=im, 
    #                      im_remaining=np.array(Image.open(path + 'jpng1000/00002357_J_1000_palm_07.png')), 
    #                      transparency=0.5, pos_fraction=0.2)
    
    # # darken
    # noi_im = darken(im)
    
    # save the image
    noi_im = Image.fromarray(noi_im)
    im = Image.fromarray(im)
    im.save('origin.png')
    noi_im.save('noisy.png')
    