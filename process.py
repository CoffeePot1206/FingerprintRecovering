import os
import argparse

import PIL
from PIL import Image
import numpy as np
import random

import torch
from torchvision import transforms
from tqdm import tqdm
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from diffusers.utils.torch_utils import randn_tensor

from damage import *
from recover import recover

trans_ori = transforms.Compose([
    transforms.Resize(256, transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(256),
])

trans_rec = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--origin", type=str, default=None
    )
    parser.add_argument(
        "--noisy", action="store_true", default=False
    )
    parser.add_argument(
        "--recover", action="store_true", default=False
    )

    return parser.parse_args()

def load_origin(data_path, save_dir):
    i = 0
    print("loading original images...")
    for root, dir, names in os.walk(data_path):
        random.shuffle(names)
        names = names[:80]
        for name in tqdm(names):
            file = os.path.join(root, name)
            pre, suf = name.split(".")
            ori_im = Image.open(file, mode="r")
            ori_im = trans_ori(ori_im)
            ori_im.save(os.path.join(save_dir, f"ori_{i}.png"))

            i += 1

def corrupt(im, type="gaussian", std=None):

    if type == "gaussian":
        noi_im, mask = add_gaussian_noise(
            im,
            std=70,
            left_fraction=0.0,
            right_fraction=0.0,
            up_fraction=0.0,
            down_fraction=0.0,
            return_mask=True
        )
    elif type == "stain":
        ratio = 0.9
        l = np.random.random() * ratio
        r = ratio - l
        u = np.random.random() * ratio
        d = ratio - u
        noi_im, mask = stain(
            im,
            left_fraction=l,
            right_fraction=r,
            up_fraction=u,
            down_fraction=d,
            return_mask=True
        )
    return noi_im, mask

def noise_and_recover(type="gaussian", bs=16):
    batch = []
    ind_batch = []
    mask_batch = []
    print("generating noisy image...")
    for root, dir, names in os.walk(save_dir):
        for name in tqdm(names):
            if not "ori" in name:
                continue
            # corrupt
            file = os.path.join(root, name)
            index = name.split(".")[0][4:]
            ori_im = Image.open(file, 'r')
            ori_im = np.array(ori_im, dtype=np.uint8)
            # test shift 
            mean, var = ori_im.mean(), ori_im.var()

            noi_im, mask = corrupt(ori_im, type=type)
            Image.fromarray(noi_im, mode="L").save(os.path.join(save_dir, f"noisy_{index}.png"))
            # test shift 
            noi_im += mask * ((torch.randn(noi_im.shape).numpy() + mean) * var).clip(0, 255).astype(np.uint8)
            noi_im = Image.fromarray(noi_im, mode="L")
            # noi_im.save(os.path.join(save_dir, f"gaussian_{index}.png"))

            # recover
            if type == "gaussian": steps = 300
            elif type == "stain": steps = 500
            else: raise NotImplementedError

            noi_im = trans_rec(noi_im.convert("RGB"))
            batch.append(noi_im)
            ind_batch.append(index)
            mask_batch.append(mask)
            if len(batch) == bs:
                batch = torch.stack(batch)
                mask_batch = np.stack(mask_batch)[:, np.newaxis, :, :]
                # print(batch.shape)
                # print(mask_batch.shape)
                # exit()
                rec = recover(model, batch, mask=mask_batch, steps=steps, batch_size=bs).images
                for ind, img in zip(ind_batch, rec):
                    img.save(os.path.join(save_dir, f"recover_{ind}.png"), mode="L")
                batch = []
                ind_batch = []
                mask_batch = []
    if len(batch) > 0:
        batch = torch.stack(batch)
        rec = recover(model, batch, batch_size=bs).images
        for ind, img in zip(ind_batch, rec):
            img.save(os.path.join(save_dir, f"recover_{ind}.png"), mode="L")
        batch = []
        ind_batch = []

def recover_noisy(bs):
    batch = []
    ind_batch = []
    print("recovering...")
    for root, dir, names in os.walk(save_dir):
        for name in tqdm(names):
            if not "noisy" in name:
                continue
            file = os.path.join(root, name)
            ind_batch.append(name.split(".")[0][6:])
            noi_im = Image.open(file, mode='r')
            noi_im = trans_rec(noi_im.convert("RGB"))
            batch.append(noi_im)
            if len(batch) == bs:
                batch = torch.stack(batch)
                rec = recover(model, batch, batch_size=bs).images
                for ind, img in zip(ind_batch, rec):
                    img.save(os.path.join(save_dir, f"recover_{ind}.png"), mode="L")
                batch = []
                ind_batch = []
    if len(batch) > 0:
        batch = torch.stack(batch)
        rec = recover(model, batch, batch_size=bs).images
        for ind, img in zip(ind_batch, rec):
            img.save(os.path.join(save_dir, f"recover_{ind}.png"), mode="L")
        batch = []
        ind_batch = []

if __name__ == "__main__":
    args = parse_args()

    base_path = os.path.dirname(__file__)
    data_path = os.path.join(base_path, 'Db2_b')
    # data_path = args.origin

    save_dir = os.path.join(base_path, "test")
    os.makedirs(save_dir, exist_ok=True)

    # load origin
    load_origin(data_path, save_dir)

    # load model and recover
    model = DDPMPipeline.from_pretrained(
        "./models",
    ).to("cuda")
    model.unet.requires_grad_(False)

    # noise_and_recover(type="stain", bs=16)
    noise_and_recover(type="gaussian", bs=16)