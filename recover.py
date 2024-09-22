import torch
import torch.nn.functional as F
from torchvision import transforms

from PIL import Image
import PIL
import os
import numpy as np

import diffusers
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput

from damage import gaussian_blur
from damage import *

def recover(
    model,
    noi_im,
    mask = None,
    steps: int = 200,
    batch_size: int = 1,
    generator = None,
    num_inference_steps: int = 1000,
    output_type = "pil",
    return_dict: bool = True,
):
    # if isinstance(model.unet.config.sample_size, int):
    #     image_shape = (
    #         batch_size,
    #         model.unet.config.in_channels,
    #         model.unet.config.sample_size,
    #         model.unet.config.sample_size,
    #     )
    # else:
    #     image_shape = (batch_size, model.unet.config.in_channels, *model.unet.config.sample_size)

    # if model.device.type == "mps":
    #     # randn does not work reproducibly on mps
    #     image = randn_tensor(image_shape, generator=generator)
    #     image = image.to(model.device)
    # else:
    #     image = randn_tensor(image_shape, generator=generator, device=model.device)

    if isinstance(mask, np.ndarray):
        mask = torch.tensor(mask, dtype=torch.uint8, device="cuda")
    
    image = noi_im.requires_grad_(False).to("cuda")
    noi_im = noi_im.to("cuda")

    # set step values
    model.scheduler.set_timesteps(num_inference_steps)
    # print(model.scheduler.timesteps)
    # exit()

    for t in model.progress_bar(model.scheduler.timesteps[-steps:]):
        # 1. predict noise model_output
        model_output = model.unet(image, t).sample
        # model_output *= mask

        # 2. compute previous image: x_t -> x_t-1
        image = model.scheduler.step(model_output, t, image, generator=generator).prev_sample
        if mask is not None:
            image = noi_im * (1 - mask) + image * mask

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).detach().numpy()
    if output_type == "pil":
        image = model.numpy_to_pil(image)

    if not return_dict:
        return (image,)

    return ImagePipelineOutput(images=image)

if __name__ == "__main__":
    base_path = os.path.dirname(__file__)

    # load original image
    ori_im = Image.open(os.path.join(base_path, 'origin.png'))
    ori_im = np.array(ori_im)

    augmentations = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    # get noisy image
    # noi_im, mask = gaussian_blur(im=ori_im, return_mask=True)
    # noi_im, mask = gaussian_blur(ori_im, sigma=3, left_fraction=0.1, right_fraction=0.2, up_fraction=0.5, down_fraction=0.2, return_mask=True)
    noi_im, mask = add_gaussian_noise(
        ori_im, 
        std=70, 
        left_fraction=0.1,
        right_fraction=0.1,
        up_fraction=0.1,
        down_fraction=0.1,
        return_mask=True
    )

    # noi_im, mask = stain(
    #     ori_im,
    #     left_fraction=0.3,
    #     right_fraction=0.6,
    #     up_fraction=0.5,
    #     down_fraction=0.4,
    #     return_mask=True
    # )
    # stained_im = noi_im.copy()
    # noi_im += mask * (randn_tensor(noi_im.shape).numpy() * 255).astype(np.uint8)

    # stained_im = Image.fromarray(stained_im)
    # stained_im.save('stained.png')


    noi_im = Image.fromarray(noi_im)
    noi_im.save('noisy.png')
    noi_im = augmentations(noi_im.convert("RGB")).unsqueeze(0)

    # mask *= 255
    Image.fromarray(mask * 255, mode="L").save('mask.png')
    # exit()
    # mask = torch.tensor(mask)
    # print(noi_im.shape)
    # exit()

    # load model
    model = DDPMPipeline.from_pretrained(
        "./models/",
    ).to("cuda")
    model.unet.requires_grad_(False)
    # example = model()
    rec_im = np.array(recover(model, noi_im, mask, steps=300).images[0].convert("L"), dtype=np.uint8)
    # rec_im = ori_im * (1 - mask) + rec_im * mask
    rec_im = Image.fromarray(rec_im)
    rec_im = rec_im.convert("L")
    rec_im.save('recover.png')