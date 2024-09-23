# Corrupted Fingerprint Recovering via Diffusion Model

This is a course project for Computer Vision, lectured by Prof. Yang Gao. For more details about the project, see `report.pdf`.

## Train

- **Training Scripts:** we use [diffusers](https://github.com/huggingface/diffusers) to train our model. To be specific, the script we use is `diffusers/examples/unconditional_image_generation/train_unconditional.py`.
- **Data:** we mainly use [NIST special database](https://www.nist.gov/itl/iad/image-group/nist-special-database-302) for model training, and [FVC2000](http://bias.csr.unibo.it/fvc2000/) for testing.

## Test

For testing, simply run the script `process.py`. It will automatically corrupt the fingerprint images in test set, and recover it using the pretrained model. 
