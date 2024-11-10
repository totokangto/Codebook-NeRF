# Codebook-NeRF: Improving NeRF resolution based on codebook

This is the official implementation of our KSC 2024 paper `Codebook-NeRF: Improving NeRF resolution based on codebook`. Pull requests and issues are welcome.

### [Project Page](https://drawingprocess.github.io/Codebook-NeRF) | [Paper](https://arxiv.org/abs/2112.01759)

Abstract: *In this paper, we propose a new NeRF[1] method that can restore high-resolution details of low-resolution images without reference images. To this end, while maintaining the Super Resolution process of NeRF-SR[2], the codebook structure of VQ-VAE[3] is introduced to learn the patterns of high-resolution images and improve the definition technique. The number of embedding vectors in the codebook was increased to learn more high-resolution information, and it is trained to imitate high-resolution latent characteristics without reference images through Imaging Inference. As a result of the experiment, the proposed model maintained the PSNR performance of NeRF-SR[2], and succeeded in generating clear and detail-rich images.*

## Requirements
The codebase is tested on 
* Python 3.6.9 (should be compatible with Python 3.7+)
* PyTorch 1.8.1
* GeForce 1080Ti, 2080Ti, RTX 3090

Create a virtual environment and then run:
```
pip install -r requirements.txt
```

## Dataset
In our paper, we use the same dataset as in NeRF.
but Blender is still possible:
- [LLFF](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)
- [Blender](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)

However, our method is compatible to any dataset than can be trained on NeRF to perform super-resolution. Feel free to try out.

## Render the pretrained model
We provide pretrained models in [Google Drive](https://drive.google.com/drive/folders/1uLx2bbKzyJJMw3Nr3gOEo45acfpF0TUd?usp=sharing).

For supersampling, first download the pretrained models and put them under the `checkpoints/nerf-sr/${name}` directory, then run:
```bash
bash scripts/test_llff_downX.sh
```
or
```bash
bash scripts/test_blender_downX.sh
```
For the `${name}` parameter, you can directly use the one in the scripts. You can also modify it to your preference, then you have to change the script.

For refinement, run:
```bash
bash scripts/test_llff_refine.sh
```

## Train a new NeRF-SR model
Please check the configuration in the scripts. You can always modify it to your desired model config (especially the dataset path and input/output resolutions).
### Supersampling
```bash
bash scripts/train_llff_downX.sh
```
to train a 504x378 NeRF inputs.
or
```bash
bash scripts/train_blender_downX.sh
```

### Refinement
After supersampling and before refinement, we have to perform depth warping to find relevant patches, run:
```
python warp.py
```
to create `*.loc` files. An example of `*.loc` files can be found in the provided `fern` checkpoints (in the `30_val_vis` folder), which can be used directly for refinement.

After that, you can train the refinement model.
It is needed only for training with LLFF dataset:
```bash
bash scripts/train_llff_refine.sh
```


## Baseline Models
To replicate the results of baseline models, first train a vanilla NeRF using command:
```
bash scripts/train_llff.sh
```
or 
```
bash scrpts/train_blender.sh
```

For vanilla-NeRF, just test the trained NeRF under high resolutions using `bash scripts/test_llff.sh` or `bash scripts/test_blender.sh` (change the `img_wh` to your desired resolution). For NeRF-Bi, NeRF-Liif and NeRF-Swin, you need to super-resolve testing images with the corresponding model. The pretrained models of NeRF-Liif and NeRF-Swin can be found below:
- NeRF-Liif: We used the RDN-LIIF pretrained model. The download link can be found in the official [LIIF repo](https://github.com/yinboc/liif).
- NeRF-Swin: We used the "Real-world image SR" setting of [SwinIR](https://github.com/JingyunLiang/SwinIR) and the pretrained SwinIR-M model. Click to download the [x2](https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.pth) and [x4](https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth) model.


## Citation
If you consider our paper or code useful, please cite our paper:
```
@inproceedings{lee2024nerf,
  title={Codebook-NeRF : Improving NeRF resolution based on codebook},
  author={KangHyun Lee, SungJun Choi, Jung UK Kim},
  booktitle={KSC},
  year={2024}
}
```
