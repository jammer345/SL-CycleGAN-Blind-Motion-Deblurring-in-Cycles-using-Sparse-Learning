# SL-CycleGAN-Blind-Motion-Deblurring-in-Cycles-using-Sparse-Learning (2022)
[![arXiv Prepring](https://img.shields.io/badge/arXiv-Preprint-lightgrey?logo=arxiv)](https://arxiv.org/abs/2111.04026)
 
 Ali Syed Saqlain<sup>1</sup>, Li Yun Wang<sup>2</sup>, & Fang Fang<sup>1</sup>
 <br/>
 <sup>1 </sup>North China Electric Power University, Beijing
 <br/>
 <sup>2 </sup>Portland State University, USA
 
## Abstract
In this paper, we introduce an end-to-end generative adversarial network (GAN) based on sparse learning for single image motion deblurring, which we called SL-CycleGAN. For the first time in image motion deblurring, we propose a sparse ResNet-block as a combination of sparse convolution layers and a trainable spatial pooler k-winner based on HTM (Hierarchical Temporal Memory) to replace non-linearity such as ReLU in the ResNet-block of SL-CycleGAN generators. Furthermore, we take our inspiration from the domain-to-domain translation ability of the CycleGAN, and we show that image deblurring can be cycle-consistent while achieving the best qualitative results. Finally, we perform extensive experiments on popular image benchmarks both qualitatively and quantitatively and achieve the highest PSNR of 38.087 dB on GoPro dataset, which is 5.377 dB better than the most recent deblurring method.
## Network Architecture 
![](imgs/arc.png)
