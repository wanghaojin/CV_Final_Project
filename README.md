## CV_Final_Project

 This project utilizes MAE to reduce the memory usage based on composite regularization method when implementing video style transfer, and uses SAM2 for selective style transfer of images.

### Catalog

- [x] Video and image Style transfer
- [x] MAE predict images
- [x] Selective style transfer

### Video and image Stlye transfer
This is the video style transfer method we refer to
[Consistent Video Style Transfer via Compound Regularization](https://aaai.org/papers/12233-consistent-video-style-transfer-via-compound-regularization/):
```
@inproceedings{wang2020consistent,
  title={Consistent video style transfer via compound regularization},
  author={Wang, Wenjing and Xu, Jizheng and Zhang, Li and Wang, Yue and Liu, Jiaying},
  booktitle={Proceedings of the AAAI conference on artificial intelligence},
  volume={34},
  number={07},
  pages={12233--12240},
  year={2020}
}
```

In the **test** folder of ReReVST, we use the model to implement style transfer for each frame images of the video

### MAE predict images

This is the MAE method we refer to [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377):
```
@article{DBLP:journals/corr/abs-2111-06377,
  author= {Kaiming He and Xinlei Chen and Saining Xie and Yanghao Li and Piotr Doll{\'{a}}r and Ross B. Girshick},
  title= {Masked Autoencoders Are Scalable Vision Learners},
  journal= {CoRR},
  volume= {abs/2111.06377},
  year= {2021},
  url= {https://arxiv.org/abs/2111.06377},
  eprinttype= {arXiv},
  eprint= {2111.06377},
  timestamp= {Tue, 16 Nov 2021 12:12:31 +0100},
  biburl= {https://dblp.org/rec/journals/corr/abs-2111-06377.bib},
  bibsource= {dblp computer science bibliography, https://dblp.org}
}
```
In the **MAE** folder of ReReVST, we have attempted to reconstruct images using MAE under different mask conditions
* Separate each frame of the video and merge it with the image to form a video in **imd_video_transform** folder
* Implement different mask conditions for images in **mask** folder
* Attempt to restore image size and MAE in **test.ipynb**

Additionally, the attempt to divide images into 4 patches and stlye transfer is completed in **process.ipynb**

### Selective style transfer
This is the SAM2 method we refer to [SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714):
```
@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Doll{\'a}r, Piotr and Feichtenhofer, Christoph},
  journal={arXiv preprint arXiv:2408.00714},
  url={https://arxiv.org/abs/2408.00714},
  year={2024}
}
```
In the **Selective** folder, we have completed the selective style conversion after MAE reconstruction of fox images (mask ratio ≈ 38.3%, retaining the middle 64 plus random 57 patches)
