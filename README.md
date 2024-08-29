# \[3DV2021\] Dynamic Multi-Person Mesh Recovery From Uncalibrated Multi-View Cameras (DMMR)
The code for 3DV 2021 paper "Dynamic Multi-Person Mesh Recovery From Uncalibrated Multi-View Cameras"<br>
[Buzhen Huang](http://www.buzhenhuang.com/), [Yuan Shu](https://scholar.google.com/citations?hl=zh-CN&user=hkvxsI8AAAAJ), [Tianshu Zhang](https://scholar.google.com/citations?user=S5M_CncAAAAJ&hl=zh-CN), [Yangang Wang](https://www.yangangwang.com/)<br>
\[[Paper](https://arxiv.org/pdf/2110.10355.pdf)\]  \[[Video](https://www.bilibili.com/video/BV1Qq4y1d78S)\]

![figure](/images/teaser.jpg)


![figure](/images/video.gif)

## Dependencies
Windows or Linux, Python3.7

```conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch```<br>
```pip install -r requirements.txt```


## Getting Started
**Step1:**<br>
Download the official SMPL model from [SMPLify website](http://smplify.is.tuebingen.mpg.de/) and put it in ```models/smpl```. (see [models/smpl/readme.txt](./models/smpl/readme.txt))<br>

**Step2:**<br>
Download the test data and trained motion prior from [Google Drive](https://drive.google.com/file/d/1JyHiTfeHTqKJW6fvtNec983fF25VdPit/view) or [Baidu Netdisk](https://pan.baidu.com/s/1jHXXp7xuPAYWoivD60gPdA) (**extraction code \[jomn\]**) and put them in ```data```.<br>

**Step3:**<br>
Run 
```bash
python main.py --config cfg_files/fit_smpl.yaml
```

You can visualize the motions and cameras in optimization with the command:<br>
```bash
python main.py --config cfg_files/fit_smpl.yaml --visualize true
```

<div align="center" width="100%">
      <img style="max-height: 300px; max-width: 300px;" align="center" width="100%" class="image" src="/images/optimize.gif" >
</div>

The code can also be used for motion capture with known cameras:<br>
```bash
python main.py --config cfg_files/fit_smpl.yaml --opt_cam false
```


## Results
The fitted results will be saved in ```output```.<br>
You can visualize the estimated extrinsic camera parameters by running:<br>
```bash
python viz_cameras.py
```

![figure](/images/results.jpg)

## Citation
If you find this code useful for your research, please consider citing the paper.
```
@article{huang2023simultaneously,
  title={Simultaneously Recovering Multi-Person Meshes and Multi-View Cameras with Human Semantics},
  author={Huang, Buzhen and Ju, Jingyi and Shu, Yuan and Wang, Yangang},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2023},
  publisher={IEEE}
}
@inproceedings{huang2021dynamic,
      title={Dynamic Multi-Person Mesh Recovery From Uncalibrated Multi-View Cameras}, 
      author={Buzhen Huang and Yuan Shu and Tianshu Zhang and Yangang Wang},
      year={2021},
      booktitle={3DV},
}
```

## Acknowledgments
Some of the code are based on the following works. We gratefully appreciate the impact it has on our work.<br>
[SMPLify-x](https://github.com/vchoutas/smplify-x)<br>
[SPIN](https://github.com/nkolot/SPIN)<br>
[EasyMocap](https://github.com/zju3dv/EasyMocap)<br>
[MvSMPLfitting](https://github.com/boycehbz/MvSMPLfitting)<br>