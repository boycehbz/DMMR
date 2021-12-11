# \[3DV2021\] Dynamic Multi-Person Mesh Recovery From Uncalibrated Multi-View Cameras (DMMR)
The code for 3DV 2021 paper "Dynamic Multi-Person Mesh Recovery From Uncalibrated Multi-View Cameras"<br>
\[[Paper]\](https://arxiv.org/pdf/2110.10355.pdf)  \[[Video]\](https://www.bilibili.com/video/BV1Qq4y1d78S)

![figure](/images/video.gif)

## Dependencies
Windows or Linux, Python3

```conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch```<br>
```pip install -r requirements.txt```


## Getting Started
Step1: Download the official SMPL model from [SMPLify website](http://smplify.is.tuebingen.mpg.de/) and put it in ```models/smpl``` folder. (see [models/smpl/readme.txt](./models/smpl/readme.txt))<br>

Step2: Download the test data and trained motion prior from [here](http://smplify.is.tuebingen.mpg.de/) (**extraction code \[jhwp\]**) and put them in ```data``` folder.<br>

Step3: Run ```python main.py --config cfg_files/fit_smpl.yaml```<br>

You can visualize the motions and cameras in optimization with the command:<br>
```python main.py --config cfg_files/fit_smpl.yaml --visualize true```

<div align="center" width="100%">
      <img style="max-height: 300px; max-width: 300px;" align="center" width="100%" class="image" src="/images/optimize.gif" >
</div>

The code can also be used for motion capture with known cameras:<br>
```python main.py --config cfg_files/fit_smpl.yaml --opt_cam false```


## Results
The fitted results will be saved in ```output```<br>
You can visualize the estimated extrinsic camera parameters by running:<br>
```python viz_cameras.py```


## Citation
If you find this code useful for your research, please consider citing the paper.
```
@inproceedings{huang2021dynamic,
      title={Dynamic Multi-Person Mesh Recovery From Uncalibrated Multi-View Cameras}, 
      author={Buzhen Huang and Yuan Shu and Tianshu Zhang and Yangang Wang},
      year={2021},
      booktitle={3DV},
}
```

## Acknowledgments
Some of the code are based on the following works. We gratefully appreciate the impact it had on our work.<br>
[SMPLify-x](https://github.com/vchoutas/smplify-x)<br>
[SPIN](https://github.com/nkolot/SPIN)<br>
[EasyMocap](https://github.com/zju3dv/EasyMocap)<br>
[MvSMPLfitting](https://github.com/boycehbz/MvSMPLfitting)<br>