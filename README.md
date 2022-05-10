# TDD

The implementation of our CVPR2022 paper "Cross Domain Object Detection by Target-Perceived Dual Branch Distillation"

Paper: https://arxiv.org/pdf/2205.01291

2022.5.2
update the test code and model.
2022.5.10
update the joint-pretrain train code and model.
**Inference:**

`python train_net.py --eval-only --num-gpus 8 --config-file configs/TDD/***.yaml MODEL.WEIGHTS ***.pth`

**Released Weights**

For your convenience, the trained models are provided in this [link](https://pan.baidu.com/s/1efE8Y3Bl3arP7C-6MVOvHw). (BaiduYun code: e0lh)


**Joint Pretrain:**
The first stage of pretraining our detector with source images and target-like images.
`python train_net.py --num-gpus 8 --config-file configs/prertain_r50_FPN/faster_rcnn_R_50_FPN_focal_cross_bdd.yaml`

The weights are also preovided in the link above.
`python train_net.py --num-gpus 8 --eval-only --config-file *****.yaml MODEL.WEIGHTS ********.pth`

