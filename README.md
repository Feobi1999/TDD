# TDD

The implementation of our CVPR2022 paper "Cross Domain Object Detection by Target-Perceived Dual Branch Distillation"

Paper: https://arxiv.org/pdf/2205.01291

2022.5.2
update the test code and model. \
2022.5.10
update the  train code and model.\

**The enviroment and data preparation will be updated soon.**

**Inference:**

`python train_net.py --eval-only --num-gpus 8 --config-file configs/TDD/***.yaml MODEL.WEIGHTS ***.pth`

**Released Weights**

For your convenience, the trained models are provided in this [link](https://pan.baidu.com/s/1efE8Y3Bl3arP7C-6MVOvHw). (BaiduYun code: e0lh)


**Joint Pretrain:**\
The first stage of pretraining our detector with source images and target-like images.
`python train_net.py --num-gpus 8 --config-file configs/prertain_r50_FPN/faster_rcnn_R_50_FPN_focal_cross_bdd.yaml`

The weights are also preovided in the link above.
`python train_net.py --num-gpus 8 --eval-only --config-file *****.yaml MODEL.WEIGHTS ********.pth`\

**TDD Train**\
`python train_net.py --num-gpus 8 --config-file configs/TDD/faster_rcnn_R_50_FPN_foggy.yaml`\

Note: Now only support that one gpu two images ( when GPUS=8, set IMG_PER_BATCH_LABEL: 8
IMG_PER_BATCH_UNLABEL: 8)

The two stage can be trained with one config file by modify the ` BURN_UP_STEP` which means the step of joint pretrain.