# Naver_BoostCamp_NOTA Final Project

# 팀 소개

### 

|김광연|김민준|김병준|김상혁|서재명|
| :-: | :-: | :-: | :-: | :-: |
|![광연님](https://user-images.githubusercontent.com/59431433/217448461-bb7a37d4-f5d4-418b-a1b9-583b561b5733.png)|![민준님](https://user-images.githubusercontent.com/59431433/217448432-a3d093c4-0145-4846-a775-00650198fc2f.png)|![병준님](https://user-images.githubusercontent.com/59431433/217448424-11666f05-dda6-406d-95e8-47b3bab7c2f6.png)|![상혁2](https://user-images.githubusercontent.com/59431433/217448849-758c8e25-87db-4902-ab06-0aa8c359500c.png)|![재명님](https://user-images.githubusercontent.com/59431433/217448416-b2ba2070-6cfb-4829-a3bd-861f526cb74a.png)|

### Training
1. 데이터셋 준비
[ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/), [Tiny_ImageNet](https://paperswithcode.com/dataset/tiny-imagenet)
```
dataset
    |--ADE20K
    |--Tiny_ImageNet
```

2. 학습 시작
    - tiny_imagenet Pretraining
    ```bash
    bash dist_train.sh {사용하는 gpu 개수} \
        --data-path {tiny_imagenet path} \ # 이름에 tiny가 포함되어야함
        --output_dir {save dir path} \
        --batch-size {batch size per gpu } # default=128

    # example
    bash dist_train.sh 4 \
        --data-path /workspace/dataset/tiny_imagenet \
        --output_dir result/mod_segformer/ \
        --batch-size 64

    ```
    - ADE20K fine-tuning
    ```bash
    # 현재 디렉토리: /root/Naver_BoostCamp_NOTA
    python train.py \
        --data_dir {ADE20K의 path} \
        --device 0,1,2,3 \ # 환경에 맞게 수정 
        --save_path {save하고자 하는 dir의 path} \ 
        --pretrain {pretrain 모델 dir 혹은 .pth의 path} # .pth(pretrain의 output), dir(huggingface의 모델허브에서 제공하는 형태)
        --batch_size {batch size} # default=16
    ```

### Evaluation & FLOPs, 파라미터 개수 확인
- evaluate 수행

```bash
# phase를 통해 val 또는 test set 설정
python eval.py \ # eval.py 내의 model을 정의하는 코드 수정
    --data_dir {ADE20K의 path} \
    --pretrain {pretrain 모델 dir의 path}
```

- FLOPs, 파라미터 개수 확인

```bash
python util/get_flops_params.py \ # get_flops_params.py 내의 model을 정의하는 코드 수정
    --data_dir {ADE20K의 path}
```

# 프로젝트 소개

### 
![5번](https://user-images.githubusercontent.com/59431433/217443017-fa24917e-f63b-458f-878a-a8d20f21d606.png)
![6](https://user-images.githubusercontent.com/59431433/217443410-86cf320d-dbf4-4100-ac57-32bb51e59114.png)
![20](https://user-images.githubusercontent.com/59431433/217443765-e52a506e-a170-4d9d-9c5c-169956d0fadf.png)
![21](https://user-images.githubusercontent.com/59431433/217443838-6e966b29-a6e4-4e63-8798-7b06e2f25531.png)
![22](https://user-images.githubusercontent.com/59431433/217443528-748c1104-1a7b-46e5-bb03-56477da58132.png)
![23](https://user-images.githubusercontent.com/59431433/217443632-e13abfd0-e57e-4244-b1d6-203ed25c9bd1.png)

## 주요 참고자료

SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers

Feature Pyramid Networks for Object Detection

Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions

PVTv2: Improved Baselines with Pyramid Vision Transformer

Depth Estimation with Simplified Transformer

SegNeXt: Rethinking Convolutional Attention Design for Semantic Segmentation

Next-ViT: Next Generation Vision Transformer for Efficient Deployment in Realistic Industrial Scenarios

IS ATTENTION BETTER THAN MATRIX DECOMPOSITION

Efficient Attention: Attention with Linear Complexities

Swin Transformer: Hierarchical Vision Transformer using Shifted Windows

MLP-Mixer: An all-MLP Architecture for Vision

Lawin Transformer: Improving Semantic Segmentation Transformer with Multi-Scale Representations via Large Window Attention
