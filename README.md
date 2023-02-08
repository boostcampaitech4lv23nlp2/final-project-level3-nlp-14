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
![d7ab99c2_1](https://user-images.githubusercontent.com/59431433/217456663-608f0e4d-47e2-4195-a265-72db62a6839f.png)
![d7ab99c2_2](https://user-images.githubusercontent.com/59431433/217456668-fb28e993-6ca7-41eb-a433-32ee1bcac5c1.png)
![d7ab99c2_3](https://user-images.githubusercontent.com/59431433/217456669-4a568328-4a85-495e-a95d-233a84e0db75.png)
![d7ab99c2_4](https://user-images.githubusercontent.com/59431433/217456672-78d11b66-d9a8-460e-88b5-4494921b6b86.png)
![d7ab99c2_5](https://user-images.githubusercontent.com/59431433/217456674-91051a7f-d581-46c1-a424-0201ce5c1c09.png)
![d7ab99c2_6](https://user-images.githubusercontent.com/59431433/217456675-6430f896-9a65-4def-a269-97c723cfcde3.png)

## 주요 참고자료

[SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203)

[Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144)

[Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions](https://arxiv.org/abs/2102.12122)

[PVTv2: Improved Baselines with Pyramid Vision Transformer](https://arxiv.org/abs/2106.13797)

[Depth Estimation with Simplified Transformer](https://arxiv.org/abs/2204.13791)

[SegNeXt: Rethinking Convolutional Attention Design for Semantic Segmentation](https://arxiv.org/abs/2209.08575)

[Next-ViT: Next Generation Vision Transformer for Efficient Deployment in Realistic Industrial Scenarios](https://arxiv.org/abs/2207.05501)

[IS ATTENTION BETTER THAN MATRIX DECOMPOSITION](https://arxiv.org/abs/2109.04553)

[Efficient Attention: Attention with Linear Complexities](https://arxiv.org/abs/1812.01243)

[Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)

[MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/abs/2105.01601)

[Lawin Transformer: Improving Semantic Segmentation Transformer with Multi-Scale Representations via Large Window Attention](https://arxiv.org/abs/2201.01615)
