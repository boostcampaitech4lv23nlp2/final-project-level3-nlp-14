# Naver_BoostCamp_NOTA Final Project

|  팀원  | 역할 |
|  :---------:  | :--------- |
|  김광연  | Next-Vit 논문을 참하여 인코더에 convolution layer를 활용하여 Hybrid구조로 변경하는 실험 |
|  김민준  | SegFormer의 Efficient Self Attention의 SRA를 PVTv2의 LSRA로 변경하는 실험,<br>SegFormer의 Mix-FFN에 Batch Normalization, Convolution layer를 추가한 DEST Mix-FFN으로 변경하는 실험<br>SegFormer의 encoder에 Recursive skip-connection을 적용하여 효율적인 학습이 가능하도록 변경하는 실험<br>SegFormer의 Decoder의 표현력을 높이기 위해 LawinASPP를 적용하는 실험 |
|  김병준  | SegFormer의 Self Attention module을 SegNeXt의 MSCA로 변경하는 실험<br>LSRA에 Efficient Attention을 적용한 LSREA 모듈 설계 |
|  김상혁  | SegFormer의 encoder를 SegNeXt의 MSCAN으로 변경하고 docoder를 SegNeXt의 Hamburger로 변경하는 실험<br>SegFormer의 불필요한 레이어를 제거하고 SegNeXt의 decoder를 경량화하는 실험 |
|  서재명  | Window Attention(Swin Transformer 참조)로 encoder 경량화 하는 실험 |

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
