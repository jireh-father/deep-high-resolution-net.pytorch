# Landmark Detection Using Deep High-Resolution Representation Learning for Human Pose Estimation (CVPR 2019)

## Install
### conda 가상환경 생성
```bash
conda create --name hrnet python=3.6
source activate hrnet
```

### cuda 10.0 설치
- 아래 사이트 참고, 꼭 10.0 이어야만 돌아갑니다.
- https://developer.nvidia.com/cuda-10.0-download-archive?target_os=Linux

### pytorch 설치
```bash
pip install -U https://download.pytorch.org/whl/cu100/torch-1.0.0-cp36-cp36m-linux_x86_64.whl
pip install torchvision==0.2.1
```

- python 실행 후 아래와 버전이 동일한지 확인하세요.
```bash
python
>>> import torch
>>> torch.__version__
'1.0.0'
>>> torch.version.cuda
'10.0.130'
```

# hrnet 소스 다운로드
```bash
cd $YOUR_SOURCE_ROOT
git clone https://github.com/jireh-father/deep-high-resolution-net.pytorch.git
cd deep-high-resolution-net.pytorch
```

### 파이썬 라이브러리 설치
```bash
pip install -r requirements.txt
```

### hrnet 설치
```bash
export CUDAHOME=/usr/local/cuda-10.0
cd lib
make
```

### cocoapi 설치
```bash
cd $YOUR_SOURCE_ROOT
git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
cd cocoapi/PythonAPI
```

- cocoeval 소스코드 변경
```bash
vi pycocotools/cocoeval.py
```
- 206 line근처를 아래 코드와 동일하게 바꿔주세요.
```python
ious = np.zeros((len(dts), len(gts)))
#sigmas = p.kpt_oks_sigmas
sigmas = np.array([1.0, 1.0, 1.0]) / 10.0
vars = (sigmas * 2)**2
```
- 설치
```bash
python setup.py install --user
```

### 학습 세팅
```bash
cd $YOUR_SOURCE_ROOT/deep-high-resolution-net.pytorch
mkdir output 
mkdir log
```

### 데이터셋 준비
```bash
wget https://font-recognizer-bucket.s3.us-east-2.amazonaws.com/resource/ai_for_pet/20201005/data.zip
unzip data.zip
```

# 학습
```bash
python -u tools/train.py --cfg experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml
```

# 시각화(학습 후)
```bash
python visualization/plot_coco.py --prediction output/coco/pose_hrnet/w32_256x192_adam_lr1e-3/results/keypoints_val2017_results_0.json --save-path visualization/results
```
- 시각화 결과는 $YOUR_SOURCE_ROOT/deep-high-resolution-net.pytorch/visualization/ 안에 있습니다.
