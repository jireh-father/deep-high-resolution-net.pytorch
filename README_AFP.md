# Landmark Detection Using Deep High-Resolution Representation Learning for Human Pose Estimation (CVPR 2019)

## Install
```bash
conda create --name hrnet python=3.6
source activate hrnet
```
install cuda 10.0 (아래 사이트 참고, 꼭 10.0 이어야만 돌아갑니다.)
https://developer.nvidia.com/cuda-10.0-download-archive?target_os=Linux
```bash
pip install -U https://download.pytorch.org/whl/cu100/torch-1.0.0-cp36-cp36m-linux_x86_64.whl
pip install torchvision==0.2.1
```
python 실행 후 아래와 버전이 동일한지 확인하세요.
```bash
python
>>> import torch
>>> torch.__version__
'1.0.0'
>>> torch.version.cuda
'10.0.130'

```
```bash
cd $YOUR_SOURCE_DIR
git clone https://github.com/jireh-father/deep-high-resolution-net.pytorch.git
cd deep-high-resolution-net.pytorch
pip install -r requirements.txt
cd lib
export CUDAHOME=/usr/local/cuda-10.0
make
```

```bash
cd $YOUR_SOURCE_DIR
git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
cd cocoapi/PythonAPI
make install
```

```bash
cd $YOUR_SOURCE_DIR/deep-high-resolution-net.pytorch
mkdir output 
mkdir log

```

```bash
wget 
unzip data.zip


```