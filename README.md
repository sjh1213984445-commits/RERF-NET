conda create -n CoreNet python=3.8
conda activate CoreNet
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install mmengine==0.10.4
pip install mmcv==2.0.1
pip install -r requirements.txt
pip install "opencv-python-headless<4.3"
python3 -m pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ mmdet==3.1.0
pip install pyquaternion==0.9.9
pip install trimesh==3.23.0
pip install lyft-dataset-sdk
pip install nuscenes-devkit==1.1.10
pip install einops
pip install timm
pip install spconv-cu111
pip install h5py
pip install imagecorruptions
pip install distortion
pip install PyMieScatt 
pip install -v -e .
python projects/corenet/setup.py develop
