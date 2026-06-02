# pidnet_ros

PIDNetのROS 2ラッパー

## 動作要件

- CUDA 12.6
- [uv](https://docs.astral.sh/uv/getting-started/installation)
- ROS 2 Humble

## インストール

クローン・依存パッケージをインストール

```bash
cd ~ # 例
git clone git@github.com:XuJiacong/PIDNet.git
```

```bash
cd ~/minitruck_ws/src # 例
git clone git@github.com:mitukou1109/pidnet_ros.git
rosdep install -iyr --from-paths src
```

Jetsonの場合は、システムにcuSPARSELt、PyTorch、TorchVisionをインストール
PyTorchのwheelやTorchVisionのリポジトリのバージョンは環境に合わせる

```bash
cd ~/minitruck_ws/src/pidnet_ros
uv remove --no-sync torch
uv remove --no-sync torchvision

cd ~ # 例
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/sbsa/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cusparselt-cuda-12

pip install --no-cache https://developer.nvidia.com/w/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl

git clone https://github.com/pytorch/vision.git -b v0.20.0
cd vision
pip install . --no-build-isolation
```

ワークスペースをビルド

```bash
cd ~/minitruck_ws
colcon build --symlink-install
```

### 使用方法

```bash
source ~/minitruck_ws/install/local_setup.bash
PYTHONPATH=$PYTHONPATH:$HOME:$HOME/PIDNet \ # PIDNetをクローンした場所
ros2 run pidnet_ros segmentation_node --ros-args -p checkpoint_file:=/path/to/checkpoint.pt
```
