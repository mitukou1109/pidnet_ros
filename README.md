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

Jetsonの場合は、PyTorchとTorchvisionを対応するバージョンに変更
https://pypi.jetson-ai-lab.io/jp6/cu126 からURLを取得し、以下のコマンドを実行

```bash
cd ~/minitruck_ws/src/clipseg_ros
uv add --no-sync <url to torch whl>
uv add --no-sync <url to torchvision whl>
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
