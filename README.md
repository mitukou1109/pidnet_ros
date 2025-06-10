# pidnet_ros

## 使用方法

Humble @ Ubuntu 22.04 @ Jetson AGX Orin でテスト済み

### 本体のクローン

```bash
cd ~ # 例
git clone git@github.com:XuJiacong/PIDNet.git
```

### 仮想環境の作成

```bash
cd ~/PIDNet
virtualenv -p python3 .venv # 仮想環境マネージャは何でもいいがシステムインタプリタを使うこと
source .venv/bin/activate
```

### PyTorch のインストール

> AMD64 の場合は`pip install torch torchvision`で OK

Jetson の場合

```bash
pip install http://jetson.webredirect.org/jp6/cu126/+f/5cf/9ed17e35cb752/torch-2.5.0-cp310-cp310-linux_aarch64.whl#sha256=5cf9ed17e35cb7523812aeda9e7d6353c437048c5a6df1dc6617650333049092
pip install http://jetson.webredirect.org/jp6/cu126/+f/5f9/67f920de3953f/torchvision-0.20.0-cp310-cp310-linux_aarch64.whl#sha256=5f967f920de3953f2a39d95154b1feffd5ccc06b4589e51540dc070021a9adb9
```

### ROS 2 ラッパーをクローン・ビルド

```bash
cd ~/minitruck_ws/src # 例
git clone git@github.com:mitukou1109/pidnet_ros.git
cd pidnet_ros
pip install -r requirements.txt
deactivate
cd ..
colcon build # symlink-installするとimportエラーになる（原因不明）
```

### 実行

カメラ（[`usb_cam`](https://github.com/ros-drivers/usb_cam)、[`v4l2_camera`](https://gitlab.com/boldhearts/ros2_v4l2_camera)）や画像ファイル（[`image_publisher`](https://github.com/ros-perception/image_pipeline/tree/humble/image_publisher)）等の入力ソースを用意（`/image_raw`に publish）

```bash
source ~/minitruck_ws/install/local_setup.bash
source ~/PIDNet/.venv/bin/activate
PYTHONPATH=$HOME/PIDNet:$HOME:$PYTHONPATH ros2 run pidnet_ros object_segmenter_node --ros-args -p checkpoint_file:=/path/to/checkpoint/file -p use_compressed_image:=false
```
