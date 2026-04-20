# 目标检测模型综合对比实验

本项目用于课程实验中的目标检测模型对比，当前已经接入以下 6 种模型：

- `R-CNN`
- `Fast R-CNN`
- `Faster R-CNN`
- `YOLO`
- `SSD`
- `DETR`

目前项目提供两种使用方式：

- `toy` 小数据集：用于快速冒烟测试，先验证代码、环境和训练流程是否跑通
- `VOC` / `COCO`：用于后续正式实验对比

注意：

- 当前版本已经优先保证 6 个模型都可以在 `toy` 数据集上跑通
- `Faster R-CNN` 走的是 torchvision 检测接口
- 其它 5 个模型目前接的是简化训练分支，适合课程实验前期联调、结构对比和流程验证

## 项目结构

```text
objDet/
├─ datasets/                  # 数据集目录
├─ models/                    # 模型定义
│  ├─ backbone.py             # 通用 backbone 配置与默认策略
│  ├─ yolo_backbones.py       # YOLO 可选 backbone
│  ├─ rcnn.py                 # R-CNN
│  ├─ fast_rcnn.py            # Fast R-CNN
│  ├─ faster_rcnn.py          # Faster R-CNN
│  ├─ yolo_resnet.py          # YOLO baseline
│  ├─ ssd_resnet.py           # SSD baseline
│  └─ detr_resnet.py          # DETR baseline
├─ outputs/                   # 训练输出目录
├─ utils/
│  ├─ dataset.py              # 数据集加载
│  ├─ toy_dataset.py          # 内置小数据集
│  ├─ metrics.py              # 检测指标
│  └─ selective_search.py     # Selective Search
├─ download_dataset.py        # 下载数据集脚本
├─ plot_results.py            # 绘制结果
├─ requirements.txt           # 项目依赖
├─ train.py                   # 统一训练入口
└─ readme.md                  # 项目说明
```

## 环境准备

Conda 环境，如 `conda create -n torch290`。

### 1. 激活环境

```powershell
conda activate torch290
```

### 2. 安装依赖

```powershell
pip install -r requirements.txt
```

## 支持的数据集

### 1. toy 小数据集

项目内置了一个极小的合成目标检测数据集，不依赖外部标注文件，适合先检查：

- CUDA 是否可用
- 模型是否能前向和反向传播
- checkpoint 是否正常保存
- log 是否正常记录

### 2. VOC / COCO

训练入口也支持：

- `VOC`
- `COCO`

示例：

```powershell
python train.py --dataset VOC --model faster_rcnn
python train.py --dataset COCO --model faster_rcnn
```

## 支持的模型切换

`train.py` 里的 `--model` 参数支持以下 6 种：

```text
rcnn
fast_rcnn
faster_rcnn
yolo
ssd
detr
```

`train.py` 里的 `--backbone` 参数当前支持：

```text
auto
resnet18
resnet34
resnet50
mobilenetv3
cspdarknet
```

其中：

- `resnet18 / resnet34 / resnet50` 可以用于当前项目里的所有模型
- `mobilenetv3 / cspdarknet` 当前主要用于 `YOLO` 的 backbone 对比实验
- `--backbone auto` 会按“速度优先”自动选择默认 backbone

当前默认的速度优先 backbone 映射如下：

```text
rcnn        -> resnet18
fast_rcnn   -> resnet18
faster_rcnn -> resnet18
yolo        -> mobilenetv3
ssd         -> resnet18
detr        -> resnet18
```

如果你只想直接跑最快的配置，推荐直接使用 `--backbone auto`，例如：

```powershell
python train.py --dataset toy --model faster_rcnn --backbone auto
python train.py --dataset toy --model yolo --backbone auto
```

如果你还想做 backbone 对比实验，更推荐用 `YOLO` 来对比，例如：

```powershell
python train.py --dataset toy --model yolo --backbone resnet18
python train.py --dataset toy --model yolo --backbone resnet34
python train.py --dataset toy --model yolo --backbone resnet50
python train.py --dataset toy --model yolo --backbone mobilenetv3
python train.py --dataset toy --model yolo --backbone cspdarknet
```

## 运行示例

### 1. 用 toy 数据集快速跑通

```powershell
conda activate torch290
python train.py --dataset toy --model faster_rcnn --backbone auto --epochs 3 --batch_size 4 --output_dir outputs\faster_toy
```

### 2. 切换不同模型

```powershell
python train.py --dataset VOC --model rcnn --epochs 30 --batch_size 32 --output_dir outputs\rcnn_voc
python train.py --dataset VOC --model fast_rcnn --epochs 30 --batch_size 32 --output_dir outputs\fast_rcnn_voc
python train.py --dataset VOC --model faster_rcnn --epochs 30 --batch_size 32 --output_dir outputs\faster_rcnn_voc --amp
python train.py --dataset VOC --model yolo --epochs 30 --batch_size 128 --output_dir outputs\yolo_voc --amp
python train.py --dataset VOC --model ssd --epochs 30 --batch_size 32 --output_dir outputs\ssd_voc
python train.py --dataset VOC --model detr --epochs 30 --batch_size 32 --output_dir outputs\detr_voc

python train.py --dataset COCO --model rcnn --epochs 30 --batch_size 32 --output_dir outputs\rcnn_coco
python train.py --dataset COCO --model fast_rcnn --epochs 30 --batch_size 32 --output_dir outputs\fast_rcnn_coco
python train.py --dataset COCO --model faster_rcnn --epochs 30 --batch_size 32 --output_dir outputs\faster_rcnn_coco
python train.py --dataset COCO --model yolo --epochs 30 --batch_size 32 --output_dir outputs\yolo_coco
python train.py --dataset COCO --model ssd --epochs 30 --batch_size 32 --output_dir outputs\ssd_coco
python train.py --dataset COCO --model detr --epochs 30 --batch_size 32 --output_dir outputs\detr_coco
```

### 3. 同一模型切换不同 backbone

下面以 `YOLO` 为例：

```powershell
python train.py --dataset VOC --model yolo --backbone resnet18 --epochs 30 --batch_size 32 --output_dir outputs\yolo_resnet18_voc
python train.py --dataset VOC --model yolo --backbone resnet34 --epochs 30 --batch_size 32 --output_dir outputs\yolo_resnet34_voc
python train.py --dataset VOC --model yolo --backbone resnet50 --epochs 30 --batch_size 32 --output_dir outputs\yolo_resnet50_voc
python train.py --dataset VOC --model yolo --backbone mobilenetv3 --epochs 30 --batch_size 32 --output_dir outputs\yolo_mobilenetv3_voc
python train.py --dataset VOC --model yolo --backbone cspdarknet --epochs 30 --batch_size 32 --output_dir outputs\yolo_cspdarknet_voc
```

### 4. 开启 AMP 混合精度

```powershell
python train.py --dataset toy --model faster_rcnn --amp
```

### 5. 指定输入尺寸

```powershell
python train.py --dataset toy --model yolo --input_size 256
```

### 6. 指定 toy 数据集大小

```powershell
python train.py --dataset toy --model detr --toy_train_samples 32 --toy_val_samples 8
```

### 7. 使用 VOC 训练

```powershell
python train.py --dataset VOC --model faster_rcnn --epochs 20 --batch_size 4 --output_dir outputs\voc_faster
```

## 断点续训

训练过程中会自动保存 checkpoint，可以通过 `--resume` 继续训练。

### 从上一次训练继续

```powershell
python train.py --dataset toy --model faster_rcnn --resume outputs\faster_toy\last_checkpoint.pth
```

### 从最佳模型继续

```powershell
python train.py --dataset toy --model faster_rcnn --resume outputs\faster_toy\best_checkpoint.pth
```

## 输出文件说明

每次训练会在 `--output_dir` 下生成以下内容：

- `last_checkpoint.pth`
- `best_checkpoint.pth`
- `checkpoint_epoch_*.pth`
- `*_training_log.csv`
- `latest_run_summary.json`

此外还可以使用 `plot_results.py` 根据多个日志文件自动生成训练曲线和汇总表。

### 模型对比出图

```powershell
python plot_results.py --mode model_compare --logs `
outputs\rcnn_toy\rcnn_resnet18_toy_training_log.csv `
outputs\fast_rcnn_toy\fast_rcnn_resnet18_toy_training_log.csv `
outputs\faster_rcnn_toy\faster_rcnn_resnet18_toy_training_log.csv `
outputs\yolo_toy\yolo_mobilenetv3_toy_training_log.csv `
outputs\ssd_toy\ssd_resnet18_toy_training_log.csv `
outputs\detr_toy\detr_resnet18_toy_training_log.csv
```

### backbone 对比出图

```powershell
python plot_results.py --mode backbone_compare --logs `
outputs\yolo_resnet18_toy\yolo_resnet18_toy_training_log.csv `
outputs\yolo_resnet34_toy\yolo_resnet34_toy_training_log.csv `
outputs\yolo_resnet50_toy\yolo_resnet50_toy_training_log.csv `
outputs\yolo_mobilenetv3_toy\yolo_mobilenetv3_toy_training_log.csv `
outputs\yolo_cspdarknet_toy\yolo_cspdarknet_toy_training_log.csv
```

### 1. CSV 日志

CSV 会记录每个 epoch 的信息，便于后续画图和分析：

- `epoch`
- `train_loss`
- `val_loss`
- `lr`
- `epoch_time_sec`
- `iter_time_sec`
- `checkpoint`

### 2. JSON 汇总

`latest_run_summary.json` 会保存最近一次训练状态，例如：

- 当前 epoch
- 当前训练 loss
- 当前验证 loss
- 最优验证 loss
- 学习率
- 最新 checkpoint 路径

## 常用参数

`train.py` 当前常用参数如下：

```text
--model               模型名称，可选 rcnn / fast_rcnn / faster_rcnn / yolo / ssd / detr
--backbone            backbone，可选 auto / resnet18 / resnet34 / resnet50 / mobilenetv3 / cspdarknet
--dataset             数据集，可选 toy / VOC / COCO
--epochs              训练轮数
--batch_size          批大小
--lr                  学习率
--num_workers         DataLoader 线程数
--output_dir          输出目录
--resume              checkpoint 路径
--save_every          每隔多少个 epoch 存一次模型
--pretrained_backbone 是否启用预训练 backbone
--amp                 是否启用混合精度
--input_size          输入图像尺寸
--toy_train_samples   toy 训练集大小
--toy_val_samples     toy 验证集大小
```

## 建议的调试顺序

推荐按下面顺序进行，能少走很多弯路：

1. 先在 `toy` 数据集上测试 6 个模型都能跑通
2. 再切换到 `VOC`
3. 跑通 `Faster R-CNN` 正式训练流程
4. 再做不同模型和不同 backbone 的实验对比
5. 最后整理日志和实验结果

## 当前状态说明

当前项目已经完成的部分：

- 6 个模型统一接入 `train.py`
- 支持 `toy` 小数据集快速联调
- 支持保存 checkpoint
- 支持断点续训
- 支持每个 epoch 自动写日志

当前还适合继续完善的部分：

- 将 `VOC / COCO` 上的训练和评估流程进一步标准化
- 为 6 个模型补更严格的正式 loss 和评测逻辑
- 增加实验结果可视化脚本

## 一个最推荐的起步命令

如果你现在只是想先确认项目整体可跑，直接执行：

```powershell
conda activate torch290
python train.py --dataset toy --model faster_rcnn --backbone auto --epochs 3 --batch_size 2 --output_dir outputs\quick_start
```
