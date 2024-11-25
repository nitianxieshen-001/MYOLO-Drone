# MYOLO-Drone

## Model Weight Files
We have provided the weight files for our model that has been trained.

## Modules
In the `modules`, we have our proposed:
- **MDSCA module** This includes the SGA,CGA and GFFN modules.
- **MFA module** This includes the SS2D, Gate Block, LLF, and Tiny-LLF modules.

## Model Architecture
The architecture of our model network is given in the YAML file.

## Training Results
The `result` folder contains some of the results from our model training.

# 代码复现说明

## 1. 环境需求

- **YOLOv8基础环境**：需要CUDA版本为11.8。
- **Vmamba基础环境**：确保已安装vmamba。

## 2. 复现步骤

- 根据所提供的`yaml`文件，将`backbone`和`head`对应的模块做出相应替换。
- 将提供的各个模块导入`block`中或者创建新文件进行定义。
- 在`task.py`文件中配置所定义的新模块。
- 可以使用提供的`best.pt`预训练模型进行训练。

## 3. 数据集说明

本实验所运用的数据集均为公共数据集，`Visdrone2019`和`PASCAL VOC`数据集均可直接获得。

## 4. 结果说明

- 在文件中提供了模型的训练数据图像，PR曲线以及混淆矩阵。
- 还给出了`best.pt`模型在训练集和验证集上的结果图。

请根据上述步骤进行代码复现，并确保环境配置正确。如果有任何问题，可以参考提供的文档和资源。
