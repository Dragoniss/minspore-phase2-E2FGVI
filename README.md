

# 相机标定

## 1 案例概述

### 1.1 概要描述

本代码根据噪声模型来标定相机的噪声参数，用于对相机进行去噪。

### 1.2 代码目录结构与说明

本工程名称为calib，工程目录如下图所示

```
├── calib.py
├── calib.sh
├── raw2numpy.sh			       
├── vnne_data_process_multiprocess_subpath_visual_raw_without_yuv.py
├── IMX327.json
└── GC4663.json
 
```

## 

## 2 标定流程介绍

### 2.1  数据收集

对于每个相机，采集不同iso（本次标定iso=100, 200, 400, 800, 1600, 3200, 6400）下的数据如下：

(1) 采集低光数据（要求保证绝对无光环境) 每个iso 200-300帧

(2) 采集正常光照数据（保证光照条件一致且不变，保证色板位置居中，去除len shading）每个iso 200-300帧

### 2.2  raw数据转换为numpy数组

```bash
bash raw2numpy.sh <raw_dir>
```

其中<raw_dir>填写raw数据所在目录，运行后会在相同路径下生成/*_result文件夹，保存有numpy数据及相关可视化结果。

### 2.3  numpy数据处理及标定

经过2.2步骤后，我们将所有iso下的数据全部进行转换，并将文件目录组织成以下形式：

```
├── calib
│   ├── black				   # dark frame转换数据
│   │   ├── iso100_result
│   │   ├── iso200_result
│   │   ├── iso400_result
│   │   ├── ...
│   │   └── iso6400_result		
│   └── white				   # normal frame转换数据
│   │   ├── iso100_result
│   │   ├── iso200_result
│   │   ├── iso400_result
│   │   ├── ...
│   │   └── iso6400_result

```

然后运行如下代码：

```python
python calib.py --input_root <your_root>/calib/ --sensor <IMX327/GC4663> --black_level <black level> 
```

程序会弹出可视化raw图，标定图中各色块位置即可完成标定。标定结果会储存成json格式放于代码路径下。

## 

> 用于车型识别的GoogLeNet_cars模型，参考链接：https://gist.github.com/bogger/b90eb88e31cd745525ae
