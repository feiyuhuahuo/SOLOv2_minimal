## SOLOv2_minimal
Minimal PyTorch implementation of [SOLOv2](https://arxiv.org/abs/2003.10152).  No mmdet, no Detectron, PyTorch only. 


## Test Environments  
PyTorch 1.13  
Python 3.10  
CUDA 11.6


## Performance 
This project is trained on one RTX3090. Batch size is 16 for light-Resnet50 and light-Resnet34, 10 for Resnet50.  
[Download weights here](https://github.com/feiyuhuahuo/SOLOv2_minimal/releases/tag/v1.0)  
mask mAP:  

| configuration  | official | this project |
|:--------------:|:--------:|:------------:|
|    Resnet50    |   37.5   |     38.1     |
| light-Resnet50 |   33.7   |     33.9     |
| light-Resnet34 |   32.0   |     32.1     |
 

## Train
DDP is not supported for now.  
Modify `data_root` and the related images and label path in `configs.py`, choose a suitable batch size for `TrainBatchSize`.  
Then:
```Shell
python train.py
```
You can use `break_weight` to continue your training.  

## Evalution
This project use a self-write api to evaluate mAP ([for more detail](https://github.com/feiyuhuahuo/COCO_improved)). 
The default setting is `SelfEval(dataset.coco, coco_dt, all_points=True, iou_type='segmentation')` in `val.py`, to get the exact value in the above table, please set `all_points=False`.  
Modify `val_weight` in `configs.py`.  
Then:  
```Shell
python val.py
```


## Detect
Modify `detect_images` in `configs.py`. Set `detect_mode='contour'` to show object contours, set `detect_mode='overlap'` to show object masks.  
Then:
```Shell
python detect.py
```


## Export to ONNX    
Modify `val_weight` in `configs.py`.  
Then:  
```Shell
python export2onnx.py
```
This project use `torch.jit.trace()` to export ONNX model. But there are some if-else branches in postprocess, this is not compatible with trace mode.  
When no object is detected, the model will encounter an error, please use try-except to skip it. For more details, consult `detetc_onnx.py`.  


## Train custom datasets
![example.bmp](docs%2Fexample.bmp)  
Please use [HuaHuoLabel](https://github.com/feiyuhuahuo/HuaHuoLabel) to label your data. There are two label modes in `HuaHuoLabel`, for now the `Separate File` mode is supported in this project.  

The directory structure:  
![2023-03-23_09-52.png](docs%2F2023-03-23_09-52.png)  
The label file format:  
![2023-03-23_10-05.png](docs%2F2023-03-23_10-05.png)  
The "qcolor" and "widget_points" are not necessary for training. Check [example.json](docs%2Fexample.json) for detail.  

Write a custom configuration in `configs.py`. Choose this configuration in `train.py`.  
Then:  

```Shell
python train.py
```



- Some parameters need to be taken care of by yourself:
- todo

