

## dataset

Burned Connector Anomaly Detection Dataset  

v1: Burned connectors through the process at the factory.  
v2: Burned connector samples manually with a lighter.
   
Available Tasks: object detection, instance segmentation   
Format: [COCO](https://cocodataset.org/#home) format

## USAGE

```bash
git clone https://github.com/epsilon-deltta/burnedCAD.git
cd burnedCAD
```

## files overall structure

```
├── annotations  
│   └── *.json  
├── images  
│   ├── *.png  
│   ├── *.jpg  
├── dataset.ipynb   
├── dataset.py  
```

## sample preview

from v1 dataset
![](./images/a_dark_1.png)  
from v2 dataset
![](./images/b_v2_2.png)  
![](./images/d_v2_2.jpg)  



## Acknowledge

in [MSIS Lab](https://www.cbnu.msislab.com/).

![](./assets/msis_logo.png)