
## Note 

- main_mrcnn (mrcnn)
- main_alb (mrcnn with cp)
    - evaluator error: IndexError: list index out of range  
evaluator results are not saved

## dataset

<span style="color: red"> __To be published soon__ </span> ⏳

Burned Connector Anomaly Detection Dataset  

Available Tasks: object detection, instance segmentation   
Format: [COCO](https://cocodataset.org/#home) format

## data info 
(TBU)

## USAGE

(TBU) 
```bash
```

## Experimental results 

|Model type   | Desc  |
|---|---|
|mrcnn_{num}   | ...  |
|mrcnn_cp_{num}   | Mask RCNN w/ Copy-paste aug. that starts from mrcnn_800.pth  |
|mrcnn_cp_v4_400 | Mask RCNN w/ Copy-paste aug. that starts from mrcnn_cp_400.pth  |

|<b>Mask RCNN (0~200 epoch)</b> |
| :--: |
| ![](./assets/mrcnn_200.png)|
|<b>last epoch value</b> |
| ![](./assets/mrcnn_200_table.png)|


|<b>Mask RCNN (200~400 epoch)</b> |
| :--: |
| ![](./assets/mrcnn_400.png)|
|<b>last epoch value</b> |
| ![](./assets/mrcnn_400_table.png)|


|<b>Mask RCNN (200~400 epoch)</b> |
| :--: |
| ![](./assets/mrcnn_800.png)|
|<b>last epoch value</b> |
| ![](./assets/mrcnn_800_table.png)|


## sample preview

from v1 dataset
![](./images/a_dark_1.png)  
from v2 dataset
![](./images/b_v2_2.png)  
![](./images/d_v2_2.jpg)  

## model weights

saved path: /volume1/NFS/epsilon/model_weights/bcad (in MSIS LAB NAS)
---

## Acknowledge

in [MSIS Lab](https://www.cbnu.msislab.com/).

![](./assets/msis_logo.png)

## Contact 

person in charge: epsilon ahn<sup>1</sup>  
email: ypahn@chungbuk.ac.kr   
<sup>1</sup>: Department of Computer Science, Chungbuk National University, Cheongju 28644, South Korea  



