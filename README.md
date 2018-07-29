# MASK R-CNN  sport actions fine tuning

Mask R-CNN is a powerful framework developed by facebook ([here more](https://arxiv.org/abs/1703.06870)), main features are:
- General and flexible for object instance segmentation 
- Part of Detectron, a state-of-the-art object detection algorithm collection
- Powered in python and natively Coffe2 
- Also available a [Keras + TensorFlow version](https://github.com/matterport/Mask_RCNN)

## Aims of this repo

Address the network towards Sport activities detection using fine tuning technique.
We want that the new will be able to detect only main subjects labelled as activity name (i.e. only people)

[logo]: https://github.com/adam-p/markdown-here/raw/master/src/common/images/aims.png "Logo Title Text 2"

## Dataset

Dataset used is ucf24 (subset of ucf101): a set for Action Recognition that consists of 3194 videos from 4 kind of action. Resolution of each video is 320x240 px.
We intending to work frame-by-frame, and also need an annotated groundtruth:
A frame annotated version of this dataset it's available from this [repo by Gurkit](https://github.com/gurkirt/realtime-action-detection)
#### but:
Not whole dataset is annotated, only “frame of interest”: this produces the 70% of useless data.
Only bboxes groundtruth is available and no masks are annotated: so we produced a mask gt by ourselves.

![Aims](https://github.com/barloccia/MASK-CNN-for-actions-recognition-/blob/master/images/aims.png)


## Getting Started
Here we propose a bief explenation of the files and their usages:
Coco weights used are available [here](https://arxiv.org/abs/1703.06870)

- 


## Results
mAP without considering masks:  84.5%
mAP considering masks IoU=25: 37.4%
mAP considering masks IoU=50:  28.7%

| Class        | No Mask           | IoU = 25  | IoU = 50  |
| ------------ |:-----------------:| :--------:| :--------:|
| WalkingWithDog	| 85.8% | 57.2% | 48.9% |
| BasketballDunk	| 62.1% | 1.7% | 0.2% |
| Biking	| 92.4% | 38.3% | 27.5% |
| CliffDiving	| 22.7% | 3.2% | 0.0% |
| CricketBowling	| 47.2% | 3.8% | 2.7% |
| Diving	| 83.0% | 2.3% | 1.4% |
| Fencing	| 97.9% | 19.5% | 14.0% |
| FloorGymnastics	| 64.8% | 34.0% | 28.5% |
| GolfSwing	| 81.0% | 71.8% | 67.6% |
| HorseRiding	| 95.3% | 27.7% | 16.2% |
| IceDancing	| 93.7% | 68.8% | 64.3% |
| LongJump	| 59.9% | 25.1% | 22.1% |
| PoleVault	| 54.7% | 2.6% | 1.6% |
| RopeClimbing	| 90.6% | 30.8% | 20.5% |
| SalsaSpin	| 86.4% | 48.5% | 22.7% |
| SkateBoarding	| 86.7% | 46.9% | 34.2% |
| Skiing	| 80.7% | 46.3% | 37.2% |
| Skijet	| 87.8% | 21.9% | 13.0% |
| SoccerJuggling	| 85.8% | 58.3% | 52.8% |
| Surfing	| 78.2% | 18.1% | 12.7% |
| TennisSwing	| 64.9% | 59.3% | 56.1% |
| TrampolineJumping	| 83.5% | 16.3% | 13.8% |
| VolleyballSpiking	| 39.5% | 0.7% | 0.3% |