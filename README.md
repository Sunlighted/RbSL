## Offline Goal-Conditioned Reinforcement Learning for Safety-Critical Tasks with Recovery Policy

#### [[Paper]](see it soon on arxiv)

Chenyang Cao<sup>1</sup>, Zichen Yan<sup>1</sup>, Renhao Lu<sup>1</sup>, Junbo Tan<sup>1</sup>, Xueqian Wang<sup>1</sup>

<sup>1</sup>SIGS, Tsinghua University

This is a PyTorch implementation of our paper [Offline Goal-Conditioned Reinforcement Learning for Safety-Critical Tasks with Recovery Policy](); this code can be used to reproduce simulation experiments of the paper. 

Here is a video practicing RbSL on a real robot!
[video](coming soon)

## SetUp
### Requirements
- MuJoCo=2.0.0

### Setup Instructions
1. Create conda environment and activate it:
     ```
     conda env create -f environment.yaml
     conda activate gofar
     pip install --upgrade numpy
     pip install torch==1.10.0 torchvision==0.11.1 torchaudio===0.10.0 gym==0.17.3
2. (Optionally) install the [Panda-gym](https://github.com/qgallouedec/panda-gym) environment for the panda experiment.
3. Download the offline dataset [here](https://drive.google.com/file/d/1niq6bK262segc7qZh8m5RRaFNygEXoBR/view) and place ```/offline_data``` in the project root directory.

## Experiments
We provide commands for reproducing the main offline GCRL results in Table 1

The following command can reproduce the main results:
```
mpirun -np 1 python train.py --env $ENV --method $METHOD --expert_percent $EXPERT --random_percent $RANDOM
```
| Flags and Parameters  | Description |
| ------------- | ------------- |
| ``--env $ENV``  | constrained offline GCRL tasks: ```FetchReachObstacle, FetchPushObstacle, FetchPickObstacle, FetchSlideObstacle, PandaPush```|
| ``--method $METHOD``  | offline GCRL algorithms: ```rbsl, gofar, gcsl, wgcsl, AMlag```|
| ``--expert_percent $EXPERT --random_percent $RANDOM``  | percent: ```0 1, 0.1 0.9, 0.2 0.8, 0.5 0.5, 1 0```|

## Acknowledgement:
We referred to some code from the following repositories:
- [AWGCSL](https://github.com/YangRui2015/AWGCSL)
- [GoFAR](https://github.com/JasonMa2016/GoFAR)
