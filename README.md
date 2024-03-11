## Offline Goal-Conditioned Reinforcement Learning for Safety-Critical Tasks with Recovery Policy

#### [[Paper(ICRA2024)]](https://arxiv.org/abs/2403.01734)

Chenyang Cao<sup>1</sup>, Zichen Yan<sup>1</sup>, Renhao Lu<sup>1</sup>, Junbo Tan<sup>1</sup>, Xueqian Wang<sup>1</sup>

<sup>1</sup>SIGS, Tsinghua University

This is a PyTorch implementation of our paper [Offline Goal-Conditioned Reinforcement Learning for Safety-Critical Tasks with Recovery Policy](https://arxiv.org/abs/2403.01734); this code can be used to reproduce simulation experiments of the paper. 

Here is a video practicing RbSL on a real robot!
[video](coming soon)

## Setup
### Requirements
- MuJoCo=2.1.0

### Setup Instructions
1. Create conda environment and activate it:
     ```
     conda env create -f environment.yaml
     conda activate rbsl
     pip install --upgrade numpy
     pip install torch==1.10.0 torchvision==0.11.1 torchaudio===0.10.0 gym==0.17.3
2. (Optionally) install the [Panda-gym](https://github.com/qgallouedec/panda-gym) environment for the panda experiment.
3. Download the offline dataset [here](https://cloud.tsinghua.edu.cn/d/22d96f1efa0942d0b551/) and place ```/offline_data``` in the project root directory.

## Experiments
We provide commands for reproducing the main RbSL results in Table 1

The following command can reproduce the main results:
```
mpirun -np 1 python train.py --env $ENV --method $METHOD --expert_percent $EXPERT --random_percent $RANDOM
```
| Flags and Parameters  | Description |
| ------------- | ------------- |
| ``--env $ENV``  | constrained offline GCRL tasks: ```FetchReachObstacle, FetchPushObstacle, FetchPickObstacle, FetchSlideObstacle, PandaPush```|
| ``--method $METHOD``  | offline GCRL algorithms: ```rbsl, gofar, gcsl, wgcsl, AMlag```|
| ``--expert_percent $EXPERT`` ``--random_percent $RANDOM``  | percent: ```0 1, 0.1 0.9, 0.2 0.8, 0.5 0.5, 1 0```|

## Acknowledgement:
We referred to some code from the following repositories:
- [AWGCSL](https://github.com/YangRui2015/AWGCSL)
- [GoFAR](https://github.com/JasonMa2016/GoFAR)

## Cite:
If you use this repo, please cite as follows:
```
@article{cao2024offline,
  title={Offline Goal-Conditioned Reinforcement Learning for Safety-Critical Tasks with Recovery Policy},
  author={Cao, Chenyang and Yan, Zichen and Lu, Renhao and Tan, Junbo and Wang, Xueqian},
  journal={arXiv preprint arXiv:2403.01734},
  year={2024}
}
```
