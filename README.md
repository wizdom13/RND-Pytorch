# RND-Pytoch
#### Random Network Distillation(RND) algo in pytorch

This is a PyTorch implementation of [Exploration by Random Network Distillation](https://arxiv.org/abs/1810.12894) paper.

 
## Requirements

- [python3.6](http://www.python.org/)
- [PyTorch](http://pytorch.org/)
- [gym](http://gym.openai.com/)
- [numpy](http://www.numpy.org/)
- [PIL](https://github.com/whatupdave/pil)
- [argparse](https://github.com/python/cpython/blob/3.7/Lib/argparse.py)
- [tensorboardX](https://github.com/lanpa/tensorboardX)



## Usage:

## Prepare
```
CUDA_VISIBLE_DEVICES=0
```

## Training

```
python train.py
```

## Enjoy

```
python enjoy.py
```

## TensorboardX Graph

```
tensorboard --logdir runs
```
Open in browser [http://localhost:6006](http://localhost:6006)


