# DRL-Flappybird

This repository contains the PyTorch implementation of [DeepLearningFlappyBird](https://github.com/yenchenlin/DeepLearningFlappyBird).

## Requirements
```
pip install -r requirements.txt
```

## Training
```
python train.py
```

**Note**: before training from scratch, you should backup the pretrained weights `model/dqnnet.pt`.

## Testing
```
python test.py --weights model/dqnnet.pt
```

## Acknowledgements

* [yenchenlin/DeepLearningFlappyBird](https://github.com/yenchenlin/DeepLearningFlappyBird)
