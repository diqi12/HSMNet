#Thanks to SiaStegNet for providing the code. The implementation in this paper was developed with reference to SiaStegNet.

# HSMNet Pytorch
HSMNet: A Multi-Resolution Grayscale Image Steganalysis Method Based on Hybrid Dilated Convolution and Self-Attention Multi-Channel Network

Yang W, Liu Y, Chen Y, et al. HSMNet: A multi-resolution grayscale image steganalysis method based on hybrid dilated convolution and self-attention multi-channel network[J]. Information Sciences, 2025: 122824.


## Overview
This repository contains a implementation of HSMNet, along with pre-trained models and examples.

We hope that our HSMNet design can provide some inspiration for future research in multi-resolution image steganalysis.

## Table of contents
1. About HSMNet
2. Usage

### About HSMNet
***Abstract*** Convolutional neural networks (CNNs) have demonstrated remarkable capability in detecting hidden information within images. However, many CNN-based steganalysis methods are designed for fixed‑resolution images, leading to cover–source mismatch and accuracy degradation when confronted with complex, real‑world multi‑resolution images, and limiting their practical utility. To overcome this limitation, we propose a multi-resolution grayscale image steganalysis method that integrates hybrid dilated convolution with self-attention mechanisms in a multi-channel network architecture. In the multi-resolution grayscale image steganalysis method, we designed a steganalysis network including two branches. One uses vanilla convolution to extract local features, and the other leverages hybrid dilated convolution to extract global features. Furthermore, since steganographic algorithms mainly embed messages in rich texture regions, we incorporate a channel attention mechanism in our proposed method that dynamically weights feature maps to enhance focus on these critical areas. This strategic combination of local and global feature extraction, coupled with adaptive attention weighting, enables our model to more effectively discern subtle discrepancies between cover and stego images. Extensive experiments conducted on the BOSSbase and ALASKA \#2 benchmark datasets demonstrate that our proposed steganalysis method achieves superior detection performance compared to current state-of-the-art multi-resolution steganalysis approaches.

### Usage
#### Quickstart
```
python train.py --train-cover-dir ... --val-cover-dir ... --train-stego-dir ... --val-stego-dir ... --model kenet --ckpt-dir ...
```

#### Other Parameters
* --epoch
* --lr
* --wd
* --eps
* --batch-size default=32
* --num-workers
* --finetune Load a pretrained model
* --seed
* --log-interval
* --lr-strategy
To be continued
