# Singular Value Representation 

This code implements the theoretical framework developped by D. Meller and N. Berkouk (https://arxiv.org/abs/2302.08183) which allows to investigate the inner structure of neural networks. 
All the essential code and documentation can be found in the file svr.py. One example notebook is also provided.  
Some of the expriments we presented in the original paper can be reproduced easily by executing the 'experiments' notebook. 

## Set up 

In a terminal run : 
```bash
git clone https://github.com/danmlr/svr.git
cd svr 
pip install -r requirements.txt
```
Then install torchvision, (ideally with conda) : 
```bash
conda install torchvision -c pytorch
```
Instructions for other setups can be found here : https://pypi.org/project/torchvision/

## Minimal working example 

```python
import torch
from svr import SVR

model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16',pretrained=True)
weights = [ w.detach() for w in model.parameters() ]
svr_vgg16 = SVR(weights) 

svr_vgg16.plot()
``` 
![vgg16 svr](https://github.com/danmlr/svr/blob/main/vgg16.png)

## SVR of fully connected network during training on MNIST

<p align="center">
  <img src="https://github.com/danmlr/svr/blob/main/mnist_learning.gif" />
</p>

