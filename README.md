# Singular Value Representation 

This code implements the theoretical framework developped by D. Meller and N. Berkouk (citation) which allows to investigate the inner strurcture of neural networks. 

## Set up 

In a terminal run : 
```bash
git clone https://github.com/danmlr/svr.git
cd svr 
pip install -r requirements.txt
```


## Minimal working example 

```python
import torch
from svr import SVR

model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16',pretrained=True)
weights= [ w.detach() for w in model.parameters() ]
svr_vgg16 = SVR(weights) 

svr_vgg16.plot()
``` 

