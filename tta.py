import torch 
import numpy as np
from edafa import ClassPredictor

class TTA(ClassPredictor):
    def __init__(self,model,device,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.model = model
        self.device = device

    def predict_patches(self,patches):
        patches = torch.from_numpy(patches)
        patches = patches.permute(0, 3, 1, 2).to(self.device)

        res = self.model(patches)
        res = res.cpu().detach().numpy()
        return res
    


