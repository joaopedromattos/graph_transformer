import torch
import torch.nn as nn
import torch.nn.functional as F


class VanillaTransformer(nn.Module):
    def __init__(self, num_features:int, nhead:int, num_encoder_layers:int, *args, **kwargs) -> None:
        super(VanillaTransformer, self).__init__(*args, **kwargs)
        # import code
        # code.interact(local=locals())
        self.transformer = nn.Transformer(d_model=num_features, nhead=nhead, num_encoder_layers=num_encoder_layers)
        
        
    def forward(self, data, *args, **kwargs) -> None:
        pass
    
    def loss(self, *args, **kwargs) -> None:
        pass


