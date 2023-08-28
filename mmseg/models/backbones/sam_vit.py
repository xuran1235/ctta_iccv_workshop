import torch
import torch.nn as nn

from torch.nn import Linear,AvgPool1d,CrossEntropyLoss

from ..builder import BACKBONES

@BACKBONES.register_module()
class SAMVit(nn.Module):
#    def __init__(self, imageEncoder, config, num_classes) -> None:
#        super().__init__()
#        self.imageEncoder = imageEncoder
#        self.avgpool = AvgPool1d(config.img_size//16*config.img_size//16)
#        self.head = Linear(256, num_classes)
#        self.num_classes = num_classes
#    
#    def forward(self, x, labels=None):
#        x = self.imageEncoder(x) # B C 14 14
#        feature = x.permute(0,2,3,1)
#        x = x.flatten(2) # B C 14*14
#        #x = x.permute(0,2,1) # B C L
#        
#        x = self.avgpool(x) # B C 1
#        feature = x
#        x = x.squeeze(2)# B C
#        logits = self.head(x) # B num_classes
#        if labels is not None:
#            loss_fct = CrossEntropyLoss()
#            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
#            #return loss, feature
#            return loss
#        else:
#            #return logits, feature
#            return logits,feature
    def __init__(self, imageEncoder, config, num_classes) -> None:
        super().__init__()
        self.imageEncoder = imageEncoder
    
    def forward(self, x, labels=None, return_fea=False):
        x = self.imageEncoder(x) # B 768
        return x
