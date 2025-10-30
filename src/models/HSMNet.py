from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .modules import *



class HSMNet(nn.Module):

    def __init__(self, norm_layer=None, zero_init_residual=True, p=0.5):
        super(HSMNet, self).__init__()

        self.zero_init_residual = zero_init_residual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.srm = SRMConv2d(1, 0)
        self.bn1_intact=norm_layer(30)
        self.relu_intact = nn.ReLU(inplace=True)
        

        self.A1_intact = BlockA(30, 30, norm_layer=norm_layer)
        self.A2_intact = BlockA(30, 30, norm_layer=norm_layer)
        self.A3_intact = BlockA(30, 30, norm_layer=norm_layer)

        
        self.B1_intact = BlockB(30, 64, norm_layer=norm_layer)
        self.AB1_intact = BlockA(64, 64, norm_layer=norm_layer)
        self.B2_intact = BlockB(64, 128, norm_layer=norm_layer)
        self.AB2_intact = BlockA(128, 128, norm_layer=norm_layer)
    
        self.pool = nn.AvgPool2d(3, stride=2, padding=1)
        self.avgpool_intact = nn.AdaptiveAvgPool2d((1, 1))


        self.Dilated_conv1_intact = Dilate_BlockA(30, 30,norm_layer=norm_layer)
        
        self.Dilated_B1_intact = BlockB(30, 64, norm_layer=norm_layer)
        self.Dilated_AB1_intact = BlockA(64, 64, norm_layer=norm_layer)
        self.Dilated_B2_intact = BlockB(64, 128, norm_layer=norm_layer)
        self.Dilated_AB2_intact = BlockA(128, 128, norm_layer=norm_layer)
    
        self.avgpool_intact_Dilated = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, 2)
   
        self.dropout = nn.Dropout(p=p)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):

                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)

        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, (BlockA, BlockB)):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, intact):

        intact = intact.float()
        out_intact = self.srm(intact)
        out_intact = self.bn1_intact(out_intact)
        out_intact = self.relu_intact(out_intact)
        out_dilated = out_intact
        out_intact = self.A1_intact(out_intact)
        out_intact = self.A2_intact(out_intact)
        out_intact = self.A3_intact(out_intact)

       
        out_intact = self.B1_intact(out_intact)
        out_intact = self.AB1_intact(out_intact)
      
        out_intact = self.B2_intact(out_intact)
        out_intact = self.AB2_intact(out_intact)
 
        out_intact = self.avgpool_intact(out_intact)
        out_intact = out_intact.view(out_intact.size(0), out_intact.size(1))
        
        out_dilated = self.Dilated_conv1_intact(out_dilated)
        
        
        out_dilated = self.Dilated_B1_intact(out_dilated)
        out_dilated = self.Dilated_AB1_intact(out_dilated)
        
        out_dilated = self.Dilated_B2_intact(out_dilated)
        out_dilated = self.Dilated_AB2_intact(out_dilated)
        out_dilated = self.avgpool_intact_Dilated(out_dilated)
        out_dilated = out_dilated.view(out_dilated.size(0), out_dilated.size(1))
       
        outs = torch.cat([out_intact,out_dilated],dim=-1)

        outs = self.dropout(outs)
        outs = self.fc1(outs)

        return outs
