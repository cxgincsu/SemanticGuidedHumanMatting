import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.segmentation.deeplabv3 import ASPP

from .decoder import Decoder
from .mobilenet import MobileNetV2Encoder
from .refiner import SpectralNorm, MattingRefiner, BasicBlock
from .resnet import ResNetEncoder
from .utils import load_matched_state_dict
from .attention import ECALayer, GCTLayer

class Base(nn.Module):
    """
    A generic implementation of the base encoder-decoder network inspired by DeepLab.
    Accepts arbitrary channels for input and output.
    """
    
    def __init__(self, backbone: str, in_channels: int, out_channels: int):
        super().__init__()
        assert backbone in ["resnet50", "resnet101", "mobilenetv2"]
        if backbone in ['resnet50', 'resnet101']:
            self.backbone = ResNetEncoder(in_channels, variant=backbone)
            self.aspp = ASPP(2048, [3, 6, 9])
            self.decoder = Decoder([256, 128, 64, 48, out_channels], [512, 256, 64, in_channels])
        else:
            self.backbone = MobileNetV2Encoder(in_channels)
            self.aspp = ASPP(320, [3, 6, 9])
            self.decoder = Decoder([256, 128, 64, 48, out_channels], [32, 24, 16, in_channels])

    def forward(self, x):
        x, *shortcuts = self.backbone(x)
        x = self.aspp(x)
        x = self.decoder(x, *shortcuts)
        return x
    
    def load_pretrained_deeplabv3_state_dict(self, state_dict, print_stats=True):
        # Pretrained DeepLabV3 models are provided by <https://github.com/VainF/DeepLabV3Plus-Pytorch>.
        # This method converts and loads their pretrained state_dict to match with our model structure.
        # This method is not needed if you are not planning to train from deeplab weights.
        # Use load_state_dict() for normal weight loading.
        
        # Convert state_dict naming for aspp module
        state_dict = {k.replace('classifier.classifier.0', 'aspp'): v for k, v in state_dict.items()}

        if isinstance(self.backbone, ResNetEncoder):
            # ResNet backbone does not need change.
            load_matched_state_dict(self, state_dict, print_stats)
        else:
            # Change MobileNetV2 backbone to state_dict format, then change back after loading.
            backbone_features = self.backbone.features
            self.backbone.low_level_features = backbone_features[:4]
            self.backbone.high_level_features = backbone_features[4:]
            del self.backbone.features
            load_matched_state_dict(self, state_dict, print_stats)
            self.backbone.features = backbone_features
            del self.backbone.low_level_features
            del self.backbone.high_level_features


class HumanSegment(Base):
    """
    HumanSegment Consists of Shared Encoder and Segmentation Decoder
    Note : 
    --- Only resnet50 backbone is employed
    --- Only output err/hid are discarded  
    """
    
    def __init__(self, backbone: str = "resnet50"):
        super().__init__(backbone, in_channels=3, out_channels=(1 + 1 + 32))
        
    def forward(self, img):
        x, *shortcuts = self.backbone(img)
        x = self.aspp(x)
        x = self.decoder(x, *shortcuts)
        pha = torch.sigmoid(x[:, 0:1])
        err = torch.clamp(x[:, 1:2], 0, 1)
        hid = torch.relu(x[:, 2:])
        return pha, err, hid


class HumanMatting(HumanSegment):
    """
    HumanMatting Consists of Shared Encoder, Segmentation Decoder, Matting Decoder
    """
    def __init__(self,
                 backbone: str,
                 freeze_weights: bool = True):
        super().__init__(backbone)
        # segment weights are freezed during matting training
        if freeze_weights:
            for p in self.parameters():
                p.requires_grad = False

        # connect segment encoder feature with refine decoder
        self.shortcut_inplane = [3+1, 3+1, 64+1, 256+1, 512+1]
        self.shortcut_plane = [32, 32, 32, 64, 128]
        self.shortcut = nn.ModuleList()
        for stage, inplane in enumerate(self.shortcut_inplane):
            self.shortcut.append(self._make_shortcut(inplane, self.shortcut_plane[stage]))

        # define refine matting decoder
        self.refine = MattingRefiner(BasicBlock, [2, 3, 3, 2, 2, 2])
        del self.refine.layer1

    def _make_shortcut(self, inplane, planes):
        """
        Construct Attentive Shortcut Module
        """
        return nn.Sequential(
            SpectralNorm(nn.Conv2d(inplane, planes, kernel_size=3, padding=1, bias=False)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(planes),
            SpectralNorm(nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(planes),
            ECALayer(planes),
        )

    def forward(self, image, mask=None):
        # Downsample image for backbone
        image_quarter = F.interpolate(image, scale_factor=1.0/4, mode='bilinear',
                                   align_corners=False, recompute_scale_factor=True)

        # Shared encoder
        x4, x3, x2, x1, x0 = self.backbone(image_quarter)  # 1/64, 1/32, 1/16, 1/8, 1/4
        x4 = self.aspp(x4)  # 1/64
        # Generate mask
        if mask is None:
            # mask from segmentation decoder
            pred_seg = self.decoder(x4, x3, x2, x1, x0) 
            pred_seg = torch.sigmoid(pred_seg[:, 0:1])
            pred_seg = F.interpolate(pred_seg, scale_factor=4.0, mode='bilinear',
                                     align_corners=False, recompute_scale_factor=True)
            mask = pred_seg.gt(0.5).type(pred_seg.dtype)
            mask_quarter = F.interpolate(mask, scale_factor=1.0 / 4, mode='bilinear',
                                      align_corners=False, recompute_scale_factor=True)
        else:
            # mask from groundtruth augmentation
            pred_seg = mask
            mask_quarter = F.interpolate(mask, scale_factor=1.0 / 4, mode='bilinear',
                                      align_corners=False, recompute_scale_factor=True)

        # Sharing features
        m = mask
        x = torch.cat((image, m), dim=1)
        mid_fea = self.shortcut[0](x)      # 1/1

        m0 = mask_quarter
        x0 = torch.cat((x0, m0), dim=1)
        mid_fea0 = self.shortcut[1](x0)    # 1/4

        m1 = F.interpolate(m0, scale_factor=1.0/2, mode='bilinear', align_corners=False)
        x1 = torch.cat((x1, m1), dim=1)
        mid_fea1 = self.shortcut[2](x1)    # 1/8

        m2 = F.interpolate(m0, scale_factor=1.0/4, mode='bilinear', align_corners=False)
        x2 = torch.cat((x2, m2), dim=1)
        mid_fea2 = self.shortcut[3](x2)    # 1/16

        m3 = F.interpolate(m0, scale_factor=1.0/8, mode='bilinear', align_corners=False)
        x3 = torch.cat((x3, m3), dim=1)
        mid_fea3 = self.shortcut[4](x3)    # 1/32

        # Matting decoder  
        pred_alpha = self.refine(x4, mid_fea3, mid_fea2, mid_fea1, mid_fea0, mid_fea)
        pred_alpha["segment"] = pred_seg 
        return pred_alpha
