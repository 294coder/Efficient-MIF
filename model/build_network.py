from model.base_model import MODELS
import importlib



    # if name == 'pannet':
    #     return net(**kwargs)
    # elif name == 'fusionnet':
    #     return net(8, 32)
    # elif name == 'panformer':
    #     # return PanFormerUNet(8, multi_channels=(16, 32, 64))
    #     # return PanFormerSwitch(8, multi_channels=(27, 54, 108))
    #     return PanFormerUNet2(8, multi_channels=(16, 32, 64), hidden_c=64)
    #     # return PanFormerGAU(8, hidden_c=64, multi_channels=(16, 32, 64), attn_drop=(0., 0.2, 0.4, 0.4))
    # elif name == 'm3dnet':
    #     return M3DNet(8, 1, 32, 4)
    # elif name == 'dcfnet':
    #     return DCFNet(8, 'C')
    # else:
    #     raise NotImplementedError
