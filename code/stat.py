from models import modules
from models import net
from models.backbone_dict import backbone_dict
from torchstat import stat

model = net.CoarseNet(modules.E_resnet,modules.D_resnet,[64, 128, 256, 512],'R_CLSTM_5')
stat(model,(16,3,5,224,224))