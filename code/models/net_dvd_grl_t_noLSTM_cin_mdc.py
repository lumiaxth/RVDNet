import torch
import torch.nn as nn
from models.M_U_Net_MDC import M_U_Net_GRL, M_U_Net
from models import modules
from models.refinenet_dict import refinenet_dict
from models.grl import DomainAdaptationModule
from models import block as B
from models.backbone_dict import backbone_dict

def cubes_2_maps(cubes):
    b, c, d, h, w = cubes.shape
    cubes = cubes.permute(0, 2, 1, 3, 4) #b d c h w
#当对y使用了.contiguous()后，改变y的值时，x没有任何影响。
    return cubes.contiguous().view(b*d, c, h, w), b, d


def maps_2_cubes(maps, b, d):
    bd, c, h, w = maps.shape
    cubes = maps.contiguous().view(b, d, c, h, w)
# 当调用contiguous()时，会强制拷贝一份tensor，让它的布局和从头创建的一模一样，但是两个tensor完全没有联系。
    return cubes.permute(0, 2, 1, 3, 4)


def rev_maps(maps, b, d):
    """reverse maps temporarily."""
    cubes = maps_2_cubes(maps, b, d).flip(dims=[2])

    return cubes_2_maps(cubes)[0]


class CoarseNet(nn.Module):
    def __init__(self, encoder, decoder, block_channel, refinenet, bidirectional=False, input_residue=False):

        super(CoarseNet, self).__init__()

        self.use_bidirect = bidirectional
        self.input_residue = input_residue
# Res18
        self.E = encoder
        self.DA = DomainAdaptationModule()
        self.D = decoder
        self.MFF = modules.MFF(block_channel)
        self.Conv = nn.Conv3d(in_channels=80,
                              out_channels=3,
                              kernel_size=1,
                              padding=0,
                              )

        # self.R_fwd = refinenet_dict[refinenet](block_channel)
        # 增加时域grl
# R_CLSTM_5
        # if self.use_bidirect:
        #     self.R_bwd = refinenet_dict[refinenet](block_channel)
        #     self.bidirection_fusion = nn.Conv2d(6, 3, kernel_size=5, stride=1, padding=2, bias=True)

    def forward(self, x):
        x_cube = x

        x, b, d = cubes_2_maps(x) # cubes.contiguous().view(b*d, c, h, w), b, d
        x_block0, x_block1, x_block2, x_block3, x_block4 = self.E(x)

        spatial_domain_feature = self.DA(x_block2)

        x_decoder = self.D(x_block0, x_block1, x_block2, x_block3, x_block4)
        x_mff = self.MFF(x_block1, x_block2, x_block3, x_block4,[x_decoder.size(2),x_decoder.size(3)]) # torch.size()作用：查看Tensor的维度。h w
        # R_CLSTM_5
        # fwd_out = self.Conv(torch.cat((x_decoder, x_mff), 1))
        # fwd_out = self.R_fwd(torch.cat((x_decoder, x_mff), 1), b, d)
        fwd_in = maps_2_cubes(torch.cat((x_decoder, x_mff), 1), b, d)
        fwd_out = self.Conv(fwd_in)
        out = fwd_out

        # resolve odd number pixels
        if x_cube.shape[-1] % 2 == 1:
            out = out[:, :, :, :, :-1]

        if self.input_residue:
            return out + x_cube, spatial_domain_feature
        else:
            return out, spatial_domain_feature

class FastDVDNet(nn.Module):
    def __init__(self, in_frames=5, color=True, sigma_map=False, c_in=True):
        """
        class initial
        :param in_frames: T-2, T-1, T, T+1, T+2, generally 5 frames
        :param color: now only color images are supported
        :param sigma_map: noise map, whose value is the estimation of noise standard variation
        """
        super(FastDVDNet, self).__init__()
        self.in_frames = in_frames
        channel = 3 if color else 1
        in1 = (3 + (3 if c_in else 0) + (1 if sigma_map else 0)) * channel #18
        in2 = (3 + (1 if sigma_map else 0)) * channel #9
        self.block1 = M_U_Net_GRL(in1, channel)
        self.block2 = M_U_Net(in2, channel)

    def forward(self, c_out, c_in):
        """
        forward function
        :param input: [b, N, c, h, w], the concatenation of noisy frames and noise map
        :return: the noised frame corresponding to reference frame
        """
        # split the noisy frames and noise map
        # frames, map = torch.split(input, self.in_frames, dim=1)
        # frames = torch.split(input, self.in_frames, dim=1)
        input = torch.cat((c_out, c_in), 1)
        frames = input
        b, c, N, h, w = c_out.size() #torch.Size([32, 3, 5, 224, 224])
        #print(frames.size())
        data_temp_stage1 = []
        # temporal_domain_sum = []
        # first stage
        for i in range(self.in_frames-2):
            # print(frames[:, :, i:i+3, ...].contiguous().view(b, -1, h, w).size(),frames[:, :, i+1, ...].size())
            data_temp, temporal_domain = self.block1(frames[:, :, i:i+3, ...].contiguous().view(b, -1, h, w),
                c_out[:, :, i+1, ...]
            )
            data_temp_stage1.append(data_temp)
            # temporal_domain_sum.append(temporal_domain)
            if i == (self.in_frames-2)//2:
                temporal_domain_sum = temporal_domain            

        # second stage
        data_temp = torch.cat(data_temp_stage1, dim=1)
        # temporal_domain_sum = torch.cat(temporal_domain_sum, dim=1)
        #print(data_temp.size()) #torch.Size([32, 9, 224, 224])
        return self.block2(data_temp, c_out[:, :, N//2, ...]), temporal_domain_sum


if __name__ == '__main__':
    s = b, c, d, h, w = 1, 3, 5, 224, 224
    t = torch.ones(s).cuda()
    backbone = backbone_dict['resnet18'](use_bn=False, model_dir=None)
    encoder = modules.E_resnet(backbone, use_bn=False)
    decoder = modules.D_resnet(num_features=512)
    net = CoarseNet(encoder,decoder,block_channel=[64, 128, 256, 512],refinenet='R_CLSTM_5').cuda()
    output, spatial_domain_feature = net(t)
    F_net = FastDVDNet().cuda()
    out_f, temporal_domain_feature = F_net(output, t)
    print(out_f.size(),temporal_domain_feature.size())
    # import pdb; pdb.set_trace()


