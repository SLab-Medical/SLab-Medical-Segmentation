import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiDilationDepthwiseConv3D(nn.Module):
    def __init__(self, in_channels, conv, kernel_sizes=[1,3,5], strides=[1,1,1], dw_parallel=True):
        super(MultiDilationDepthwiseConv3D, self).__init__()
        self.in_channels = in_channels
        self.dw_parallel = dw_parallel
        self.dilations = [(kernel - 1) // 2 if kernel > 1 else 1 for kernel in kernel_sizes]
        modified_kernel_sizes = [3 if kernel > 1 else 1 for kernel in kernel_sizes]
        #print(kernel_sizes, strides)

        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                conv(self.in_channels, self.in_channels, modified_kernel_sizes[i], strides[i], kernel_sizes[i] // 2, dilation=self.dilations[i], groups=self.in_channels, bias=False),
            )
            for i in range(len(modified_kernel_sizes))
        ])

    def forward(self, x):
        # Apply the convolution layers in a loop
        outputs = []
        for dwconv in self.dwconvs:
            dw_out = dwconv(x)
            outputs.append(dw_out)
            if self.dw_parallel == False:
                x = x+dw_out
        # concatenate the features
        out = torch.cat(outputs, dim=1)
        return out

class EfficientMedNeXtBlock(nn.Module):

    def __init__(self, 
                in_channels:int, 
                out_channels:int, 
                exp_r:int=4, 
                kernel_sizes=(1,3,5), 
                strides=(1,1,1),
                do_res:int=True,
                norm_type:str = 'group',
                dim = '3d',
                conv=None,
                grn = False
                ):

        super().__init__()

        self.do_res = do_res
        self.in_channels = in_channels
        self.out_channels = out_channels
        exp_r = len(kernel_sizes)
        assert dim in ['2d', '3d']
        self.dim = dim
        if conv == None:
            if self.dim == '2d':
                conv = nn.Conv2d
            elif self.dim == '3d':
                conv = nn.Conv3d
            
        # First convolution layer with DepthWise Convolutions
        self.conv1 = MultiDilationDepthwiseConv3D(in_channels, conv, kernel_sizes=kernel_sizes, strides=strides, dw_parallel=True)

        # Normalization Layer. GroupNorm is used by default.
        if norm_type=='group':
            self.norm = nn.GroupNorm(
                num_groups=in_channels, 
                num_channels=exp_r*in_channels
                )
        elif norm_type=='layer':
            self.norm = LayerNorm(
                normalized_shape=in_channels, 
                data_format='channels_first'
                )
        
        # GeLU activations
        self.act = nn.GELU()
        
        # Third convolution (Compression) layer with Conv3D 1x1x1
        self.conv3 = conv(
            in_channels = exp_r*in_channels,
            out_channels = out_channels,
            kernel_size = 1,
            stride = 1,
            padding = 0
        )
        
        if self.do_res and (self.in_channels != self.out_channels):
            self.res_conv = conv(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = 1,
                stride = 1,
                padding = 0
            )

        self.grn = grn
        if grn:
            if dim == '3d':
                self.grn_beta = nn.Parameter(torch.zeros(1,exp_r*in_channels,1,1,1), requires_grad=True)
                self.grn_gamma = nn.Parameter(torch.zeros(1,exp_r*in_channels,1,1,1), requires_grad=True)
            elif dim == '2d':
                self.grn_beta = nn.Parameter(torch.zeros(1,exp_r*in_channels,1,1), requires_grad=True)
                self.grn_gamma = nn.Parameter(torch.zeros(1,exp_r*in_channels,1,1), requires_grad=True)

    def forward(self, x, dummy_tensor=None):
        
        x1 = x
        x1 = self.conv1(x1)
        x1 = self.act(self.norm(x1))
        if self.grn:
            # gamma, beta: learnable affine transform parameters
            # X: input of shape (N,C,H,W,D)
            if self.dim == '3d':
                gx = torch.norm(x1, p=2, dim=(-3, -2, -1), keepdim=True)
            elif self.dim == '2d':
                gx = torch.norm(x1, p=2, dim=(-2, -1), keepdim=True)
            nx = gx / (gx.mean(dim=1, keepdim=True)+1e-6)
            x1 = self.grn_gamma * (x1 * nx) + self.grn_beta + x1
        x1 = self.conv3(x1)
        if self.do_res:
            if self.in_channels != self.out_channels:
                x = self.res_conv(x)
            x1 = x + x1  
        return x1

class EfficientMedNeXtDownBlock(EfficientMedNeXtBlock):

    def __init__(self, in_channels, out_channels, exp_r:int=4, kernel_sizes=[1,3,5], strides=[2,1,1],
                do_res=False, norm_type = 'group', dim='3d', grn=False):

        if dim == '2d':
            conv = nn.Conv2d
        elif dim == '3d':
            conv = nn.Conv3d
        
        super().__init__(in_channels, out_channels, exp_r, kernel_sizes, strides=strides, 
                        do_res = False, norm_type = norm_type, dim=dim,
                        grn=grn)
        self.resample_do_res = do_res
        if do_res:
            self.res_conv = conv(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = 1,
                stride = 2
            )

    def forward(self, x, dummy_tensor=None):
        
        x1 = super().forward(x)
        
        if self.resample_do_res:
            res = self.res_conv(x)
            x1 = x1 + res

        return x1


class EfficientMedNeXtUpBlock(EfficientMedNeXtBlock):

    def __init__(self, in_channels, out_channels, exp_r:int=4, kernel_sizes=[1,3,5], strides=[2,1,1],
                do_res=False, norm_type = 'group', dim='3d', grn = False):

        self.resample_do_res = do_res
        
        self.dim = dim
        if dim == '2d':
            conv = nn.ConvTranspose2d
        elif dim == '3d':
            conv = nn.ConvTranspose3d
        
        super().__init__(in_channels, out_channels, exp_r, kernel_sizes=kernel_sizes, strides=strides,
                    do_res=False, norm_type = norm_type, dim=dim, conv=conv,
                    grn=grn) 
        if do_res:            
            self.res_conv = conv(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = 1,
                stride = 2
                )

    def forward(self, x, dummy_tensor=None):
        
        x1 = super().forward(x)
        # Asymmetry but necessary to match shape
        if self.dim == '2d':
            x1 = torch.nn.functional.pad(x1, (1,0,1,0))
        elif self.dim == '3d':
            x1 = torch.nn.functional.pad(x1, (1,0,1,0,1,0))
        
        if self.resample_do_res:
            res = self.res_conv(x)
            if self.dim == '2d':
                res = torch.nn.functional.pad(res, (1,0,1,0))
            elif self.dim == '3d':
                res = torch.nn.functional.pad(res, (1,0,1,0,1,0))
            x1 = x1 + res

        return x1


class OutBlock(nn.Module):

    def __init__(self, in_channels, n_classes, dim, stride=1):
        super().__init__()
        
        if dim == '2d':
            conv = nn.ConvTranspose2d
        elif dim == '3d':
            conv = nn.ConvTranspose3d
        self.conv_out = conv(in_channels, n_classes, kernel_size=1, stride = stride)
    
    def forward(self, x, dummy_tensor=None): 
        return self.conv_out(x)


class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-5, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))        # beta
        self.bias = nn.Parameter(torch.zeros(normalized_shape))         # gamma
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x, dummy_tensor=False):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x
         
if __name__ == "__main__":

    network = EfficientMedNeXtBlock(in_channels=12, out_channels=12, do_res=True, grn=True, norm_type='group').cuda()
    # network = LayerNorm(normalized_shape=12, data_format='channels_last').cuda()
    # network.eval()
    with torch.no_grad():
        print(network)
        x = torch.zeros((2, 12, 64, 64, 64)).cuda()
        print(network(x).shape)
