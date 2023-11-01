import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """
        
class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)
    
def normalization(n_channels,n_groups=1):
    """
    Make a standard normalization layer.

    :param n_channels: number of input channels.
    :param n_groups: number of groups. if this is 1, then it is identical to layernorm.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(num_groups=n_groups,num_channels=n_channels)

class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param n_channels: number of channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(
        self, 
        n_channels, 
        up_rate        = 2, # upsample rate
        up_mode        = 'nearest', # upsample mode ('nearest' or 'bilinear')
        use_conv       = False, # (optional) use output conv
        dims           = 2, # (optional) spatial dimension
        n_out_channels = None, # (optional) in case output channels are different from the input
        padding_mode   = 'zeros', 
        padding        = 1
    ):
        super().__init__()
        self.n_channels     = n_channels
        self.up_rate        = up_rate
        self.up_mode        = up_mode
        self.use_conv       = use_conv
        self.dims           = dims
        self.n_out_channels = n_out_channels or n_channels
        self.padding_mode   = padding_mode;
        self.padding        = padding
        
        if use_conv:
            self.conv = conv_nd(
                dims         = dims,
                in_channels  = self.n_channels,
                out_channels = self.n_out_channels,
                kernel_size  = 3, 
                padding      = padding,
                padding_mode = padding_mode)

    def forward(self, x):
        """ 
        :param x: [B x C x W x H]
        :return: [B x C x 2W x 2H]
        """
        assert x.shape[1] == self.n_channels
        if self.dims == 3: # 3D convolution
            x = F.interpolate(
                input = x,
                size  = (x.shape[2],x.shape[3]*2,x.shape[4]*2),
                mode  = self.up_mode
            )
        else:
            x = F.interpolate(
                input        = x,
                scale_factor = self.up_rate,
                mode         = self.up_mode
            ) # 'nearest' or 'bilinear'
            
        # (optional) final convolution
        if self.use_conv: 
            x = self.conv(x)
        return x
    
class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(
        self, 
        n_channels, 
        down_rate      = 2, # down rate
        use_conv       = False, # (optional) use output conv
        dims           = 2, # (optional) spatial dimension
        n_out_channels = None, # (optional) in case output channels are different from the input
        padding_mode   = 'zeros', 
        padding        = 1
    ):
        super().__init__()
        self.n_channels     = n_channels
        self.down_rate      = down_rate
        self.n_out_channels = n_out_channels or n_channels
        self.use_conv       = use_conv
        self.dims           = dims
        stride = self.down_rate if dims != 3 else (1, self.down_rate, self.down_rate)
        if use_conv:
            self.op = conv_nd(
                dims         = dims, 
                in_channels  = self.n_channels, 
                out_channels = self.n_out_channels,
                kernel_size  = 3, 
                stride       = stride,
                padding      = padding,
                padding_mode = padding_mode
            )
        else:
            assert self.n_channels == self.n_out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.n_channels
        return self.op(x)    

class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        (B:#batches, C:channel size, T:#tokens, H:#heads)

        :param qkv: an [B x (3*C) x T] tensor of Qs, Ks, and Vs.
        :return: an [B x C x T] tensor after attention.
        """
        n_batches, width, n_tokens = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(n_batches * self.n_heads, ch * 3, n_tokens).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v) # [(H*B) x (C//H) x T]
        out = a.reshape(n_batches, -1, n_tokens) # [B x C x T]
        return out

class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Input: [B x C x W x H] tensor
    Output: [B x C x W x H] tensor
    """
    def __init__(
            self,
            name               = 'attentionblock',
            n_channels         = 1,
            n_heads            = 1,
            n_groups           = 32,
    ):
        super().__init__()
        self.name               = name
        self.n_channels         = n_channels
        self.n_heads            = n_heads
        assert (
            n_channels % n_heads == 0
        ), f"n_channels:[%d] should be divisible by n_heads:[%d]."%(n_channels,n_heads)
            
        # Normalize 
        self.norm = normalization(n_channels=n_channels,n_groups=n_groups)
        
        # Tripple the channel
        self.qkv = nn.Conv1d(
            in_channels  = self.n_channels,
            out_channels = self.n_channels*3,
            kernel_size  = 1
        )
        
        # QKV Attention
        self.attention = QKVAttentionLegacy(
            n_heads = self.n_heads
        )
        
        # Projection
        self.proj_out = zero_module(
            nn.Conv1d(
                in_channels  = self.n_channels,
                out_channels = self.n_channels,
                kernel_size  = 1
            )
        )
        
    def forward(self, x):
        """
        :param x: [B x C x W x H] tensor
        :return out: [B x C x W x H] tensor
        """
        intermediate_output_dict = {}
        b, c, *spatial = x.shape
        # Triple the channel 
        x_rsh  = x.reshape(b, c, -1)    # [B x C x WH]
        x_nzd  = self.norm(x_rsh)       # [B x C x WH]
        qkv    = self.qkv(x_nzd)        # [B x 3C x WH]
        # QKV attention
        h_att  = self.attention(qkv)    # [B x C x WH]
        h_proj = self.proj_out(h_att)   # [B x C x WH]
        # Residual connection
        out = (x_rsh + h_proj).reshape(b, c, *spatial) # [B x C x W x H]
        # Append
        intermediate_output_dict['x']  = x
        intermediate_output_dict['x_rsh']  = x_rsh
        intermediate_output_dict['x_nzd']  = x_nzd
        intermediate_output_dict['qkv']    = qkv
        intermediate_output_dict['h_att']  = h_att
        intermediate_output_dict['h_proj'] = h_proj
        intermediate_output_dict['out']    = out
        return out,intermediate_output_dict

class ResBlock(TimestepBlock):
    """ 
    A residual block that can optionally change the number of channels and resolution
    
    :param n_channels: the number of input channels
    :param n_emb_channels: the number of timestep embedding channels
    :param n_out_channels: (if specified) the number of output channels
    :param n_groups: the number of groups in group normalization layer
    :param dims: spatial dimension
    :param p_dropout: the rate of dropout
    :param actv: activation
    :param use_conv: if True, and n_out_channels is specified, 
        use 3x3 conv instead of 1x1 conv
    :param use_scale_shift_norm: if True, use scale_shift_norm for handling emb
    :param upsample: if True, upsample
    :param downsample: if True, downsample
    :param sample_mode: upsample, downsample mode ('nearest' or 'bilinear')
    :param padding_mode: str
    :param padding: int
    """
    def __init__(
        self,
        name                 = 'resblock',
        n_channels           = 128,
        n_emb_channels       = 128,
        n_out_channels       = None,
        n_groups             = 16,
        dims                 = 2,
        p_dropout            = 0.5,
        kernel_size          = 3,
        actv                 = nn.SiLU(),
        use_conv             = False,
        use_scale_shift_norm = True,
        upsample             = False,
        downsample           = False,
        up_rate              = 2,
        down_rate            = 2,
        sample_mode          = 'nearest',
        padding_mode         = 'zeros',
        padding              = 1,
    ):
        super().__init__()
        self.name                 = name
        self.n_channels           = n_channels
        self.n_emb_channels       = n_emb_channels
        self.n_groups             = n_groups
        self.dims                 = dims
        self.n_out_channels       = n_out_channels or self.n_channels
        self.kernel_size          = kernel_size
        self.p_dropout            = p_dropout
        self.actv                 = actv
        self.use_conv             = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm
        self.upsample             = upsample
        self.downsample           = downsample
        self.up_rate              = up_rate
        self.down_rate            = down_rate
        self.sample_mode          = sample_mode
        self.padding_mode         = padding_mode
        self.padding              = padding
        
        # Input layers
        self.in_layers = nn.Sequential(
            normalization(n_channels=self.n_channels,n_groups=self.n_groups),
            self.actv,
            conv_nd(
                dims         = self.dims,
                in_channels  = self.n_channels,
                out_channels = self.n_out_channels,
                kernel_size  = self.kernel_size,
                padding      = self.padding,
                padding_mode = self.padding_mode
            )
        )
        
        # Upsample or downsample
        self.updown = self.upsample or self.downsample
        if self.upsample:
            self.h_upd = Upsample(
                n_channels = self.n_channels,
                up_rate    = self.up_rate,
                up_mode    = self.sample_mode,
                dims       = self.dims)
            self.x_upd = Upsample(
                n_channels = self.n_channels,
                up_rate    = self.up_rate,
                up_mode    = self.sample_mode,
                dims       = self.dims)
        elif self.downsample:
            self.h_upd = Downsample(
                n_channels = self.n_channels,
                down_rate  = self.down_rate,
                dims       = self.dims)
            self.x_upd = Downsample(
                n_channels = self.n_channels,
                down_rate  = self.down_rate,
                dims       = self.dims)
        else:
            self.h_upd = nn.Identity()
            self.x_upd = nn.Identity()
            
        # Embedding layers
        self.emb_layers = nn.Sequential(
            self.actv,
            nn.Linear(
                in_features  = self.n_emb_channels,
                out_features = 2*self.n_out_channels if self.use_scale_shift_norm 
                    else self.n_out_channels,
            ),
        )
        
        # Output layers
        self.out_layers = nn.Sequential(
            normalization(n_channels=self.n_out_channels,n_groups=self.n_groups),
            self.actv,
            nn.Dropout(p=self.p_dropout),
            zero_module(
                conv_nd(
                    dims         = self.dims, 
                    in_channels  = self.n_out_channels,
                    out_channels = self.n_out_channels,
                    kernel_size  = self.kernel_size,
                    padding      = self.padding,
                    padding_mode = self.padding_mode
                )
            ),
        )
        if self.n_channels == self.n_out_channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims         = self.dims,
                in_channels  = self.n_channels,
                out_channels = self.n_out_channels,
                kernel_size  = self.kernel_size,
                padding      = self.padding,
                padding_mode = self.padding_mode
            )
        else:
            self.skip_connection = conv_nd(
                dims         = self.dims,
                in_channels  = self.n_channels,
                out_channels = self.n_out_channels,
                kernel_size  = 1
            )
        
    def forward(self,x,emb):
        """
        :param x: [B x C x ...]
        :param x: [B x n_emb_channels]
        :return: [B x C x ...]
        """
        # Input layer (groupnorm -> actv -> conv)
        if self.updown: # upsample or downsample
            in_norm_actv = self.in_layers[:-1]
            in_conv = self.in_layers[-1]
            h = in_norm_actv(x) 
            h = self.h_upd(h)
            h = in_conv(h)
            x = self.x_upd(x)
        else:
            h = self.in_layers(x) # [B x C x ...]
            
        # Embedding layer
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None] # match 'emb_out' with 'h': [B x C x ...]
            
        # Combine input with embedding
        if self.use_scale_shift_norm:
            out_norm = self.out_layers[0]
            out_actv_dr_conv = self.out_layers[1:]
            # emb_out: [B x 2C x ...]
            scale,shift = th.chunk(emb_out, 2, dim=1) # [B x C x ...]
            h = out_norm(h) * (1.0 + scale) + shift # [B x C x ...]
            h = out_actv_dr_conv(h) # [B x C x ...]
        else:
            # emb_out: [B x C x ...]
            h = h + emb_out
            h = self.out_layers(h)
            
        # Skip connection
        out = h + self.skip_connection(x) # [B x C x ...]
        return out # [B x C x ...]
    
