import math
import numpy as np
import matplotlib.pyplot as plt
import torch as th
import torch.nn as nn
from module import (
    conv_nd,
    ResBlock,
    AttentionBlock,
    TimestepEmbedSequential,
)

def get_named_beta_schedule(
    schedule_name,
    num_diffusion_timesteps, 
    scale_betas=1.0,
    np_type=np.float64
    ):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = scale_betas * 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np_type
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")
    
def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def get_ddpm_constants(
    schedule_name = 'cosine',
    T             = 1000,
    np_type       = np.float64
    ):
    timesteps = np.linspace(start=1,stop=T,num=T)
    betas = get_named_beta_schedule(
        schedule_name           = schedule_name,
        num_diffusion_timesteps = T,
        scale_betas             = 1.0,
    ).astype(np_type) # [1,000]
    alphas                    = 1.0 - betas 
    alphas_bar                = np.cumprod(alphas, axis=0) #  cummulative product
    alphas_bar_prev           = np.append(1.0,alphas_bar[:-1])
    sqrt_recip_alphas         = np.sqrt(1.0/alphas)
    sqrt_alphas_bar           = np.sqrt(alphas_bar)
    sqrt_one_minus_alphas_bar = np.sqrt(1.0-alphas_bar)
    posterior_variance        = betas*(1.0-alphas_bar_prev)/(1.0-alphas_bar)
    # Append
    dc = {}
    dc['schedule_name']             = schedule_name
    dc['T']                         = T
    dc['timesteps']                 = timesteps
    dc['betas']                     = betas
    dc['alphas']                    = alphas
    dc['alphas_bar']                = alphas_bar
    dc['alphas_bar_prev']           = alphas_bar_prev
    dc['sqrt_recip_alphas']         = sqrt_recip_alphas
    dc['sqrt_alphas_bar']           = sqrt_alphas_bar
    dc['sqrt_one_minus_alphas_bar'] = sqrt_one_minus_alphas_bar
    dc['posterior_variance']        = posterior_variance
    return dc

def plot_ddpm_constants(dc):
    """ 
    Plot DDPM constants
    """
    plt.figure(figsize=(10,3))
    cs = [plt.cm.gist_rainbow(x) for x in np.linspace(0,1,8)]
    lw = 2
    plt.subplot(1,2,1)
    plt.plot(dc['timesteps'],dc['betas'],
             color=cs[0],label=r'$\beta_t$',lw=lw)
    plt.plot(dc['timesteps'],dc['alphas'],
             color=cs[1],label=r'$\alpha_t$',lw=lw)
    plt.plot(dc['timesteps'],dc['alphas_bar'],
             color=cs[2],label=r'$\bar{\alpha}_t$',lw=lw)
    plt.plot(dc['timesteps'],dc['sqrt_alphas_bar'],
             color=cs[5],label=r'$\sqrt{\bar{\alpha}_t}$',lw=lw)
    
    plt.plot(dc['timesteps'],dc['sqrt_one_minus_alphas_bar'],
             color=cs[6],label=r'$\sqrt{1-\bar{\alpha}_t}$',lw=lw)
    
    
    plt.plot(dc['timesteps'],dc['posterior_variance'],'--',
            color='k',label=r'$ Var[x_{t-1}|x_t,x_0] $',lw=lw)
    
    plt.xlabel('Diffusion steps',fontsize=8)
    plt.legend(fontsize=10,loc='center left',bbox_to_anchor=(1,0.5))
    plt.grid(lw=0.5)
    plt.title('DDPM Constants',fontsize=10)
    plt.subplot(1,2,2)
    plt.plot(dc['timesteps'],dc['betas'],color=cs[0],label=r'$\beta_t$',lw=lw)
    plt.plot(dc['timesteps'],dc['posterior_variance'],'--',
            color='k',label=r'$ Var[x_{t-1}|x_t,x_0] $',lw=lw)
    plt.xlabel('Diffusion steps',fontsize=8)
    plt.legend(fontsize=10,loc='center left',bbox_to_anchor=(1,0.5))
    plt.grid(lw=0.5)
    plt.title('DDPM Constants',fontsize=10)
    plt.subplots_adjust(wspace=0.7)
    plt.show()
    
def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

def forward_sample(x0_batch,t_batch,dc):
    """
    Forward diffusion sampling
    :param x0_batch: [B x C x ...]
    :param t_batch: [B]
    :return: xt_batch of [B x C x ...] and noise of [B x C x ...]
    """
    # Gather diffusion constants with matching dimension
    out_shape = (t_batch.shape[0],) + ((1,)*(len(x0_batch.shape)-1))
    device = t_batch.device
    sqrt_alphas_bar_t = th.gather(
        input = th.from_numpy(dc['sqrt_alphas_bar']).to(device), # [T]
        dim   = -1,
        index = t_batch-1
    ).reshape(out_shape) # [B x 1 x 1 x 1] if (rank==4) and [B x 1 x 1] if (rank==3)
    sqrt_one_minus_alphas_bar = th.gather(
        input = th.from_numpy(dc['sqrt_one_minus_alphas_bar']).to(device), # [T]
        dim   = -1,
        index = t_batch-1
    ).reshape(out_shape) # [B x 1 x 1 x 1]
    # Forward sample
    noise = th.randn_like(input=x0_batch) # [B x C x ...]
    xt_batch = sqrt_alphas_bar_t*x0_batch + \
        sqrt_one_minus_alphas_bar*noise # [B x C x ...]
    return xt_batch,noise

class DiffusionUNet(nn.Module):
    """ 
    U-Net for diffusion models
    """
    def __init__(
        self,
        name             = 'unet',
        dims             = 1, # spatial dimension, if dims==1, [B x C x L], if dims==2, [B x C x W x H]
        n_in_channels    = 128, # input channels
        n_model_channels = 64, # base channel size
        n_emb_dim        = 128, # time embedding size
        n_enc_blocks     = 2, # number of encoder blocks
        n_dec_blocks     = 2, # number of decoder blocks
        n_groups         = 16, # group norm paramter
        n_heads          = 4, # number of heads
        device           = 'cpu',
    ):
        super().__init__()
        self.name             = name
        self.dims             = dims
        self.n_in_channels    = n_in_channels
        self.n_model_channels = n_model_channels
        self.n_emb_dim        = n_emb_dim
        self.n_enc_blocks     = n_enc_blocks
        self.n_dec_blocks     = n_dec_blocks
        self.n_groups         = n_groups
        self.n_heads          = n_heads
        self.device           = device
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(in_features=self.n_emb_dim,out_features=self.n_emb_dim),
            nn.SiLU(),
            nn.Linear(in_features=self.n_emb_dim,out_features=self.n_emb_dim),
        ).to(self.device)
        
        # Lifting and projection
        self.lift = conv_nd(
            dims         = self.dims,
            in_channels  = self.n_in_channels,
            out_channels = self.n_model_channels,
            kernel_size  = 1,
        )
        self.proj = conv_nd(
            dims         = self.dims,
            in_channels  = self.n_model_channels,
            out_channels = self.n_in_channels,
            kernel_size  = 1,
        )
        
        # Declare U-net 
        # Encoder
        self.enc_layers = []
        for e_idx in range(self.n_enc_blocks):
            # Residual block
            self.enc_layers.append(
                ResBlock(
                    n_channels     = self.n_model_channels,
                    n_emb_channels = self.n_emb_dim,
                    n_out_channels = self.n_model_channels,
                    n_groups       = self.n_groups,
                    dims           = self.dims,
                    upsample       = False,
                    downsample     = False,
                )   
            )
            # Attention block
            self.enc_layers.append(
                AttentionBlock(
                    n_channels     = self.n_model_channels,
                    n_heads        = self.n_heads,
                    n_groups       = self.n_groups,
                )
            )
            
        # Decoder
        self.dec_layers = []
        for d_idx in range(self.n_dec_blocks):
            # Residual block
            if d_idx == 0: 
                self.dec_layers.append(
                    ResBlock(
                        n_channels     = self.n_model_channels*2,
                        n_emb_channels = self.n_emb_dim,
                        n_out_channels = self.n_model_channels,
                        n_groups       = self.n_groups,
                        dims           = self.dims,
                        upsample       = False,
                        downsample     = False,
                    )   
                )
            else:
                self.dec_layers.append(
                    ResBlock(
                        n_channels     = self.n_model_channels,
                        n_emb_channels = self.n_emb_dim,
                        n_out_channels = self.n_model_channels,
                        n_groups       = self.n_groups,
                        dims           = self.dims,
                        upsample       = False,
                        downsample     = False,
                    )   
                )
            # Attention block
            self.dec_layers.append(
                AttentionBlock(
                    n_channels     = self.n_model_channels,
                    n_heads        = self.n_heads,
                    n_groups       = self.n_groups,
                )
            )
            
        # Define U-net
        self.enc_net = nn.Sequential()
        for l_idx,layer in enumerate(self.enc_layers):
            self.enc_net.add_module(
                name   = 'enc_%02d'%(l_idx),
                module = TimestepEmbedSequential(layer)
            )
        self.dec_net = nn.Sequential()
        for l_idx,layer in enumerate(self.dec_layers):
            self.dec_net.add_module(
                name   = 'dec_%02d'%(l_idx),
                module = TimestepEmbedSequential(layer)
            )
        
    def forward(self,x,timesteps):
        """ 
        :param x: [B x n_in_channels x ...]
        :timesteps: [B]
        :return: [B x n_in_channels x ...], same shape as x
        """
        intermediate_output_dict = {}
        intermediate_output_dict['x'] = x 
        
        # time embedding
        emb = self.time_embed(
            timestep_embedding(timesteps,self.n_emb_dim)
        ) # [B x n_emb_dim]
        
        # Lift input
        h = self.lift(x) # [B x n_model_channels x ...]
        intermediate_output_dict['x_lifted'] = h
        
        # Encoder
        self.h_enc_list = []
        for m_idx,module in enumerate(self.enc_net):
            h = module(h,emb)
            if isinstance(h,tuple): h = h[0] # in case of having tuple
            # Append
            intermediate_output_dict['enc_%02d'%(m_idx)] = h
            # Append encoder output
            if (m_idx%2) == 1:
                self.h_enc_list.append(h)
            
        # Decoder
        for h_idx,h_enc in enumerate(self.h_enc_list):
            if h_idx == 0: h_enc_stack = h_enc
            else: h_enc_stack = th.cat([h_enc_stack,h_enc],dim=1)
        h = h_enc_stack
        for m_idx,module in enumerate(self.dec_net):
            h = module(h,emb)
            if isinstance(h,tuple): h = h[0] # in case of having tuple
            # Append
            intermediate_output_dict['dic_%02d'%(m_idx)] = h
                
        # Projection
        out = self.proj(h)
        intermediate_output_dict['out'] = out
        
        return out,intermediate_output_dict
