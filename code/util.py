import numpy as np
import matplotlib.pyplot as plt
import torch as th
from scipy.spatial import distance

def th2np(x):
    return x.detach().cpu().numpy()

def get_torch_size_string(x):
    return "x".join(map(str,x.shape))

def plot_4x4_torch_tensor(
    x_torch,
    figsize  = (4,4),
    cmap     = 'gray',
    info_str = '',
    top      = 0.92,
    hspace   = 0.1
):
    """
    :param x_torch: [B x C x W x H]
    """
    batch_size = x_torch.shape[0]
    fig = plt.figure(figsize=figsize)
    for i in range(batch_size):
        ax = plt.subplot(4,4,i+1)
        plt.imshow(x_torch.permute(0,2,3,1).detach().numpy()[i,:,:,:], cmap=cmap)
        plt.axis('off')
    plt.subplots_adjust(
        left=0.0,right=1.0,bottom=0.0,top=top,wspace=0.0,hspace=hspace)
    plt.suptitle('%s[%d] Images of [%dx%d] sizes'%
                 (info_str,batch_size,x_torch.shape[2],x_torch.shape[3]),fontsize=10)
    plt.show()
    
def plot_1xN_torch_img_tensor(
    x_torch,
    title_str_list = None,
    title_fontsize = 20):
    """ 
    : param x_torch: [B x C x W x H]
    """
    xt_np = x_torch.cpu().numpy() # [B x C x W x H]
    n_imgs = xt_np.shape[0]
    plt.figure(figsize=(n_imgs*2,3))
    for img_idx in range(n_imgs):
        plt.subplot(1,n_imgs,img_idx+1)
        if xt_np.shape[1]==1:
            plt.imshow(xt_np[img_idx,0,:,:], cmap='gray')
        else:
            plt.imshow(xt_np[img_idx,:,:,:].transpose(1,2,0))
        if title_str_list:
            plt.title(title_str_list[img_idx],fontsize=title_fontsize)
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    
def plot_1xN_torch_traj_tensor(
    times,
    x_torch,
    title_str_list = None,
    title_fontsize = 20,
    ylim           = None,
    figsize        = None,
    ):
    """ 
    : param x_torch: [B x C x ...]
    """
    xt_np = x_torch.cpu().numpy() # [B x C x W x H]
    n_trajs = xt_np.shape[0]
    L = times.shape[0]
    if figsize is None: figsize = (n_trajs*2,3)
    plt.figure(figsize=figsize)
    for traj_idx in range(n_trajs):
        plt.subplot(1,n_trajs,traj_idx+1)
        plt.plot(times,x_torch[traj_idx,0,:].cpu().numpy(),'-',color='k')
        if title_str_list:
            plt.title(title_str_list[traj_idx],fontsize=title_fontsize)
        if ylim:
            plt.ylim(ylim)
    plt.tight_layout()
    plt.show()    

def print_model_parameters(model):
    """ 
        Print model parameters
    """
    for p_idx,(param_name,param) in enumerate(model.named_parameters()):
        print ("[%2d] parameter:[%27s] shape:[%12s] numel:[%10d]"%
            (p_idx,
             param_name,
             get_torch_size_string(param),
             param.numel()
             )
            )
        
def print_model_layers(model,x_torch):
    """ 
        Print model layers
    """
    y_torch,intermediate_output_list = model(x_torch)
    batch_size = x_torch.shape[0]
    print ("batch_size:[%d]"%(batch_size))
    print ("[  ] layer:[%15s] size:[%14s]"
        %('input',"x".join(map(str,x_torch.shape)))
        )
    for idx,layer_name in enumerate(model.layer_names):
        intermediate_output = intermediate_output_list[idx]
        print ("[%2d] layer:[%15s] size:[%14s] numel:[%10d]"%
            (idx,
             layer_name,
             get_torch_size_string(intermediate_output),
             intermediate_output.numel()
            )) 
        
def model_train(model,optm,loss,train_iter,test_iter,n_epoch,print_every,device):
    """ 
        Train model
    """
    model.init_param(VERBOSE=False)
    model.train()
    for epoch in range(n_epoch):
        loss_val_sum = 0
        for batch_in,batch_out in train_iter:
            # Forward path
            if isinstance(model.x_dim,int):
                y_pred,_ = model(batch_in.view(-1,model.x_dim).to(device))
            else:
                y_pred,_ = model(batch_in.view((-1,)+model.x_dim).to(device))
            loss_out = loss(y_pred,batch_out.to(device))
            # Update
            optm.zero_grad()    # reset gradient 
            loss_out.backward() # back-propagate loss 
            optm.step()         # optimizer update
            loss_val_sum += loss_out
        loss_val_avg = loss_val_sum/len(train_iter)
        # Print
        if ((epoch%print_every)==0) or (epoch==(n_epoch-1)):
            train_accr = model_eval(model,train_iter,device)
            test_accr  = model_eval(model,test_iter,device)
            print ("epoch:[%2d/%d] loss:[%.3f] train_accr:[%.4f] test_accr:[%.4f]."%
                (epoch,n_epoch,loss_val_avg,train_accr,test_accr))
        
def model_eval(model,data_iter,device):
    """ 
        Evaluate model
    """
    with th.no_grad():
        n_total,n_correct = 0,0
        model.eval() # evaluate (affects DropOut and BN)
        for batch_in,batch_out in data_iter:
            y_trgt       = batch_out.to(device)
            if isinstance(model.x_dim,int):
                model_pred,_ = model(batch_in.view(-1,model.x_dim).to(device))
            else:
                model_pred,_ = model(batch_in.view((-1,)+model.x_dim).to(device))
            _,y_pred     = th.max(model_pred.data,1)
            n_correct    += (y_pred==y_trgt).sum().item()
            n_total      += batch_in.size(0)
        val_accr = (n_correct/n_total)
        model.train() # back to train mode 
    return val_accr
        
def model_test(model,test_data,test_label,device):
    """ 
        Test model
    """
    n_sample = 25
    sample_indices = np.random.choice(len(test_data),n_sample,replace=False)
    test_data_samples = test_data[sample_indices]
    test_label_samples = test_label[sample_indices]
    with th.no_grad():
        model.eval()
        if isinstance(model.x_dim,int):
            x_in = test_data_samples.view(-1,model.x_dim).type(th.float).to(device)/255.
        else:
            x_in = test_data_samples.view((-1,)+model.x_dim).type(th.float).to(device)/255.
        y_pred,_ = model(x_in)
    y_pred = y_pred.argmax(axis=1)
    # Plot
    plt.figure(figsize=(6,6))
    plt.subplots_adjust(top=1.0)
    for idx in range(n_sample):
        plt.subplot(5,5, idx+1)
        plt.imshow(test_data_samples[idx],cmap='gray')
        plt.axis('off')
        fontcolor = 'k' if (y_pred[idx] == test_label_samples[idx]) else 'r'
        plt.title("Pred:%d, Label:%d"%(y_pred[idx],test_label_samples[idx]),
                fontsize=8,color=fontcolor)
    plt.show()
    
def kernel_se(x1,x2,hyp={'gain':1.0,'len':1.0}):
    """ Squared-exponential kernel function """
    D = distance.cdist(x1/hyp['len'],x2/hyp['len'],'sqeuclidean')
    K = hyp['gain']*np.exp(-D)
    return K

def gp_sampler(
    times    = np.linspace(start=0.0,stop=1.0,num=100).reshape((-1,1)), # [L x 1]
    hyp_gain = 1.0,
    hyp_len  = 1.0,
    meas_std = 0e-8,
    n_traj   = 1
    ):
    """ 
        Gaussian process sampling
    """
    if len(times.shape) == 1: times = times.reshape((-1,1))
    L = times.shape[0]
    K = kernel_se(times,times,hyp={'gain':hyp_gain,'len':hyp_len}) # [L x L]
    K_chol = np.linalg.cholesky(K+1e-8*np.eye(L,L)) # [L x L]
    traj = K_chol @ np.random.randn(L,n_traj) # [L x n_traj]
    traj = traj + meas_std*np.random.randn(*traj.shape)
    return traj.T

def hbm_sampler(
    times    = np.linspace(start=0.0,stop=1.0,num=100).reshape((-1,1)), # [L x 1]
    hyp_gain = 1.0,
    hyp_len  = 1.0,
    meas_std = 0e-8,
    n_traj   = 1
    ):
    """
        Hilbert Brownian motion sampling
    """
    if len(times.shape) == 1: times = times.reshape((-1,1))
    L = times.shape[0]
    K = kernel_se(times,times,hyp={'gain':hyp_gain,'len':hyp_len}) # [L x L]
    K = K + 1e-8*np.eye(L,L)
    U,V = np.linalg.eigh(K,UPLO='L')
    traj = V @ np.diag(np.sqrt(U)) @ np.random.randn(L,n_traj) # [L x n_traj]
    traj = traj + meas_std*np.random.randn(*traj.shape)
    return traj.T

def get_colors(n):
    return [plt.cm.Set1(x) for x in np.linspace(0,1,n)]

def periodic_step(times,period,time_offset=0.0,y_min=0.0,y_max=1.0):
    times_mod = np.mod(times+time_offset,period)
    y = np.zeros_like(times)
    y[times_mod < (period/2)] = 1
    y*=(y_max-y_min)
    y+=y_min
    return y

def plot_ddpm_1d_result(
    times,x_data,step_list,x_t_list,
    plot_ancestral_sampling=True,
    plot_one_sample=False,
    lw_gt=1,lw_sample=1/2,
    ls_gt='-',ls_sample='-',
    lc_gt='b',lc_sample='k',
    ylim=(-4,+4),figsize=(6,3),title_str=None
    ):
    """
    :param times: [L x 1] ndarray
    :param x_0: [N x C x L] torch tensor, training data
    :param step_list: [M] ndarray, diffusion steps to append x_t
    :param x_t_list: list of [n_sample x C x L] torch tensors
    """
    
    x_data_np = x_data.detach().cpu().numpy() # [n_data x C x L]
    n_data = x_data_np.shape[0] # number of GT data
    C = x_data_np.shape[1] # number of GT data
    
    # Plot a seqeunce of ancestral sampling procedure
    if plot_ancestral_sampling:
        for c_idx in range(C):
            plt.figure(figsize=(15,2)); plt.rc('xtick',labelsize=6); plt.rc('ytick',labelsize=6)
            for i_idx,t in enumerate(step_list):
                plt.subplot(1,len(step_list),i_idx+1)
                x_t = x_t_list[t] # [n_sample x C x L]
                x_t_np = x_t.detach().cpu().numpy() # [n_sample x C x L]
                n_sample = x_t_np.shape[0]
                for i_idx in range(n_data): # GT
                    plt.plot(times.flatten(),x_data_np[i_idx,c_idx,:],ls='-',color=lc_gt,lw=lw_gt)
                for i_idx in range(n_sample): # sampled trajectories
                    plt.plot(times.flatten(),x_t_np[i_idx,c_idx,:],ls='-',color=lc_sample,lw=lw_sample)
                plt.xlim([0.0,1.0]); plt.ylim(ylim)
                plt.xlabel('Time',fontsize=8); plt.title('Step:[%d]'%(t),fontsize=8)
            plt.tight_layout(); plt.show()
    
    # Plot generated data
    for c_idx in range(C):
        plt.figure(figsize=figsize)
        x_0_np = x_t_list[0].detach().cpu().numpy() # [n_sample x C x L]
        for i_idx in range(n_data): # GT
            plt.plot(times.flatten(),x_data_np[i_idx,c_idx,:],ls=ls_gt,color=lc_gt,lw=lw_gt)
        n_sample = x_0_np.shape[0]
        if plot_one_sample:
            i_idx = np.random.randint(low=0,high=n_sample)
            plt.plot(times.flatten(),x_0_np[i_idx,c_idx,:],ls=ls_sample,color=lc_sample,lw=lw_sample)
        else:
            for i_idx in range(n_sample): # sampled trajectories
                plt.plot(times.flatten(),x_0_np[i_idx,c_idx,:],ls=ls_sample,color=lc_sample,lw=lw_sample)
        plt.xlim([0.0,1.0]); plt.ylim(ylim)
        plt.xlabel('Time',fontsize=8)
        if title_str is None:
            plt.title('[%d] Groundtruth and Generated trajectories'%(c_idx),fontsize=10)
        else:
            plt.title('[%d] %s'%(c_idx,title_str),fontsize=10)
        plt.tight_layout(); plt.show()


def plot_ddpm_2d_result(
    x_data,step_list,x_t_list,n_plot=1,
    tfs=10
    ):
    """
    :param x_data: [N x C x W x H] torch tensor, training data
    :param step_list: [M] ndarray, diffusion steps to append x_t
    :param x_t_list: list of [n_sample x C x L] torch tensors
    """
    for sample_idx in range(n_plot):
        plt.figure(figsize=(15,2))
        for i_idx,t in enumerate(step_list):
            x_t = x_t_list[t] # [n_sample x C x W x H]
            x_t_np = x_t.detach().cpu().numpy() # [n_sample x C x W x H]
            plt.subplot(1,len(step_list),i_idx+1)
            if x_data.shape[1]==1: # gray image
                plt.imshow(x_t_np[sample_idx,0,:,:], cmap='gray')
            else:
                plt.imshow(x_t_np[sample_idx,:,:,:].transpose(1,2,0))
            plt.axis('off')
            plt.title('Step:[%d]'%(t),fontsize=tfs)
        plt.tight_layout()
        plt.show()
    
    
def get_hbm_M(times,hyp_gain=1.0,hyp_len=0.1,device='cpu'):
    """ 
    Get a matrix M for Hilbert Brownian motion
    :param times: [L x 1] ndarray
    :return: [L x L] torch tensor
    """
    L = times.shape[0]
    K = kernel_se(times,times,hyp={'gain':hyp_gain,'len':hyp_len}) # [L x L]
    K = K + 1e-8*np.eye(L,L)
    U,V = np.linalg.eigh(K,UPLO='L')
    M = V @ np.diag(np.sqrt(U))
    M = th.from_numpy(M).to(th.float32).to(device) # [L x L]
    return M

def get_resampling_steps(t_T, j, r,plot_steps=False,figsize=(15,4)):
    """
    Get resampling steps for repaint, inpainting method using diffusion models
    :param t_T: maximum time steps for inpainting
    :param j: jump length
    :param r: the number of resampling
    """
    jumps = np.zeros(t_T+1)
    for i in range(1, t_T-j, j):
        jumps[i] = r-1
    t = t_T+1
    resampling_steps = []
    while t > 1:
        t -= 1
        resampling_steps.append(t)
        if jumps[t] > 0:
            jumps[t] -= 1
            for _ in range(j):
                t += 1
                resampling_steps.append(t)
    resampling_steps.append(0)
    
    # (optional) plot
    if plot_steps:
        plt.figure(figsize=figsize)
        plt.plot(resampling_steps,'-',color='k',lw=1)
        plt.xlabel('Number of Transitions')
        plt.ylabel('Diffusion time step')
        plt.show()
        
    # Return
    return resampling_steps