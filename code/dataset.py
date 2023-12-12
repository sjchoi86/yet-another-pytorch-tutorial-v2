import numpy as np
import matplotlib.pyplot as plt
import torch as th
from torchvision import datasets,transforms
from util import gp_sampler,periodic_step,get_torch_size_string

def mnist(root_path='./data/',batch_size=128):
    """ 
        MNIST
    """
    mnist_train = datasets.MNIST(root=root_path,train=True,transform=transforms.ToTensor(),download=True)
    mnist_test  = datasets.MNIST(root=root_path,train=False,transform=transforms.ToTensor(),download=True)
    train_iter  = th.utils.data.DataLoader(mnist_train,batch_size=batch_size,shuffle=True,num_workers=1)
    test_iter   = th.utils.data.DataLoader(mnist_test,batch_size=batch_size,shuffle=True,num_workers=1)
    # Data
    train_data,train_label = mnist_train.data,mnist_train.targets
    test_data,test_label = mnist_test.data,mnist_test.targets
    return train_iter,test_iter,train_data,train_label,test_data,test_label

def get_1d_training_data(
    traj_type = 'step', # {'step','gp'}
    n_traj    = 10,
    L         = 100,
    device    = 'cpu',
    seed      = 1,
    plot_data = True,
    figsize   = (6,2),
    ls        = '-',
    lc        = 'k',
    lw        = 1,
    verbose   = True,
    ):
    """ 
        1-D training data
    """
    if seed is not None:
        np.random.seed(seed=seed)
    times = np.linspace(start=0.0,stop=1.0,num=L).reshape((-1,1)) # [L x 1]
    if traj_type == 'gp':
        traj = th.from_numpy(
            gp_sampler(
                times    = times,
                hyp_gain = 2.0,
                hyp_len  = 0.2,
                meas_std = 1e-8,
                n_traj   = n_traj
            )
        ).to(th.float32).to(device) # [n_traj x L]
    elif traj_type == 'gp2':
        traj_np = np.zeros((n_traj,L))
        for i_idx in range(n_traj):
            traj_np[i_idx,:] = gp_sampler(
                times    = times,
                hyp_gain = 2.0,
                hyp_len  = np.random.uniform(1e-8,1.0),
                meas_std = 1e-8,
                n_traj   = 1
            ).reshape(-1)
        traj = th.from_numpy(
            traj_np
        ).to(th.float32).to(device) # [n_traj x L]
    elif traj_type == 'step':
        traj_np = np.zeros((n_traj,L))
        for i_idx in range(n_traj):
            period      = np.random.uniform(low=0.38,high=0.42)
            time_offset = np.random.uniform(low=-0.02,high=0.02)
            y_min       = np.random.uniform(low=-3.2,high=-2.8)
            y_max       = np.random.uniform(low=2.8,high=3.2)
            traj_np[i_idx,:] = periodic_step(
                times       = times,
                period      = period,
                time_offset = time_offset,
                y_min       = y_min,
                y_max       = y_max
            ).reshape(-1)
        traj = th.from_numpy(
            traj_np
        ).to(th.float32).to(device) # [n_traj x L]
    elif traj_type == 'step2':
        traj_np = np.zeros((n_traj,L))
        for i_idx in range(n_traj): # for each trajectory
            # First, sample value and duration
            rate = 5
            val = np.random.uniform(low=-3.0,high=3.0)
            dur_tick = int(L*np.random.exponential(scale=1/rate))
            dim_dur  = 0.1 # minimum duration in sec
            dur_tick = max(dur_tick,int(dim_dur*L))
            
            tick_fr = 0
            tick_to = tick_fr+dur_tick
            while True:
                # Append
                traj_np[i_idx,tick_fr:min(L,tick_to)] = val
                
                # Termination condition
                if tick_to >= L: break 
                
                # sample value and duration
                val = np.random.uniform(low=-3.0,high=3.0)
                dur_tick = int(L*np.random.exponential(scale=1/rate))
                dur_tick = max(dur_tick,int(dim_dur*L))
                
                # Update tick
                tick_fr = tick_to
                tick_to = tick_fr+dur_tick
        traj = th.from_numpy(
            traj_np
        ).to(th.float32).to(device) # [n_traj x L]
    elif traj_type == 'triangle':
        traj_np = np.zeros((n_traj,L))
        for i_idx in range(n_traj):
            period      = 0.2
            time_offset = np.random.uniform(low=-0.02,high=0.02)
            y_min       = np.random.uniform(low=-3.2,high=-2.8)
            y_max       = np.random.uniform(low=2.8,high=3.2)
            times_mod = np.mod(times+time_offset,period)/period
            y = (y_max - y_min) * times_mod + y_min
            traj_np[i_idx,:] = y.reshape(-1)
        traj = th.from_numpy(
            traj_np
        ).to(th.float32).to(device) # [n_traj x L]
    else:
        print ("Unknown traj_type:[%s]"%(traj_type))
    # Plot
    if plot_data:
        plt.figure(figsize=figsize)
        for i_idx in range(n_traj): 
            plt.plot(times,traj[i_idx,:].cpu().numpy(),ls=ls,color=lc,lw=lw)
        plt.xlim([0.0,1.0])
        plt.ylim([-4,+4])
        plt.xlabel('Time',fontsize=10)
        plt.title('Trajectory type:[%s]'%(traj_type),fontsize=10)
        plt.show()
    # Print
    x_0 = traj[:,None,:] # [N x C x L]
    if verbose:
        print ("x_0:[%s]"%(get_torch_size_string(x_0)))
    # Out
    return times,x_0

def get_mdn_data(
    n_train    = 1000,
    x_min      = 0.0,
    x_max      = 1.0,
    y_min      = -1.0,
    y_max      = 1.0,
    freq       = 1.0,
    noise_rate = 1.0,
    seed       = 0,
    FLIP_AUGMENT = True,
):
    np.random.seed(seed=seed)
    
    if FLIP_AUGMENT:
        n_half    = n_train // 2
        x_train_a = x_min + (x_max-x_min)*np.random.rand(n_half,1) # [n_half x 1]
        x_rate    = (x_train_a-x_min)/(x_max-x_min) # [n_half x 1]
        sin_temp  = y_min + (y_max-y_min)*np.sin(2*np.pi*freq*x_rate)
        cos_temp  = y_min + (y_max-y_min)*np.cos(2*np.pi*freq*x_rate)
        y_train_a = np.concatenate(
            (sin_temp+1*(y_max-y_min)*x_rate,
            cos_temp+1*(y_max-y_min)*x_rate),axis=1) # [n_half x 2]
        x_train_b = x_min + (x_max-x_min)*np.random.rand(n_half,1) # [n_half x 1]
        x_rate    = (x_train_b-x_min)/(x_max-x_min) # [n_half x 1]
        sin_temp  = y_min + (y_max-y_min)*np.sin(2*np.pi*freq*x_rate)
        cos_temp  = y_min + (y_max-y_min)*np.cos(2*np.pi*freq*x_rate)
        y_train_b = -np.concatenate(
            (sin_temp+1*(y_max-y_min)*x_rate,
            cos_temp+1*(y_max-y_min)*x_rate),axis=1) # [n_half x 2]
        # Concatenate
        x_train = np.concatenate((x_train_a,x_train_b),axis=0) # [n_train x 1]
        y_train = np.concatenate((y_train_a,y_train_b),axis=0) # [n_train x 2]
    else:
        x_train   = x_min + (x_max-x_min)*np.random.rand(n_train,1) # [n_train x 1]
        x_rate    = (x_train-x_min)/(x_max-x_min) # [n_train x 1]
        sin_temp  = y_min + (y_max-y_min)*np.sin(2*np.pi*freq*x_rate)
        cos_temp  = y_min + (y_max-y_min)*np.cos(2*np.pi*freq*x_rate)
        y_train   = np.concatenate(
            (sin_temp+1*(y_max-y_min)*x_rate,
            cos_temp+1*(y_max-y_min)*x_rate),axis=1) # [n_train x 2]
        
    # Add noise
    x_rate  = (x_train-x_min)/(x_max-x_min) # [n_train x 1]
    noise   = noise_rate * (y_max-y_min) * (2*np.random.rand(n_train,2)-1) * ((x_rate)**2) # [n_train x 2]
    y_train = y_train + noise # [n_train x 2]
    return x_train,y_train