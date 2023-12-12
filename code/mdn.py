import numpy as np
import torch as th
import torch.nn as nn
import matplotlib.pyplot as plt
from util import th2np

def get_argmax_mu(pi,mu):
    """
    :param pi: [N x K x D]
    :param mu: [N x K x D]
    """
    max_idx = th.argmax(pi,dim=1) # [N x D]
    argmax_mu = th.gather(input=mu,dim=1,index=max_idx.unsqueeze(dim=1)).squeeze(dim=1) # [N x D]
    return argmax_mu

def gmm_forward(pi,mu,sigma,data):
    """
    Compute Gaussian mixture model probability
    
    :param pi: GMM mixture weights [N x K x D]
    :param mu: GMM means [N x K x D]
    :param sigma: GMM stds [N x K x D]
    :param data: data [N x D]
    """
    data_usq = th.unsqueeze(data,1) # [N x 1 x D]
    data_exp = data_usq.expand_as(sigma) # [N x K x D]
    ONEOVERSQRT2PI = 1/np.sqrt(2*np.pi)
    probs = ONEOVERSQRT2PI * th.exp(-0.5 * ((data_exp-mu)/sigma)**2) / sigma # [N x K x D]
    probs = probs*pi # [N x K x D]
    probs = th.sum(probs,dim=1) # [N x D]
    log_probs = th.log(probs) # [N x D]
    log_probs = th.sum(log_probs,dim=1) # [N]
    nlls = -log_probs # [N]

    # Most probable mean [N x D]
    argmax_mu = get_argmax_mu(pi,mu) # [N x D]
    
    out = {
        'data_usq':data_usq,'data_exp':data_exp,
        'probs':probs,'log_probs':log_probs,'nlls':nlls,'argmax_mu':argmax_mu
    }
    return out

def gmm_uncertainties(pi, mu, sigma):
    """ 
    :param pi: [N x K x D]
    :param mu: [N x K x D]
    :param sigma: [N x K x D]
    """
    # Compute Epistemic Uncertainty
    mu_avg     = th.sum(th.mul(pi,mu),dim=1).unsqueeze(1) # [N x 1 x D]
    mu_exp     = mu_avg.expand_as(mu) # [N x K x D]
    mu_diff_sq = th.square(mu-mu_exp) # [N x K x D]
    epis_unct  = th.sum(th.mul(pi,mu_diff_sq), dim=1)  # [N x D]

    # Compute Aleatoric Uncertainty
    alea_unct = th.sum(th.mul(pi,sigma), dim=1)  # [N x D]

    # (Optional) sqrt operation helps scaling
    epis_unct,alea_unct = th.sqrt(epis_unct),th.sqrt(alea_unct)
    return epis_unct,alea_unct

class MixturesOfGaussianLayer(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        k,
        sig_max=None
    ):
        super(MixturesOfGaussianLayer,self).__init__()
        self.in_dim  = in_dim
        self.out_dim = out_dim
        self.k       = k
        self.sig_max = sig_max
        
        # Netowrks
        self.fc_pi    = nn.Linear(self.in_dim,self.k*self.out_dim)
        self.fc_mu    = nn.Linear(self.in_dim,self.k*self.out_dim)
        self.fc_sigma = nn.Linear(self.in_dim,self.k*self.out_dim)

    def forward(self,x):
        pi_logit = self.fc_pi(x) # [N x KD]
        pi_logit = th.reshape(pi_logit,(-1,self.k,self.out_dim)) # [N x K x D]
        pi       = th.softmax(pi_logit,dim=1) # [N x K x D]
        mu       = self.fc_mu(x) # [N x KD]
        mu       = th.reshape(mu,(-1,self.k,self.out_dim)) # [N x K x D]
        sigma    = self.fc_sigma(x) # [N x KD]
        sigma    = th.reshape(sigma,(-1,self.k,self.out_dim)) # [N x K x D]
        if self.sig_max is None:
            sigma = th.exp(sigma) # [N x K x D]
        else:
            sigma = self.sig_max * th.sigmoid(sigma) # [N x K x D]
        return pi,mu,sigma

class MixtureDensityNetwork(nn.Module):
    def __init__(
        self,
        name       = 'mdn',
        x_dim      = 1,
        y_dim      = 1,
        k          = 5,
        h_dim_list = [32,32],
        actv       = nn.ReLU(),
        sig_max    = 1.0,
        mu_min     = -3.0,
        mu_max     = +3.0,
        p_drop     = 0.1,
        use_bn     = False,
    ):
        super(MixtureDensityNetwork,self).__init__()
        self.name       = name
        self.x_dim      = x_dim
        self.y_dim      = y_dim
        self.k          = k
        self.h_dim_list = h_dim_list
        self.actv       = actv
        self.sig_max    = sig_max
        self.mu_min     = mu_min
        self.mu_max     = mu_max
        self.p_drop     = p_drop
        self.use_bn     = use_bn

        # Declare layers
        self.layer_list = []
        h_dim_prev = self.x_dim
        for h_dim in self.h_dim_list:
            # dense -> batchnorm -> actv -> dropout
            self.layer_list.append(nn.Linear(h_dim_prev,h_dim))
            if self.use_bn: self.layer_list.append(nn.BatchNorm1d(num_features=h_dim)) # (optional) batchnorm
            self.layer_list.append(self.actv)
            self.layer_list.append(nn.Dropout1d(p=self.p_drop))
            h_dim_prev = h_dim
        self.layer_list.append(
            MixturesOfGaussianLayer(
                in_dim = h_dim_prev,
                out_dim = self.y_dim,
                k = self.k,
                sig_max = self.sig_max
            )
        )

        # Definie network
        self.net = nn.Sequential()
        self.layer_names = []
        for l_idx,layer in enumerate(self.layer_list):
            layer_name = "%s_%02d"%(type(layer).__name__.lower(),l_idx)
            self.layer_names.append(layer_name)
            self.net.add_module(layer_name,layer)

        # Initialize parameters
        self.init_param(VERBOSE=False)

    def init_param(self,VERBOSE=False):
        """
            Initialize parameters
        """
        for m_idx,m in enumerate(self.modules()):
            if VERBOSE:
                print ("[%02d]"%(m_idx))
            if isinstance(m,nn.Conv2d): # init conv
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m,nn.BatchNorm1d): # init BN
                nn.init.constant_(m.weight,1.0)
                nn.init.constant_(m.bias,0.0)
            elif isinstance(m,nn.Linear): # lnit dense
                nn.init.kaiming_normal_(m.weight,nonlinearity='relu')
                nn.init.zeros_(m.bias)
        # (Hueristics) mu bias between mu_min ~ mu_max
        self.layer_list[-1].fc_mu.bias.data.uniform_(self.mu_min,self.mu_max)
        
    def forward(self, x):
        """
            Forward propagate
        """
        intermediate_output_list = []
        for idx,layer in enumerate(self.net):
            x = layer(x)
            intermediate_output_list.append(x)
        # Final output
        final_output = x
        return final_output

def eval_mdn_1d(
    mdn,
    x_train_np,
    y_train_np,
    figsize=(12,3),
    device='cpu',
):
    # Eval
    mdn.eval()
    x_margin = 0.2
    x_test = th.linspace(
        start = x_train_np.min()-x_margin,
        end   = x_train_np.max()+x_margin,
        steps = 300
    ).reshape((-1,1)).to(device)
    pi_test,mu_test,sigma_test = mdn.forward(x_test)

    # Get the most probable mu
    argmax_mu_test = get_argmax_mu(pi_test,mu_test) # [N x D]

    # To numpy array
    x_test_np,pi_np,mu_np,sigma_np = th2np(x_test),th2np(pi_test),th2np(mu_test),th2np(sigma_test)
    argmax_mu_test_np = th2np(argmax_mu_test) # [N x D]
    
    # Uncertainties
    epis_unct,alea_unct = gmm_uncertainties(pi_test,mu_test,sigma_test) # [N x D]
    epis_unct_np,alea_unct_np = th2np(epis_unct),th2np(alea_unct)
    
    # Plot fitted results
    y_dim = y_train_np.shape[1]
    plt.figure(figsize=figsize)
    cmap = plt.get_cmap('gist_rainbow')
    colors = [cmap(ii) for ii in np.linspace(0, 1, mdn.k)] # colors 
    pi_th = 0.1
    for d_idx in range(y_dim): # for each output dimension
        plt.subplot(1,y_dim,d_idx+1)
        # Plot training data
        plt.plot(x_train_np,y_train_np[:,d_idx],'.',color=(0,0,0,0.2),markersize=3,
                 label="Training Data")
        # Plot mixture standard deviations
        for k_idx in range(mdn.k): # for each mixture
            pi_high_idx = np.where(pi_np[:,k_idx,d_idx] > pi_th)[0]
            mu_k = mu_np[:,k_idx,d_idx]
            sigma_k = sigma_np[:,k_idx,d_idx]
            upper_bound = mu_k + 2*sigma_k
            lower_bound = mu_k - 2*sigma_k
            plt.fill_between(x_test_np[pi_high_idx,0].squeeze(),
                             lower_bound[pi_high_idx],
                             upper_bound[pi_high_idx],
                             facecolor=colors[k_idx], interpolate=False, alpha=0.3)
        # Plot mixture means
        for k_idx in range(mdn.k): # for each mixture
            pi_high_idx = np.where(pi_np[:,k_idx,d_idx] > pi_th)[0]  # [?,]
            pi_low_idx = np.where(pi_np[:,k_idx,d_idx] <= pi_th)[0]  # [?,]
            plt.plot(x_test_np[pi_high_idx,0],mu_np[pi_high_idx,k_idx,d_idx],'-',
                     color=colors[k_idx],linewidth=1/2)
            plt.plot(x_test_np[pi_low_idx,0],mu_np[pi_low_idx,k_idx,d_idx],'-',
                     color=(0,0,0,0.3),linewidth=1/2)

        # Plot most probable mu
        plt.plot(x_test_np[:,0],argmax_mu_test_np[:,d_idx],'-',color='b',linewidth=2,
                 label="Argmax Mu")
        plt.xlim(x_test_np.min(),x_test_np.max())
        plt.legend(loc='lower right',fontsize=8)
        plt.title("y_dim:[%d]"%(d_idx),fontsize=10)
    plt.show()
    
    # Plot uncertainties
    plt.figure(figsize=figsize)
    for d_idx in range(y_dim): # for each output dimension
        plt.subplot(1,y_dim,d_idx+1)
        plt.plot(x_test_np[:,0],epis_unct_np[:,d_idx],'-',color='r',linewidth=2,
                 label="Epistemic Uncertainty")
        plt.plot(x_test_np[:,0],alea_unct_np[:,d_idx],'-',color='b',linewidth=2,
                 label="Aleatoric Uncertainty")
        plt.xlim(x_test_np.min(),x_test_np.max())
        plt.legend(loc='lower right',fontsize=8)
        plt.title("y_dim:[%d]"%(d_idx),fontsize=10)
    plt.show()
    
    # Back to train
    mdn.train()
    