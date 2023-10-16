import numpy as np
import matplotlib.pyplot as plt
import torch as th


def get_torch_size_string(x):
    return "x".join(map(str,x.shape))

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
