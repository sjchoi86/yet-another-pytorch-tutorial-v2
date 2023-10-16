import torch
from torchvision import datasets,transforms

def mnist(root_path='./data/',batch_size=128):
    """ 
        MNIST
    """
    mnist_train = datasets.MNIST(root=root_path,train=True,transform=transforms.ToTensor(),download=True)
    mnist_test  = datasets.MNIST(root=root_path,train=False,transform=transforms.ToTensor(),download=True)
    train_iter  = torch.utils.data.DataLoader(mnist_train,batch_size=batch_size,shuffle=True,num_workers=1)
    test_iter   = torch.utils.data.DataLoader(mnist_test,batch_size=batch_size,shuffle=True,num_workers=1)
    # Data
    train_data,train_label = mnist_train.data,mnist_train.targets
    test_data,test_label = mnist_test.data,mnist_test.targets
    return train_iter,test_iter,train_data,train_label,test_data,test_label