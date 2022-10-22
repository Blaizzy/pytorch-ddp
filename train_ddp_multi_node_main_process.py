import os 
import hashlib
import time


import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist

import torchvision
import neptune.new as neptune


from utils import setup_for_distributed, save_on_master, is_main_process


def create_data_loader_cifar10():
    rank = int(os.environ['RANK'])
    
    transform = transforms.Compose(
        [
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 256
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)                                  
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset=trainset, rank=rank)                                                  
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            sampler=train_sampler, num_workers=14, pin_memory=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    test_sampler = torch.utils.data.distributed.DistributedSampler(dataset=testset, rank=rank)                                         
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, sampler=test_sampler, num_workers=14)
    return trainloader, testloader


def train(net, trainloader, run, rank):
    print("Start training...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    epochs = 2
    num_of_batches = len(trainloader)
    for epoch in range(epochs):  # loop over the dataset multiple times
        trainloader.sampler.set_epoch(epoch)
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            images, labels = inputs.to(f'cuda:{rank}'), labels.to(f'cuda:{rank}')

            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            outputs = net(images)
            loss = criterion(outputs, labels)

            
            loss.backward()
            optimizer.step()
            
            # print(loss.get_device())
            
            # synchronizes all the threads to reach this point before moving on
            dist.reduce(tensor=loss, dst=0)
            dist.barrier()
            if rank==0:
                running_loss += (loss.item() / dist.get_world_size())
    
            
        if rank==0:
            epoch_loss = running_loss / num_of_batches
            run['metrics/train/loss'].log(epoch_loss)    
            print(f'[Epoch {epoch + 1}/{epochs}] loss: {epoch_loss:.3f}')
    
    print('Finished Training')


def test(net, PATH, testloader, run, rank):
    # if is_main_process:
    #     net.load_state_dict(torch.load(PATH))
    # dist.barrier()

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data

            images, labels = images.to(f'cuda:{rank}'), labels.to(f'cuda:{rank}')
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            
            dist.reduce(tensor=labels, dst=0)
            dist.barrier()
            dist.reduce(tensor=predicted, dst=0)
            dist.barrier()

            if rank==0:
                correct+=(predicted == labels).sum().item()


    
    if rank == 0:
        acc = 100 * correct // total
        run['metrics/valid/acc'] = acc
        print(f'Accuracy of the network on the 10000 test images: {acc} %')

def init_distributed():

    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    dist_url = "env://" # default

    # only works with torch.distributed.launch // torch.run
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])

    dist.init_process_group(
            backend="nccl",
            init_method=dist_url,
            world_size=world_size,
            rank=rank)

    # synchronizes all the threads to reach this point before moving on
    dist.barrier()
    # setup_for_distributed(rank == 0)


if __name__ == '__main__':
    start = time.time()
    
    init_distributed()
    rank = int(os.environ["RANK"])
    
    
    PATH = './cifar_net.pth'
    trainloader, testloader = create_data_loader_cifar10()
    
    rank = int(os.environ["RANK"])
    net = torchvision.models.resnet50(False).to(f'cuda:{rank}')
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)

    # Convert BatchNorm to SyncBatchNorm. 
    net = nn.SyncBatchNorm.convert_sync_batchnorm(net)

    # local_rank = int(os.environ['LOCAL_RANK'])
    net = nn.parallel.DistributedDataParallel(
        net, 
        device_ids=[rank]
    )
    
    # init neptune
    run = neptune.init_run(
        project='common/showroom',
        api_token='ANONYMOUS',
        monitoring_namespace=f"monitoring/rank/{rank}",
    )
        
    
    start_train = time.time()
    train(net, trainloader, run, rank)
    end_train = time.time()

    # test
    test(net, PATH, testloader, run, rank)
    
    run.wait()
    end = time.time()
    seconds = (end - start)
    seconds_train = (end_train - start_train)
    print(f"Total elapsed time: {seconds:.2f} seconds, \
     Train 1 epoch {seconds_train:.2f} seconds")

# Log from one process (multi-node single GPU)
# Two terminals 
# torchrun --nproc_per_node=1 --nnodes=2 --node_rank=0 train_ddp_multi_node_main_process.py
# torchrun --nproc_per_node=1 --nnodes=2 --node_rank=1 train_ddp_multi_node_main_process.py

# For multi GPU: https://pytorch.org/docs/stable/distributed.html#multi-gpu-collective-functions
