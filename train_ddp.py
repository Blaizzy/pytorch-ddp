"""
Mostly based on the official pytorch tutorial
Link: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
Modified for educational purposes.
Nikolas, AI Summer
"""
import os 
gpu_list = "0,1"
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import time
import torchvision
import neptune.new as neptune
from neptune.new.types import File

from utils import setup_for_distributed, save_on_master, is_main_process
import hashlib


def create_data_loader_cifar10():
    transform = transforms.Compose(
        [
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 256

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)                                  
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset=trainset, shuffle=True)                                                  
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            sampler=train_sampler, num_workers=16, pin_memory=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    test_sampler = torch.utils.data.distributed.DistributedSampler(dataset=testset, shuffle=True)                                         
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, sampler=test_sampler, num_workers=16)
    dataset_sizes = {'train': len(trainloader), 'test': len(testloader)}
    return trainloader, testloader, dataset_sizes


def train(net, trainloader, dataset_sizes, rank):
    if rank == 0:
        run = neptune.init_run(
            project='common/showroom',
            api_token='ANONYMOUS',
            custom_run_id=os.environ['NEPTUNE_CUSTOM_RUN_ID'] 
        )

    print("Start training...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    epochs = 5
    num_of_batches = len(trainloader)
    for epoch in range(epochs):  # loop over the dataset multiple times
        trainloader.sampler.set_epoch(epoch)
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            images, labels = inputs.cuda(), labels.cuda() 

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # optionally gather every n iterations of the batch
            if rank == 0:
                # output = [loss.clone() for _ in range(dist.get_world_size())]
                dist.reduce(tensor=loss, dst=rank)
                running_loss += loss.item()
      

        print(f'[Epoch {epoch + 1}/{epochs}] loss: {running_loss / num_of_batches:.3f}')

        if rank == 0:
            epoch_loss = running_loss / dataset_sizes['train']
            run['metrics/epoch/loss'].log(epoch_loss)
    
    print('Finished Training')
    run.stop()
    



def test(net, PATH, testloader, rank):
    # if is_main_process:
    #     net.load_state_dict(torch.load(PATH))
    # dist.barrier()
    if rank == 0:
        run = neptune.init_run(
            project='common/showroom',
            api_token='ANONYMOUS',
            custom_run_id=os.environ['NEPTUNE_CUSTOM_RUN_ID'] 
        )

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data

            images, labels = images.cuda(), labels.cuda() 
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            if rank == 0:
                dist.reduce(tensor=predicted, dst=rank)
                dist.reduce(tensor=labels, dst=rank)
                correct+=(predicted == labels).sum().item()
                

        acc = 100 * correct // total
        
        if rank == 0:
            run['metrics/valid/acc'] = acc 
            
        
    print(f'Accuracy of the network on the 10000 test images: {acc} %')
    run.stop()

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
            world_size=world_size)

    # this will make all .cuda() calls work properly
    torch.cuda.set_device(local_rank)
    # synchronizes all the threads to reach this point before moving on
    dist.barrier()
    setup_for_distributed(rank == 0)


if __name__ == '__main__':
    start = time.time()
    
    os.environ['NEPTUNE_CUSTOM_RUN_ID'] = hashlib.md5(str(time.time()).encode()).hexdigest()
    
    init_distributed()
    
    PATH = './cifar_net.pth'
    trainloader, testloader, dataset_sizes = create_data_loader_cifar10()
    net = torchvision.models.resnet50(False).cuda()

    # Convert BatchNorm to SyncBatchNorm. 
    net = nn.SyncBatchNorm.convert_sync_batchnorm(net)

    local_rank = int(os.environ['LOCAL_RANK'])
    net = nn.parallel.DistributedDataParallel(net, device_ids=[local_rank])
    
    start_train = time.time()
    train(net, trainloader, dataset_sizes, local_rank)
    end_train = time.time()
    # save
    if loca_rank==0:
        save_on_master(net.state_dict(), PATH)
    dist.barrier()

    # test
    test(net, PATH, testloader, local_rank)

    end = time.time()
    seconds = (end - start)
    seconds_train = (end_train - start_train)
    print(f"Total elapsed time: {seconds:.2f} seconds, \
     Train 1 epoch {seconds_train:.2f} seconds")