import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import copy
import torch
from torch import nn, autograd
import torch.nn.functional as F

def model_string_change(asis, tobe, idx):
    mylist = list(asis)
    mylist[idx] = tobe
    new_str = ''.join(mylist)
    return new_str


def get_models(mode, args):  
    
    layer_num = len(mode) # cifar ResNet has 3 layers: class ResNet self.layer1, self.layer2, self.layer3
    x = [[]]*layer_num

    if args.model_name == 'resnet32':
        for i in range(layer_num): # alphabet-wise search
            temp = [mode[i]]
            if args.dynamic:
                if mode[i] == 'b':
                    temp.append('c')
                elif mode[i] == 'd':
                    temp.append('e')
            x[i] = temp
    elif 'resnet56' in args.model_name:
        for i in range(layer_num):
            temp = [mode[i]]
            if args.dynamic:
                if mode[i] == '1':
                    temp.append('1a')
                    temp.append('1b')
                elif mode[i] == '2':
                    temp.append('2a')
                    temp.append('2b')
                    temp.append('2c')
                elif mode[i] == '3':
                    temp.append('3a')
                elif mode[i] == '5':
                    temp.append('5a')
                    temp.append('5b')
                    temp.append('5c')
                    # temp.append('5d')
                    # temp.append('5e')
                    # temp.append('5b1')
                    # temp.append('5c1')
                    # temp.append('5d1')
                    # temp.append('5e1')
                elif mode[i] == '6':
                    temp.append('6a')
                    temp.append('6b')
                    temp.append('6c')
                    # temp.append('6c1')
            x[i] = temp
    
    modes = [[]]
    # modes = [['a', 'b'], ['a', 'c']]
    b = 1 # branch num
    for i in range(layer_num):
        temp = copy.deepcopy(modes)
        for k in range(len(x[i])-1):
            modes = modes+copy.deepcopy(temp)

        for j in range(b*len(x[i])):
            modes[j].append(x[i][j//b])
        b *= len(x[i])

    option_list = []
    if args.model_name == 'resnet32':
        for j in range(len(modes)):
            mode = modes[j]
            step_list = []
            for i in range(layer_num):
                if mode[i] == 'a':
                    x = [1,1,1,1,1]
                elif mode[i] == 'b': # 1 zero
                    x = [1,2,0,1,1]
                elif mode[i] == 'c': # 1 zero
                    x = [1,1,2,0,1]
                elif mode[i] == 'd': # 2 zero
                    x = [1,3,0,0,1]
                elif mode[i] == 'e': # 2 zero
                    x = [1,2,0,2,0]
                elif mode[i] == 'f': # 3 zero
                    x = [1,4,0,0,0]
                elif mode[i] == 'g': # 4 zero
                    x = [5,0,0,0,0]
                elif mode[i] == 'h':
                    x = [2,0,4,0,0]
                elif mode[i] == 'i':
                    x = [0,4,0,0,0]
                step_list.append(x)
            option_list.append(step_list)
    elif 'resnet56' in args.model_name:
        for j in range(len(modes)):
            mode = modes[j]
            step_list = []
            for i in range(layer_num):
                if mode[i] == '0':
                    x = [1]*9
                elif mode[i][0] == '1': # 1 zero
                    if mode[i] == '1a':
                        x = [1]*5 + [2,0] + [1]*2
                    elif mode[i] == '1b':
                        x = [1]*4 + [2,0] + [1]*3
                    else:
                        x = [1]*6 + [2,0] + [1]*1
                        if args.model_set == 2:
                            x = [1]*5 + [2,0] + [1]*2 # (2)
                elif mode[i][0] == '2':
                    if mode[i] == '2a':
                        x = [1]*3 + [2,0,2,0] + [1]*2
                    elif mode[i] == '2b':
                        x = [1]*2 + [2,0,2,0] + [1]*3
                    elif mode[i] == '2c':
                        x = [1]*1 + [2,0,2,0] + [1]*4
                    else:
                        x = [1]*4 + [2,0,2,0] + [1]*1
                        x = [1]*2 + [2,0,2,0] + [1]*3 # (2)
                elif mode[i][0] == '3':
                    if mode[i] == '3a':
                        x = [1]*1 + [2,0,2,0,2,0] + [1]*2
                    else:
                        x = [1]*2 + [2,0,2,0,2,0] + [1]*1
                elif mode[i] == '4':
                    x = [1,3,0,0,2,0,2,0,1]
                elif mode[i][0] == '5':
                    if mode[i] == '5a':
                        x = [1,6] + [0]*5 + [1]*2
                    elif mode[i] == '5b':
                        x = [1,5] + [0]*4 + [2,0,1]
                    elif mode[i] == '5b1':
                        x = [1,5] + [0]*4 + [1,2,0]
                    elif mode[i] == '5c':
                        x = [1,4] + [0]*3 + [3,0,0,1]
                    elif mode[i] == '5c1':
                        x = [1,4] + [0]*3 + [1,3,0,0,]
                    elif mode[i] == '5d':
                        x = [1,3] + [0]*2 + [4,0,0,0,1]
                    elif mode[i] == '5d1':
                        x = [1,3] + [0]*2 + [1,4,0,0,0]
                    elif mode[i] == '5e':
                        x = [1,2] + [0]*1 + [5,0,0,0,0,1]
                    elif mode[i] == '5e1':
                        x = [1,2] + [0]*1 + [1,5,0,0,0,0]
                    else:
                        x = [1,1,6] + [0]*5 + [1]
                        if args.model_set == 1:
                            x = [1,4] + [0]*3 + [3,0,0,1] # (1)
                elif mode[i][0] == '6':
                    if mode[i] == '6a':
                        x = [1,1,7,0,0,0,0,0,0]
                    elif mode[i] == '6b':
                        x = [1,6,0,0,0,0,0,2,0]
                    elif mode[i] == '6c':
                        x = [1,5,0,0,0,0,3,0,0]
                    elif mode[i] == '6c1':
                        x = [1,3,0,0,5,0,0,0,0]
                    else:
                        x = [1,7,0,0,0,0,0,0,1]
                        if args.model_set == 1:
                            x = [1,4,0,0,0,4,0,0,0] # (1)
                elif mode[i] == '7':
                    x = [1,8,0,0,0,0,0,0,0]
                step_list.append(x)
            option_list.append(step_list)
    
    return option_list


def get_sd_models(mode, args):  
    # stochastic depth
    layer_num = len(mode) # cifar ResNet has 3 layers: class ResNet self.layer1, self.layer2, self.layer3
    x = [[]]*layer_num

    if args.model_name == 'resnet32':
        for i in range(layer_num): # alphabet-wise search
            temp = [mode[i]]
            if args.dynamic:
                if mode[i] == 'b':
                    temp.append('c')
                elif mode[i] == 'd':
                    temp.append('e')
            x[i] = temp
    elif args.model_name == 'resnet56':
        for i in range(layer_num):
            temp = [mode[i]]
            if args.dynamic:
                if mode[i] == '1':
                    temp.append('1a')
                    temp.append('1b')
                elif mode[i] == '2':
                    temp.append('2a')
                    temp.append('2b')
            x[i] = temp
    
    modes = [[]]
    # modes = [['a', 'b'], ['a', 'c']]
    b = 1 # branch num
    for i in range(layer_num):
        temp = copy.deepcopy(modes)
        for k in range(len(x[i])-1):
            modes = modes+copy.deepcopy(temp)

        for j in range(b*len(x[i])):
            modes[j].append(x[i][j//b])
        b *= len(x[i])

    option_list = []
    if args.model_name == 'resnet32':
        for j in range(len(modes)):
            mode = modes[j]
            step_list = []
            for i in range(layer_num):
                if mode[i] == 'a':
                    x = [1,1,1,1,1]
                elif mode[i] == 'b': # 1 zero
                    x = [1,2,0,1,1]
                elif mode[i] == 'c': # 1 zero
                    x = [1,1,2,0,1]
                elif mode[i] == 'd': # 2 zero
                    x = [1,3,0,0,1]
                elif mode[i] == 'e': # 2 zero
                    x = [1,2,0,2,0]
                elif mode[i] == 'f': # 3 zero
                    x = [1,4,0,0,0]
                elif mode[i] == 'g': # 4 zero
                    x = [5,0,0,0,0]
                elif mode[i] == 'h':
                    x = [2,0,4,0,0]
                elif mode[i] == 'i':
                    x = [0,4,0,0,0]
                step_list.append(x)
            option_list.append(step_list)
    elif args.model_name == 'resnet56':
        for j in range(len(modes)):
            mode = modes[j]
            step_list = []
            for i in range(layer_num):
                if mode[i] == '0':
                    x = [1]*9
                elif mode[i][0] == '1': # 1 zero
                    if mode[i] == '1a':
                        x = [1]*6 + [0] + [1]*2
                    elif mode[i] == '1b':
                        x = [1]*5 + [0] + [1]*3
                    elif mode[i] == '1c':
                        x = [1]*4 + [0] + [1]*4
                    elif mode[i] == '1d':
                        x = [1]*3 + [0] + [1]*5
                    else:
                        x = [1]*7 + [0] + [1]*1
                elif mode[i][0] == '2':
                    if mode[i] == '2a':
                        x = [1]*4 + [0,1,0] + [1]*2
                    elif mode[i] == '2b':
                        x = [1]*3 + [0,1,0] + [1]*3
                    elif mode[i] == '2c':
                        x = [1]*5 + [0,1,0] + [1]*1
                    else:
                        x = [1]*6 + [0]*2 + [1]*1
                elif mode[i] == '3':
                    x = [1]*5 + [0]*3 + [1]*1
                elif mode[i] == '4':
                    x = [1]*4 + [0]*4 + [1]*1
                elif mode[i] == '5':
                    x = [1]*3 + [0]*5 + [1]*1
                elif mode[i] == '6':
                    x = [1]*2 + [0]*6 + [1]*1
                elif mode[i] == '7':
                    x = [1]*1 + [0]*7 + [1]*1
                step_list.append(x)
            option_list.append(step_list)
    
    return option_list

# def get_models56(mode, args):
#     option_list = []    
    
#     layer_num = len(mode) # cifar ResNet has 3 layers: class ResNet self.layer1, self.layer2, self.layer3
#     x = [[]]*layer_num  
#     for i in range(layer_num): # alphabet-wise search
#         temp = [mode[i]]
#         if args.dynamic:
#             if mode[i] == 'b':
#                 temp.append('c')
#             elif mode[i] == 'd':
#                 temp.append('e')
#         x[i] = temp
    
#     modes = [[]]
#     # modes = [['a', 'b'], ['a', 'c']]
#     b = 1 # branch num
#     for i in range(layer_num):
#         for k in range(len(x[i])-1):
#             modes = modes+copy.deepcopy(modes)
#         for j in range(b*len(x[i])):
#             modes[j].append(x[i][j//b])
#         b *= len(x[i])

#     for j in range(len(modes)):
#         mode = modes[j]
#         step_list = []
#         for i in range(layer_num):
#             if mode[i] == 'a':
#                 x = [1,1,1,1,1]
#             elif mode[i] == 'b': # 1 zero
#                 x = [1,2,0,1,1]
#             elif mode[i] == 'c': # 1 zero
#                 x = [1,1,2,0,1]
#             elif mode[i] == 'd': # 2 zero
#                 x = [1,3,0,0,1]
#             elif mode[i] == 'e': # 2 zero
#                 x = [1,2,0,2,0]
#             elif mode[i] == 'f': # 3 zero
#                 x = [1,4,0,0,0]
#             elif mode[i] == 'g': # 4 zero
#                 x = [5,0,0,0,0]
#             elif mode[i] == 'h':
#                 x = [2,0,4,0,0]
#             elif mode[i] == 'i':
#                 x = [0,4,0,0,0]
#             step_list.append(x)
#         option_list.append(step_list)
#     return option_list

    # 'bca-aab-aad-cee-eeg-ggg'
        # full_stepSize, # follow the order (the larger, the earlier)
    #         [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 2, 0, 2, 0]],
    #         [[1, 1, 1, 1, 1], [1, 2, 0, 2, 0], [1, 1, 1, 1, 1]],
    #         [[1, 2, 0, 2, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
    #         [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 4, 0, 0, 0]],
    #         [[1, 1, 1, 1, 1], [1, 4, 0, 0, 0], [1, 1, 1, 1, 1]],
    #         [[1, 4, 0, 0, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]
    #     ]

    # elif args.mode == 'b':
    #      s2D = [
    #     [[1, 2, 0, 2, 0], [1, 1, 1, 1, 1], [1, 2, 0, 2, 0]],
    #     [[1, 1, 1, 1, 1], [1, 2, 0, 2, 0], [1, 1, 1, 1, 1]],
    #     [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 2, 0, 2, 0]],
    #     [[1, 1, 1, 1, 1], [1, 4, 0, 0, 0], [1, 4, 0, 0, 0]],
    #     [[1, 4, 0, 0, 0], [1, 4, 0, 0, 0], [1, 1, 1, 1, 1]],
    #     [[1, 4, 0, 0, 0], [1, 1, 1, 1, 1], [1, 4, 0, 0, 0]]
    #     ]

    # elif args.mode == 'c':
    #      s2D = [
    #     [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 4, 0, 0, 0]]
    #     ]*args.num_users
    
    # elif args.mode == 'd':
    #      s2D = [
    #     [[1, 1, 1, 1, 1], [1, 4, 0, 0, 0], [1, 4, 0, 0, 0]]
    #     ]*args.num_users
    
    # elif args.mode == 'e':
    #      s2D = [
    #     [[1, 4, 0, 0, 0], [1, 4, 0, 0, 0], [1, 4, 0, 0, 0]]
    #     ]*args.num_users

    # elif args.mode == 'f':
    #      s2D = [
    #     [[5, 0, 0, 0, 0], [5, 0, 0, 0, 0], [5, 0, 0, 0, 0]]
    #     ]*args.num_users    

    # elif args.mode == 'g':
    #      s2D = [
    #     [[1, 2, 0, 2, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
    #     [[1, 1, 1, 1, 1], [1, 2, 0, 2, 0], [1, 1, 1, 1, 1]],
    #     [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 2, 0, 2, 0]],
    #     [[1, 1, 2, 0, 1], [1, 1, 2, 0, 1], [1, 1, 1, 1, 1]],
    #     [[1, 1, 2, 0, 1], [1, 1, 1, 1, 1], [1, 1, 2, 0, 1]],
    #     [[5, 0, 0, 0, 0], [5, 0, 0, 0, 0], [5, 0, 0, 0, 0]]
    # ]

    # elif args.mode == 'h':
    #      s2D = [
    #     [[1, 4, 0, 0, 0], [1, 4, 0, 0, 0], [5, 0, 0, 0, 0]]
    #     ]*args.num_users

    # elif args.mode == 'h2':
    #      s2D = [
    #     [[1, 4, 0, 0, 0], [1, 4, 0, 0, 0], [1, 4, 0, 0, 0]]
    #     ]*args.num_users

    # elif args.mode == 'i':
    #      s2D = [
    #     [[1, 2, 0, 2, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
    #     [[1, 1, 1, 1, 1], [1, 2, 0, 2, 0], [1, 1, 1, 1, 1]],
    #     [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 2, 0, 2, 0]],
    #     [[1, 1, 2, 0, 1], [1, 1, 2, 0, 1], [1, 1, 1, 1, 1]],
    #     [[1, 1, 2, 0, 1], [1, 1, 1, 1, 1], [1, 1, 2, 0, 1]],
    #     [[1, 4, 0, 0, 0], [1, 4, 0, 0, 0], [5, 0, 0, 0, 0]]
    # ]

    # elif args.mode == 'i2':
    #      s2D = [
    #     [[1, 2, 0, 2, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
    #     [[1, 1, 1, 1, 1], [1, 2, 0, 2, 0], [1, 1, 1, 1, 1]],
    #     [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 2, 0, 2, 0]],
    #     [[1, 1, 2, 0, 1], [1, 1, 2, 0, 1], [1, 1, 1, 1, 1]],
    #     [[1, 1, 2, 0, 1], [1, 1, 1, 1, 1], [1, 1, 2, 0, 1]],
    #     [[1, 4, 0, 0, 0], [1, 4, 0, 0, 0], [1, 4, 0, 0, 0]]
    # ]




def get_fl_cifar_datasets():
    transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

    transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

    dataset_train = datasets.CIFAR10('.data/cifar', train=True, download=True, transform=transform_train) # augmentation on client should be implemented
    dataset_test = datasets.CIFAR10('.data/cifar', train=False, download=True, transform=transform_test)

    return dataset_train, dataset_test

def cifar_iid(dataset, num_users, seed):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    np.random.seed(seed)

    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def cifar_iid_all(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset))
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        # all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def get_fl_svhn_datasets(data_aug=False, batch_size=128, test_batch_size=1000):
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=63. / 255., saturation=[0.5, 1.5], contrast=[0.2, 1.8]),
        transforms.ToTensor(),
        transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
    ])

    dataset_train = datasets.SVHN(root='.data/svhn', split='train', download=True, transform=transform_train)
    dataset_test = datasets.SVHN(root='.data/svhn', split='train', download=True, transform=transform_test)

    return dataset_train, dataset_test

def count_param(step_vector):
    paramNum = 1114 # ResNet 18
    for i in range(5):
        j = 1 if step_vector[0][i] else 0
        paramNum += j*4672
    j = 1 if step_vector[1][0] else 0
    paramNum += j*13952
    j = 1 if step_vector[2][0] else 0
    paramNum += j*55552
    for i in range(1,5):
        j = 1 if step_vector[1][i] else 0
        paramNum += j*18560
    for i in range(1,5):
        j = 1 if step_vector[2][i] else 0
        paramNum += j*73984
    return paramNum

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, weight_decay=0):
        # torch.manual_seed(args.rs)
        # torch.cuda.manual_seed(args.rs)
        # torch.cuda.manual_seed_all(args.rs)
        self.args = args
        self.device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
        self.loss_func = nn.CrossEntropyLoss().to(self.device)
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True, num_workers=2)
        self.weight_decay = weight_decay

    def train(self, net, learning_rate):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=self.args.momentum, weight_decay=self.weight_decay)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        w = net.state_dict()
        BN = {}
        for key in w.keys():
            if len(w[key].shape)<=1 and key!='linear.bias':
                BN[key] = w[key]
        return w, BN, sum(epoch_loss) / len(epoch_loss)
    
    def sBN(self, net):
        net.train()
        # train and update
        for batch_idx, (images, labels) in enumerate(self.ldr_train):
            images, labels = images.to(self.device), labels.to(self.device)
            net.zero_grad()
            net(images)
            
        w = net.state_dict()
        return w


class LocalUpdateKD(object):
    def __init__(self, args, dataset=None, idxs=None, weight_decay=0):
        # torch.manual_seed(args.rs)
        # torch.cuda.manual_seed(args.rs)
        # torch.cuda.manual_seed_all(args.rs)
        self.args = args
        self.device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
        self.loss_func = nn.CrossEntropyLoss().to(self.device)
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True, num_workers=2)
        self.weight_decay = weight_decay

    def train(self, tnet, snet, learning_rate):
        # tnet.train()
        tnet.eval()
        snet.train()
        # train and update
        # t_optimizer = torch.optim.SGD(tnet.parameters(), lr=learning_rate, momentum=self.args.momentum, weight_decay=self.weight_decay)
        s_optimizer = torch.optim.SGD(snet.parameters(), lr=learning_rate, momentum=self.args.momentum, weight_decay=self.weight_decay)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                teacher_logits = tnet(images)
                logits = snet(images)
                T = 1
                distillation_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(logits/T, dim=1), F.softmax(teacher_logits/T, dim=1)) * (T * T)
                loss = distillation_loss
                # loss = self.loss_func(t_log_probs, s_log_probs)
                distillation_loss.backward()
                # t_optimizer.step()
                s_optimizer.step()
                s_optimizer.zero_grad()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        w = snet.state_dict()
        BN = {}
        for key in w.keys():
            if len(w[key].shape)<=1 and key!='linear.bias':
                BN[key] = w[key]
        return tnet.state_dict(), snet.state_dict(), BN, sum(epoch_loss) / len(epoch_loss)


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def FedAvgDV(w, agg_layer_keys, non_agg_layer_keys):
    w_avg = copy.deepcopy(w[0])
    for k in agg_layer_keys:
        if k in agg_layer_keys:
            for i in range(1, len(w)):
                w_avg[k] += w[i][k]
            w_avg[k] = torch.div(w_avg[k], len(w))
    
    # for k in non_agg_layer_keys:
    #     w_avg[k] *= len(w)
    #     w_avg[k] = torch.div(w_avg[k], len(w))

    return w_avg

def FedAvgDV2(w, agg_layer_keys, non_agg_layer_keys):
    w_avg = copy.deepcopy(w[0])
    for k in agg_layer_keys[1]:
        if k in agg_layer_keys[0]:
            for i in range(1, len(w)):
                w_avg[k] += w[i][k]
            w_avg[k] = torch.div(w_avg[k], len(w))
        elif k in non_agg_layer_keys[0]:
            w_avg[k] = w[1][k]

    # for k in non_agg_layer_keys:
    #     w_avg[k] *= len(w)
    #     w_avg[k] = torch.div(w_avg[k], len(w))

    return w_avg


def CrowdAvg(w, common_layers, singular_layers, agg_user_indices):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] = 0*w_avg[k]

    for k in common_layers:
        for i in range(len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    
    for k in singular_layers:
        layer_idx = int(k[5])-1
        block_idx = int(k[7])
        for i in agg_user_indices[layer_idx][block_idx]:
            w_avg[k] += w[i][k]
        if len(agg_user_indices[layer_idx][block_idx]):
            w_avg[k] = torch.div(w_avg[k], len(agg_user_indices[layer_idx][block_idx]))
    return w_avg

def DVAvg(prev_w, w, common_layers, singular_layers, s2Ds):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] = 0*w_avg[k]

    for k in common_layers:
        for i in range(len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    
    for k in singular_layers:
        layer_idx = int(k[5])-1
        block_idx = int(k[7])
        
        temp = 0
        for i in range(len(w)):
            if s2Ds[i][layer_idx][block_idx]:
                w_avg[k] += w[i][k]
                temp += 1
        if temp:
            w_avg[k] = torch.div(w_avg[k], temp)
        else:
            w_avg[k] = prev_w[k]
                
    return w_avg

def test_img(net_g, datatest, args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.test_bs)
    l = len(data_loader)
    with torch.no_grad():
        for idx, (data, target) in enumerate(data_loader):
            if args.gpu != -1:
                with torch.cuda.device(args.gpu):        
                    data, target = data.cuda(), target.cuda()
            log_probs = net_g(data)
            # sum up batch loss
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            # get the index of the max log-probability
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.
            test_loss, correct, len(data_loader.dataset), accuracy)
    return accuracy, test_loss
