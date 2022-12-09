from urllib.parse import non_hierarchical
import torch
import torch.nn as nn
from utils.packs import *
from utils.fl_pack import *
from random import randint
import copy
from models import *
import random
from torch.autograd import Variable
import torchsummary

import os, argparse
import wandb
import numpy as np
from datetime import datetime
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

parser = argparse.ArgumentParser()
parser.add_argument('--nrounds', type=int, default=500)
parser.add_argument('--nftrounds', type=int, default=10) # fine-tuning round

parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--data_aug', type=bool, default=True)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--test_bs', type=int, default=32, help="test batch size")

parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
parser.add_argument('--frac', type=float, default=0.1, help="number of users: K")

parser.add_argument('--momentum', type=float, default=0, help="SGD momentum (default: 0.9)")
parser.add_argument('--local_bs', type=int, default=32, help="local batch size: B")
parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
parser.add_argument('--verbose', action='store_true', help='verbose print')
parser.add_argument('--cluster_select', action='store_false') # default: true
parser.add_argument('--dynamic', action='store_true') # default: false
parser.add_argument('--log_all', action='store_true') # default: false
parser.add_argument('--log_rate', type=float, default=0.1)
parser.add_argument('--larg_lr', action='store_true') # default: false
parser.add_argument('--KD', action='store_true') # default: false

parser.add_argument('--full_condition', action='store_true') # default: false
parser.add_argument('--worst_three', action='store_true') # default: false
parser.add_argument('--min_flex_num', type=int, default=0, help="0:0~ max(0,tc-args.min_flex_num)")
parser.add_argument('--max_flex_num', type=int, default=4, help="0:~4 min(tc+args.max_flex_num+1,5)")

parser.add_argument('--model_set', type=int, default=2)
parser.add_argument('--model_cluster_idx', type=int, default=4)
parser.add_argument('--model_name', type=str, default='resnet56') # 34, 56, 110
parser.add_argument('--wandb', type=bool, default=True)
parser.add_argument('--num_experiment', type=int, default=3, help="the number of experiments")

parser.add_argument('--submodels', type=str, default='000-012-553-665-777')
# parser.add_argument('--decay', type=bool, default=False)
parser.add_argument('--mode', type=str, default='normal')
parser.add_argument('--name', type=str, default='[no-name-LB]')
parser.add_argument('--rs', type=int, default=0)

parser.add_argument('--save', type=str, default='/data/dv_fl_1206/')
args = parser.parse_args()

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')


def embed_param(w_glob, BN): # BN layer은 해당 p에 저장된 것을 가져옴
    w_sub = copy.deepcopy(w_glob)
    # len(w_glob['bn1.weight'].shape) = 1. weight/bias/running_mean/running_var
    # len(w_glob['bn1.num_batches_tracked'].shape) = 0
    # len(w_glob['linear.bias'].shape) = 1
    for key in w_glob.keys():
        if len(w_glob[key].shape)<=1 and key!='linear.bias':
            w_sub[key] = BN[key]
    return w_sub

def main(logger):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    fileName = args.save + str(args.submodels) + '/resnet56_cifar_' + timestamp + '_' + str(args.rs)

    if not os.path.exists(fileName):
        os.makedirs(fileName)

    if args.wandb:
        run = wandb.init(dir=fileName, project='DVBN-FL-R56-1206', name= str(args.name)+ str(args.rs), reinit=True)
        wandb.config.update(args)

    logger = get_logger(logpath=os.path.join(fileName, 'logs'), filepath=os.path.abspath(__file__))
    logger.info(args)
    
    if args.model_name == 'resnet32':
        blocks = [5, 5, 5]
    elif args.model_name == 'resnet56':
        blocks = [9, 9, 9]
    elif args.model_name == 'resnet110':
        blocks = [18, 18, 18]
    full_stepSize = []
    for i in range(3):
        full_stepSize.append([1 for _ in range(blocks[i])])

    if args.model_name == 'resnet32':
        model = resnet32(full_stepSize).to(device) # global model
    elif args.model_name == 'resnet56':
        model = resnet56(full_stepSize).to(device) # global model
    elif args.model_name == 'resnet110':
        model = resnet110(full_stepSize).to(device) # global model
    # torchsummary.summary(model, (3, 32, 32))


    model_modes = args.submodels.split('-')
    s2D = []
    if args.model_set == 2:
        s2D = [
            [ [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]] ],
            [ [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 2, 0, 1, 1, 1], [1, 2, 0, 2, 0, 1, 1, 1, 1]] ],
            [ [[1, 1, 6, 0, 0, 0, 0, 0, 1], [1, 4, 0, 0, 0, 3, 0, 0, 1], [1, 1, 2, 0, 2, 0, 2, 0, 1]] ],
            [ [[1, 6, 0, 0, 0, 0, 0, 2, 0], [1, 5, 0, 0, 0, 0, 3, 0, 0], [1, 5, 0, 0, 0, 0, 2, 0, 1]] ],
            [ [[1, 8, 0, 0, 0, 0, 0, 0, 0], [1, 8, 0, 0, 0, 0, 0, 0, 0], [1, 8, 0, 0, 0, 0, 0, 0, 0]] ]
            ]
    else:
        for i in range(len(model_modes)):
            s2D.append(get_models(model_modes[i], args))
            '''
            if TRUE dynamic, several submodels for a fixed submodel rate
            s2D[0][0]
            s2D[1] has 4 models s2D[1][0~3]
            '''
    
    m = max(int(args.frac * args.num_users), 1)
    
    submodel_num = len(s2D)
    local_models = [[] for _ in range(submodel_num)]
    for i in range(submodel_num):
        print(i)
        for j in range(len(s2D[i])):
            if args.model_name == 'resnet32':
                print(j, end=" ")
                local_models[i].append(resnet32(s2D[i][j]).to(device))
            elif args.model_name == 'resnet56':
                print(j, end=" ")
                local_models[i].append(resnet56(s2D[i][j]).to(device))
            elif args.model_name == 'resnet110':
                print(j, end=" ")
                local_models[i].append(resnet110(s2D[i][j]).to(device))
    
    BNs = []
    for i in range(submodel_num):
        BN = {}
        w = copy.deepcopy(local_models[i][0].state_dict())
        for key in w.keys():
            if len(w[key].shape)<=1 and key!='linear.bias':
                BN[key] = w[key]
        BNs.append(copy.deepcopy(BN))

    w_glob = model.state_dict()

    com_layers = []  # common layers: conv1, bn1, linear
    sing_layers = []  # singular layers: layer1.0.~

    for i in w_glob.keys():
        if 'layer' in i:
            sing_layers.append(i)
        else:
            com_layers.append(i)
            
    dataset_train, dataset_test = get_fl_cifar_datasets()
    dict_users = cifar_iid(dataset_train, args.num_users, args.rs)
       
    lr = args.lr
    
    for itr in range(1, args.nrounds+1): # communication round iteratin
        
        model.train()
        
        if itr==args.nrounds/2:
            lr = lr*0.1
            print(lr)
        elif itr==3*args.nrounds/4:
            lr = lr*0.1
            print(lr)

        idxs_users = np.random.choice(range((args.model_cluster_idx)*20, (args.model_cluster_idx+1)*20), 2, replace=False) # range(0,3): 0~2
        loss_locals = []
       
        w_locals = []
        BN_locals = [[] for _ in range(submodel_num)]

        num_submodel = [0]*submodel_num  # num_submodel: number for each submodels at this round 
        num_submodel_sub = [0]*submodel_num
        subs2D = []
        selected_rate_and_idx = []

        for idx in idxs_users:
            tc = args.model_cluster_idx
            c = tc
            num_submodel_sub[c] += 1
            num_submodel[tc] += 1 # maximum model
            # print(idx, c)
            model_choice = int(np.random.choice(range(len(s2D[c])), 1)) # dynamic, if non-dynamic, only one models are loaded
            ss = s2D[c][model_choice]
            # print(c, model_choice)
                            
            subs2D.append(ss)
            local_models[c][model_choice].load_state_dict(embed_param(w_glob, BNs[c]))
            local_models[c][model_choice].train()
            
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                
            w, BN_s, loss = local.train(net=local_models[c][model_choice], learning_rate=lr)
            w_locals.append(copy.deepcopy(w))
            
            BN_locals[c].append(copy.deepcopy(BN_s))
            loss_locals.append(copy.deepcopy(loss))

        # print(loss_locals)
        print(num_submodel)
        
        w_glob = DVAvg(w_glob, w_locals, com_layers, sing_layers, subs2D)
        for ii in range(submodel_num):
            if len(BN_locals[ii]) > 0:
                BNs[ii] = BNAvg(BN_locals[ii])

        model.load_state_dict(w_glob)

        loss_avg = sum(loss_locals) / len(loss_locals)
        print("round:{}, loss: {}".format(itr, loss_avg))

        if itr % 10 == 0:
            local_models[c][0].load_state_dict(embed_param(w_glob, BNs[c]))
            local_acc_test, local_loss_test = test_img(local_models[c][0], dataset_test, args)
            logger.info("{:04d}".format(itr))
            logger.info("G {:.4f}".format(local_acc_test))
            if args.wandb:
                wandb.log({
                    "Communication round": itr,
                    "A model test accuracy": local_acc_test
                })
        
    modelName = str(itr) + '.pth'
    torch.save({'state_dict': model.state_dict(), 'args': args, 'bns': BNs}, os.path.join(fileName, modelName))
    print(fileName)


    w_local_glob = copy.deepcopy(w_glob)
    
    for itrf in range(args.nftrounds):  # fine-tuning rounds

        idxs_users = np.random.choice(range((args.model_cluster_idx)*20, (args.model_cluster_idx+1)*20), (args.model_cluster_idx+1)*2, replace=False)
        ###########################        
        subs2D = []
        w_locals = []
        BN_locals = [[] for _ in range(submodel_num)]

        for idx in idxs_users:
            # print(idx, c)
            # model_choice = int(np.random.choice(range(len(s2D[i])), 1)) # dynamic, if non-dynamic, only one models are loaded
            model_choice = 0
            s = s2D[c][model_choice]
            
        # print(c, model_choice)
            subs2D.append(s)

            # local_models[c][model_choice].load_state_dict(w_glob)
            local_models[c][model_choice].train()
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            
            w, BN_s, loss = local.train(net=local_models[c][model_choice], learning_rate=lr)
            w_locals.append(copy.deepcopy(w))
            BN_locals[c].append(BN_s)
            loss_locals.append(copy.deepcopy(loss))

        
        w_local_glob = DVAvg(w_local_glob, w_locals, com_layers, sing_layers, subs2D)
        for ii in range(submodel_num):
            if len(BN_locals[ii]) > 0:
                BNs[ii] = BNAvg(BN_locals[ii])
        local_models[c][model_choice].load_state_dict(embed_param(w_local_glob, BNs[c]))
        
        if (itrf+1) % 5 == 0:
            local_acc_test, local_loss_test = test_img(local_models[c][model_choice], dataset_test, args)
            logger.info(" | F-A{} {:.4f}".format(c, local_acc_test))
            if args.wandb:
                wandb.log({
                        "Communication round": itr+itrf+1,
                        "A model (F) " + str(c) + "-" + str(0) + " test accuracy": local_acc_test
                    })
    
    if args.wandb:
        run.finish()
    
    return

if __name__ == "__main__":
    for i in range(args.num_experiment):
        loggers = [0 for _ in range(args.num_experiment)]
        torch.manual_seed(args.rs)
        torch.cuda.manual_seed(args.rs)
        torch.cuda.manual_seed_all(args.rs) # if use multi-GPU
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        np.random.seed(args.rs)
        random.seed(args.rs)
        main(loggers[i])
        args.rs = args.rs+1