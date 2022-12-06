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
parser.add_argument('--flex_num', type=int, default=5, help="0:0~4 min(tc+args.flex_num,5)")

parser.add_argument('--model_set', type=int, default=2)
parser.add_argument('--model_name', type=str, default='resnet56') # 34, 56, 110
parser.add_argument('--wandb', type=bool, default=True)
parser.add_argument('--num_experiment', type=int, default=3, help="the number of experiments")

parser.add_argument('--submodels', type=str, default='000-012-553-665-777')
# parser.add_argument('--decay', type=bool, default=False)
parser.add_argument('--mode', type=str, default='normal')
parser.add_argument('--name', type=str, default='[no-name]')
parser.add_argument('--rs', type=int, default=0)

parser.add_argument('--save', type=str, default='/data/dv_fl_1125/')
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

def main():
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    fileName = args.save + str(args.submodels) + '/resnet56_cifar_' + timestamp + '_' + str(args.rs)
    
    if not os.path.exists(fileName):
        os.makedirs(fileName)        

    if args.wandb:
        run = wandb.init(dir=fileName, project='DVBN-FL-R56-1206', name= str(args.name)+ str(args.rs), reinit=True)
        wandb.config.update(args)
    
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
        BNs.append(BN)

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
        
    logger = get_logger(logpath=os.path.join(fileName, 'logs'), filepath=os.path.abspath(__file__))
    logger.info(args)
    lr = args.lr
    
    for itr in range(1, args.nrounds+1): # communication round iteratin
        
        model.train()
        
        if itr==args.nrounds/2:
            lr = lr*0.1
            print(lr)
        elif itr==3*args.nrounds/4:
            lr = lr*0.1
            print(lr)

        if args.worst_three:
            idxs_users = np.random.choice(range(40,args.num_users), 6,replace=False)
        else:
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        loss_locals = []
       
        w_locals = []

        num_submodel = [0]*submodel_num  # num_submodel: number for each submodels at this round 
        num_submodel_sub = [0]*submodel_num
        subs2D = []
        selected_rate_and_idx = []

        for idx in idxs_users:
            if args.mode == 'worst':
                c = -1
                model_choice = int(np.random.choice(range(len(s2D[c])), 1))
                ss = s2D[c][model_choice]
                num_submodel[c] += 1
            elif args.mode == 'best':
                c = 0
                model_choice = int(np.random.choice(range(len(s2D[c])), 1))
                ss = s2D[c][model_choice]
                num_submodel[c] += 1
            else: # normal
                if args.full_condition:
                    tc = 0
                else:
                    tc = idx//(args.num_users//submodel_num)
                '''
                tc: 0,1,2,3,4 (maximum performance of a user)
                tc = 0 # every device has full condition
                c: selected submodel among executable models
                '''
                if args.cluster_select:
                    '''
                    a user selects model that is executable (min ~ executable)
                    submodel 0 can execute all submodels (0 ~ 4)
                    0: 0,1,2 / 1: 1,2,3 / 2: 2,3,4 / 3: 3,4 / 4: 4
                    '''
                    c = random.choice(list(range(tc,5)))
                    if args.limit_num:
                        c = random.choice(list(range(tc,min(tc+args.flex_num,5))))
                    # c = random.choice(list(range(max(0,tc-2),tc+1)))
                    # c = random.choice(list(range(max(2,tc-2),tc+1)))
                else:
                    '''
                    a user executes its largest model that is executable
                    '''
                    c = tc
                num_submodel_sub[c] += 1
                num_submodel[tc] += 1 # maximum model
                # print(idx, c)
                model_choice = int(np.random.choice(range(len(s2D[c])), 1)) # dynamic, if non-dynamic, only one models are loaded
                ss = s2D[c][model_choice]
                # print(c, model_choice)

                if args.KD and c != tc and itr>3*args.nrounds/4:
                    model_choice_t = int(np.random.choice(range(len(s2D[tc])), 1)) # dynamic, if non-dynamic, only one models are loaded
                    st = s2D[tc][model_choice_t]
                    local_models[tc][model_choice_t].load_state_dict(embed_param(w_glob, BNs[tc]))
                    # local_models[tc][model_choice_t].train()
                    # local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                    # w_t, BN_t, loss_t = local.train(net=local_models[tc][model_choice_t], learning_rate=lr)
                    # local_models[tc][model_choice_t].load_state_dict(w_t)
                            
            subs2D.append(ss)
            local_models[c][model_choice].load_state_dict(embed_param(w_glob, BNs[c]))
            local_models[c][model_choice].train()
            
            if args.KD and c != tc  and itr>3*args.nrounds/4:
                local = LocalUpdateKD(args=args, dataset=dataset_train, idxs=dict_users[idx])
                w_t, w_s, BN_s, loss = local.train(tnet = local_models[tc][model_choice_t], snet=local_models[c][model_choice], learning_rate=lr)
                # w_locals.append(copy.deepcopy(w_t))
                w_locals.append(copy.deepcopy(w_s))
            else:
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                w, BN_s, loss = local.train(net=local_models[c][model_choice], learning_rate=lr)
                w_locals.append(copy.deepcopy(w))
            
            BNs[c] = BN_s
            loss_locals.append(copy.deepcopy(loss))

        # print(loss_locals)
        print(num_submodel)
        if 'normal' in args.mode:
            print(num_submodel_sub)
        
        w_glob = DVAvg(w_glob, w_locals, com_layers, sing_layers, subs2D)
        model.load_state_dict(w_glob)

                
        loss_avg = sum(loss_locals) / len(loss_locals)
        print("round:{}, loss: {}".format(itr, loss_avg))

        if itr % 10 == 0:
            itrl = int(itr/(5/args.local_ep))
            local_acc_test, local_loss_test = [[] for _ in range(submodel_num)], [[] for _ in range(submodel_num)]
            if args.mode == 'worst':
                local_models[-1][0].load_state_dict(w_glob)
                local_acc_test, local_loss_test = test_img(local_models[-1][0], dataset_test, args)
                logger.info("{:04d}".format(itr))
                logger.info("G {:.4f}".format(local_acc_test))
                if args.wandb:
                    wandb.log({
                        "Communication round": itr,
                        "Global model test accuracy": local_acc_test
                    })
            elif args.mode == 'best':
                local_models[0][0].load_state_dict(w_glob)
                local_acc_test, local_loss_test = test_img(local_models[0][0], dataset_test, args)
                logger.info("{:04d}".format(itr))
                logger.info("G {:.4f}".format(local_acc_test))
                if args.wandb:
                    wandb.log({
                        "Communication round": itr,
                        "Global model test accuracy": local_acc_test
                    })
            else:
                acc_test, loss_test_f = test_img(model, dataset_test, args)

                for i in range(submodel_num):
                    if args.log_all:
                        for j in range(len(s2D[i])):
                            local_models[i][j].load_state_dict(embed_param(w_glob, BNs[i]))
                            temp_acc_test, temp_loss_test = test_img(local_models[i][j], dataset_test, args)
                            local_acc_test[i].append(temp_acc_test)
                            local_loss_test[i].append(temp_loss_test)
                    else:
                        idxs_models = np.random.choice(range(len(s2D[i])), max(1,int(len(s2D[i])*args.log_rate)), replace=False)
                        for j in idxs_models:
                            local_models[i][j].load_state_dict(embed_param(w_glob, BNs[i]))
                            temp_acc_test, temp_loss_test = test_img(local_models[i][j], dataset_test, args)
                            local_acc_test[i].append(temp_acc_test)
                            local_loss_test[i].append(temp_loss_test)
            
                logger.info("{:04d}".format(itr))
                logger.info("G {:.4f}".format(acc_test))

                for i in range(submodel_num):
                    if args.log_all:
                        for j in range(len(s2D[i])):
                            logger.info("L{}-{} {:.4f}".format(i, j, local_acc_test[i][j]))
                    else:
                        for j in idxs_models:
                            logger.info("L{}-{} {:.4f}".format(i, j, local_acc_test[i][0]))
                if args.wandb:
                    wandb.log({
                        "Communication round": itr,
                        "Global model test accuracy": acc_test
                    })
                    for i in range(submodel_num):
                        if args.log_all:
                            for j in range(len(s2D[i])):
                                wandb.log({
                                    "Communication round": itr,
                                    "Local model " + str(i) + "-" + str(j) + " test accuracy": local_acc_test[i][j]
                                })
                        else:
                            for j in idxs_models:
                                wandb.log({
                                        "Communication round": itr,
                                        "Local model " + str(i) + "-" + str(0) + " test accuracy": local_acc_test[i][0]
                                    })

    modelName = str(itr) + '.pth'
    torch.save({'state_dict': model.state_dict(), 'args': args, 'bns': BNs}, os.path.join(fileName, modelName))
    print(fileName)

    if args.mode == 'worst':
        st = submodel_num-1 # 4
        en = st+1
    elif args.mode == 'best':
        st = 0
        en = st+1
    else:
        st = 0
        en = submodel_num
    
    if args.larg_lr:
        lr = 10*lr

    all_users = cifar_iid(dataset_train, 1, args.rs)

    for i in range(st, en):
        w_local_glob = copy.deepcopy(w_glob)
        
        for itrf in range(args.nftrounds):
            # if itrf == args.nftrounds/2:
            #     lrl = lrl*0.1
            if args.larg_lr:
                if itrf == args.nftrounds/2:
                    lr = 0.1*lr

            if args.mode == 'best':
                idxs_users = np.random.choice(range(args.num_users), m, replace=False)
            else:
                idxs_users = np.random.choice(range(0, (i+1)*20), (i+1)*2, replace=False) # range(0,3): 0~2
            subs2D = []
            w_locals = []
            for idx in idxs_users:
                # print(idx, c)
                # model_choice = int(np.random.choice(range(len(s2D[i])), 1)) # dynamic, if non-dynamic, only one models are loaded
                model_choice = 0
                s = s2D[i][model_choice]
                
            # print(c, model_choice)
                subs2D.append(s)

                # local_models[c][model_choice].load_state_dict(w_glob)
                local_models[i][model_choice].train()

                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                # local = LocalUpdate(args=args, dataset=dataset_train, idxs=all_users[0])
                w, BN_s, loss = local.train(net=local_models[i][model_choice], learning_rate=lr)
                w_locals.append(copy.deepcopy(w))
                loss_locals.append(copy.deepcopy(loss))

                BNs[c] = BN_s
            
            w_local_glob = DVAvg(w_local_glob, w_locals, com_layers, sing_layers, subs2D)
            local_models[i][model_choice].load_state_dict(embed_param(w_local_glob, BNs[i]))
            
            if (itrf+1) % 5 == 0:
                local_acc_test, local_loss_test = test_img(local_models[i][model_choice], dataset_test, args)
                logger.info(" | F-L{} {:.4f}".format(i, local_acc_test))
                if args.wandb:
                    wandb.log({
                            "Communication round": itr+itrf+1,
                            "Local model (F) " + str(i) + "-" + str(0) + " test accuracy": local_acc_test
                        })
    
        local_models[i][model_choice].train()
        lcl = LocalUpdate(args=args, dataset=dataset_train, idxs=all_users[0])
        w = lcl.sBN(local_models[i][model_choice])
        local_models[i][model_choice].load_state_dict(w)
        local_acc_test, local_loss_test = test_img(local_models[i][model_choice], dataset_test, args)
        logger.info(" | sBN-F-L{} {:.4f}".format(i, local_acc_test))
        if args.wandb:
            wandb.log({
                    "Communication round": itr+itrf+2,
                    "Local model (sBN-F) " + str(i) + "-" + str(0) + " test accuracy": local_acc_test
                })
        if args.wandb:
            run.finish()

    
    return

if __name__ == "__main__":
    for i in range(args.num_experiment):
        torch.manual_seed(args.rs)
        torch.cuda.manual_seed(args.rs)
        torch.cuda.manual_seed_all(args.rs) # if use multi-GPU
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        np.random.seed(args.rs)
        random.seed(args.rs)
        main()
        args.rs = args.rs+1

