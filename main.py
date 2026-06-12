import random
import numpy as np
import torch
torch.backends.cudnn.deterministic = True

from time import time
from parse import args
from data import load_dataset
from client import FedRecClient
#from client_ldp import FedRecClient, FedRecClientDefense
#from server_defense import FedRecServer
from server import FedRecServer
from attack import AttackClient, BaselineAttackClient, OursAttackClient
#from attack_ldp import AttackClient, BaselineAttackClient, OursAttackClient
from group import FindGroup
import csv
import os
#import wandb



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    
    args_str = ",".join([("%s=%s" % (k, v)) for k, v in args.__dict__.items()])
    print("Arguments: %s " % args_str)
    
    # Create results directory if it doesn't exist
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Create CSV file for storing results
    results_file = os.path.join(results_dir, f"test_results_{args.dataset}_{args.clients_limit}_{args.attack}_{args.mode}.csv")
    with open(results_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'loss', 'all_test_hr10', 'all_test_hr20', 'all_test_ndcg10', 'all_test_ndcg20',
                        'all_target_er5', 'all_target_er10', 'all_target_er20', 'all_target_er30',
                        'group_test_hr10', 'group_test_hr20', 'group_test_ndcg10', 'group_test_ndcg20',
                        'group_target_er5', 'group_target_er10', 'group_target_er20', 'group_target_er30',
                        'other_test_hr10', 'other_test_hr20', 'other_test_ndcg10', 'other_test_ndcg20',
                        'other_target_er5', 'other_target_er10', 'other_target_er20', 'other_target_er30'])
        
    
    analyzer = FindGroup(args.path + args.dataset+"/train.dat", threshold_min=args.interested_min_freq, threshold_max=args.interested_max_freq, n=args.num_interested_items)
    if(args.dataset != 'AZ'):
        interested_items_set, target_user_set = analyzer.analyze()#返回挑选出的interested items和共同交互的users
    else:
        from simulation import sample_items_and_users
        interested_items_set, target_user_set = sample_items_and_users()
    

    t0 = time()
    m_item, all_train_ind, all_test_ind, items_popularity = load_dataset(args.path + args.dataset)
    min_count = items_popularity.min()
    min_items = np.where(items_popularity == min_count)[0]
    print("********************")
    print(f"最低流行度item id: {min_items.tolist()}, 频数: {int(min_count)}")
    print("********************")


    
    #target items就一个（最小流行度的）
    _, target_items = torch.Tensor(-items_popularity).topk(1)# Select the least popular item as the target item
    target_items = target_items.tolist()
    interested_items = list(interested_items_set)  
    print("target_items", target_items)
    print("********************")

    # print(f"all user number: {len(all_train_ind)}")
    # print(f"all item number: {m_item}")
    # print(f"target items: {target_items}")

    server = FedRecServer(m_item, args.dim, eval(args.layers)).to(args.device)

    clients = []
    benign_clients = []
    clients_group = []
    other_clients = []

    
    for idx, (train_ind, test_ind) in enumerate(zip(all_train_ind, all_test_ind)):
        if args.defense != 'Pieck':
            benign_clients.append(
                FedRecClient(train_ind, test_ind, target_items, m_item, args.dim, idx).to(args.device)
            )
        else:
            benign_clients.append(
                FedRecClientDefense(train_ind, test_ind, target_items, m_item, args.dim, idx).to(args.device)
            )
        # 判断用户是否交互了interested_items中的所有物品
        if idx in target_user_set: # 如果用户交互的物品包含所有interested_items中的物品
            # 创建符合条件的FedRecClient并加入到列表中
            if args.defense != 'Pieck':
                clients_group.append(
                    FedRecClient(train_ind, test_ind, target_items, m_item, args.dim, idx).to(args.device)
                )
            else:
                clients_group.append(
                    FedRecClientDefense(train_ind, test_ind, target_items, m_item, args.dim, idx).to(args.device)
                )
        else:
            if args.defense != 'Pieck':
                other_clients.append(
                    FedRecClient(train_ind, test_ind, target_items, m_item, args.dim, idx).to(args.device)
                )
            else:
                other_clients.append(
                    FedRecClientDefense(train_ind, test_ind, target_items, m_item, args.dim, idx).to(args.device)
                )
    clients.extend(benign_clients)

    malicious_clients_limit = int(len(clients) * args.clients_limit)



    if args.attack == 'A-ra' or args.attack == 'A-hum' or args.attack == 'test':
        for idx, _ in enumerate(range(malicious_clients_limit)):
            clients.append(AttackClient(target_items, m_item, args.dim, client_id=idx).to(args.device))
    elif args.attack == 'EB':
        for idx, _ in enumerate(range(malicious_clients_limit)):
            clients.append(BaselineAttackClient(target_items, m_item, args.dim, client_id=idx).to(args.device))#baselineattackclient和benign_clients一样的训练过程
    elif args.attack == 'RA':
        for idx, _ in enumerate(range(malicious_clients_limit)):
            train_ind = [i for i in target_items]
            for __ in range(args.items_limit - len(target_items)):
                item = np.random.randint(m_item)
                while item in train_ind:
                    item = np.random.randint(m_item)
                train_ind.append(item)
            clients.append(BaselineAttackClient(train_ind, m_item, args.dim, client_id=idx).to(args.device))#baselineattackclient和benign_clients一样的训练过程
    elif args.attack == 'Base_GA' or args.attack == 'Base_GA_no_target' or args.attack == 'Base_GA_no_target_negative' or args.attack == 'GA_dual_user':#base group attack
        for idx, _ in enumerate(range(malicious_clients_limit)):
            clients.append(AttackClient(target_items, m_item, args.dim, interested_items=interested_items, client_id=idx).to(args.device))
    
    elif args.attack == 'ours': #ours attack
        for idx, _ in enumerate(range(malicious_clients_limit)):
            clients.append(OursAttackClient(target_items, m_item, args.dim, interested_items=interested_items, client_id=idx, items_popularity=items_popularity).to(args.device))
    else:
        import sys
        print("invalid attack!!")
        #sys.exit(1)


    print("Load data done [%.1f s].\n #all user=%d,#benign user=%d, #malicious users=%d, #item=%d, #train=%d, #avg_train_items=%d,  #test=%d" %
          (time() - t0, len(clients), len(benign_clients), malicious_clients_limit, m_item,
           sum([len(i) for i in all_train_ind]),#所有交互个数
           sum([len(i) for i in all_train_ind])/len(benign_clients),#
           sum([len(i) for i in all_test_ind])))
    print("Target items: %s." % str(target_items))
    print("output format: ({HR@10, HR@20, NDCG@10, NDCG@20}), ({ER@5, ER@10, ER@20, ER@30})")
    print("************************************************")

    # Init performance
    t1 = time()
    test_result, target_result= server.eval_(benign_clients)
    group_test_result, group_target_result = server.eval_(clients_group)
    other_test_result, other_target_result = server.eval_(other_clients)
    
    # Save initial results (epoch 0)
    with open(results_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([0, 0.0] + list(test_result) + list(target_result) + 
                       list(group_test_result) + list(group_target_result) + 
                       list(other_test_result) + list(other_target_result))
    
    print("Iteration 0(init), \n"+
          "all_users on test: (%.7f, %.7f, %.7f, %.7f),\n" % tuple(test_result) +
          "all_users on target: (%.7f, %.7f, %.7f, %.7f).\n" % tuple(target_result) +
          "group_users on test: (%.7f, %.7f, %.7f, %.7f),\n" % tuple(group_test_result) +
          "group_users on target: (%.7f, %.7f, %.7f, %.7f).\n" % tuple(group_target_result) +
          "other_users on test: (%.7f, %.7f, %.7f, %.7f),\n" % tuple(other_test_result) +
          " other_users on target: (%.7f, %.7f, %.7f, %.7f).\n" % tuple(other_target_result) +
          "eval time: [%.1fs]" % (time() - t1))

    try:
        for epoch in range(1, args.epochs + 1):
            t1 = time()
            rand_clients = np.arange(len(clients))
            np.random.shuffle(rand_clients)

            total_loss = []
            for i in range(0, len(rand_clients), args.batch_size):#每次顺序取batch_size个client
                batch_clients_idx = rand_clients[i: i + args.batch_size]
                loss = server.train_(clients, batch_clients_idx)#
                total_loss.extend(loss)
            total_loss = np.mean(total_loss).item()

            t2 = time()
            test_result, target_result= server.eval_(benign_clients)
            group_test_result,  group_target_result = server.eval_(clients_group)
            other_test_result, other_target_result = server.eval_(other_clients)

            # Save results for current epoch
            with open(results_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, total_loss] + list(test_result) + list(target_result) + 
                              list(group_test_result) + list(group_target_result) + 
                              list(other_test_result) + list(other_target_result))

            print("Iteration %d, loss = %.5f, training time: [%.1fs] \n" % (epoch, total_loss, t2 - t1)+
          "all_users on test: (%.7f, %.7f, %.7f, %.7f),\n" % tuple(test_result) +
          "all_users on target: (%.7f, %.7f, %.7f, %.7f).\n" % tuple(target_result) +
          "group_users on test: (%.7f, %.7f, %.7f, %.7f),\n" % tuple(group_test_result) +
          "group_users on target: (%.7f, %.7f, %.7f, %.7f).\n" % tuple(group_target_result) +
          "other_users on test: (%.7f, %.7f, %.7f, %.7f),\n" % tuple(other_test_result) +
          " other_users on target: (%.7f, %.7f, %.7f, %.7f).\n" % tuple(other_target_result) +
          "eval time: [%.1fs]" % (time() - t2))

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    # Set random seed at the start of main
    setup_seed(20220110)
    main()
    #wandb.finish()