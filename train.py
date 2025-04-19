import argparse
import pickle
import os
import time
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from backbones.TransGTE import TransGTE
from utils.haversine_distance import haversine
import numpy as np
from utils.MyDataset import Trip_parDataset
from utils.adjmatrix_gen import GenAdjmatrix
import torch.nn as nn
import torch.nn.functional as F

def evaluation(pred_y, true_y):
    totT = 0
    for j in range(0, len(pred_y)):
        dist_cc = haversine(pred_y[j][0], pred_y[j][1], true_y[j][0], true_y[j][1])
        totT = totT + dist_cc
    return totT

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_epoch(epoch, optimizer):
    start_time = time.time()
    loss_train = 0.0
    steps = 0.0

    with tqdm(total=len(trainloader), desc=f'Epoch {epoch + 1} / {EPOCHES}',
              postfix=dict, mininterval=0.3) as pbar:
        for step, (batch_x, batch_y) in enumerate(trainloader):
            # batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            train_pred_y = model(batch_x)
            pred_y = torch.matmul(train_pred_y, grid_coord)
            loss_step = criterion(pred_y, batch_y)
            optimizer.zero_grad()
            loss_step.backward()
            # torch.nn.utils.clip_grad_norm(model.parameters(), 5.)
            optimizer.step()

            loss_train += loss_step.item()
            steps += 1.0
            running_time = time.time() - start_time
            pbar.set_postfix(**{'total_loss'   : loss_train / (step + 1),
                                'lr'           : get_lr(optimizer),
                                'step/s'       : running_time})
            pbar.update(1)
            if step + 1 == len(trainloader):
                val(epoch, step + 1, len(trainloader))
            start_time = time.time()

def val(epoch, train_step, train_steps):
    global model_name, patience
    start_time = time.time()
    samples_sum = 0.0
    pred_y_list = []
    true_y_list = []
    with torch.no_grad():
        with tqdm(total=len(valloader), desc=f'Epoch {epoch + 1} / {EPOCHES}',
                  mininterval=0.3) as pbar:
            for step, (batch_x, batch_y) in enumerate(valloader):
                test_pred_y = model(batch_x)
                pred_y = torch.matmul(test_pred_y, grid_coord)
                pred_y_list.append(pred_y)
                true_y_list.append(batch_y)
                samples_sum += len(batch_y)
                pbar.update(1)
    pred_y = torch.cat(pred_y_list, dim=0).to('cpu')
    true_y = torch.cat(true_y_list, dim=0).to('cpu')

    val_mse = criterion(pred_y, true_y).item()

    end_time = time.time()
    if train_step == train_steps:
        print('**********************************************')
        print('Val epoch {:d} finished. Time: {:2f}\n'
              'mse: {:8f}\n'.format(epoch+1, end_time-start_time, val_mse))
        print('**********************************************')
    if is_saveModel:
        record = 'epoch' + str(epoch+1) + 'val_mse ' + str(val_mse)
        global min_val_mse, best_model_path
        if min_val_mse > val_mse:
            min_val_mse = val_mse
            best_model_path = model_save(record)
            patience = 0
        else:
            patience += 1
        record_tofile(record)

def test():
    global model_name, best_model_path
    #model.eval()
    print(best_model_path)
    model.load_state_dict(torch.load(best_model_path))
    start_time = time.time()
    # totT_count = 0.0
    samples_sum = 0.0

    pred_y_list = []
    true_y_list = []
    with torch.no_grad():
        for step, (batch_x, batch_y) in enumerate(testloader):
            test_pred_y = model(batch_x)
            pred_y = torch.matmul(test_pred_y, grid_coord)
            pred_y_list.append(pred_y)
            true_y_list.append(batch_y)
            samples_sum += len(batch_y)
            print('\rTesting, batch: {:d}/{:d}'.format(step + 1, len(testloader)), end='')

    pred_y = torch.cat(pred_y_list, dim=0).to('cpu')
    # pred_y = torch.tensor(grid_coord_scalar.inverse_transform(np.array(pred_y)))

    true_y = torch.cat(true_y_list, dim=0).to('cpu')
    # true_y = torch.tensor(grid_coord_scalar.inverse_transform(np.array(true_y)))


    totT_count = evaluation(pred_y, true_y)
    print()
    hs = totT_count / samples_sum
    end_time = time.time()
    print('Test finished. Time: {:2f}\n'
          'haversine: {:8f}\n'.format(end_time-start_time, hs))
    record = 'test_haversine ' + str(hs)
    record_tofile(record)

def model_save(saveName):
    rootpath = 'modelsave/{}_g{}/'.format(args.data, args.g)
    global date, best_model_path
    if os.path.exists(rootpath + date + '_' + model_name) is False:
        os.makedirs(rootpath + date + '_' + model_name)
    # best_model_path = rootpath + date + '_' + model_name + '/' + saveName + '.pkl'
    best_model_path = rootpath + date + '_' + model_name + '/' + 'best_val_model.pkl'
    torch.save(model.state_dict(), best_model_path)
    return best_model_path

def record_tofile(str):
    save_dir = 'modelsave/{}_g{}/'.format(args.data, args.g)
    dirpath = save_dir + '/' + date + '_' + model_name + '/'
    global record_file
    if record_file is None:
        if os.path.isfile(dirpath + 'train_record.txt'):
            record_file = open(dirpath + 'train_record.txt', 'a', encoding='utf-8')
        else:
            record_file = open(dirpath + 'train_record.txt', 'w', encoding='utf-8')
    record_file.write(str + '\n')

class Criterion(nn.Module):
    def __init__(self):
        super(Criterion, self).__init__()
        # self.loss = nn.CrossEntropyLoss()
        self.loss = nn.MSELoss()
        # self.centro = centro

    def forward(self, pred, true):
        return self.loss(pred, true)

def collate_fn(batch):
    # pad_index = -1(0)
    """
    args:
        batch: [[input_vector, label_vector] for seq in batch]

    return:
        [[output_vector]] * batch_size, [[label]]*batch_szie
    """

    percentile = 100
    dynamical_pad = True
    max_len = 2000
    # pad_index = 0

    lens = [len(dat[0][0]) for dat in batch]

    # find the max len in each batch
    if dynamical_pad:
        # dynamical padding
        seq_len = min(int(np.percentile(lens, percentile)), max_len)
        # or seq_len = max(lens)
    else:
        # fixed length padding
        seq_len = max_len

    out_gtrj = []
    out_trj = []
    out_dest = []
    out_w = []
    out_h = []
    seq_index = []
    mask_input = []
    for dat in batch:
        x, y = dat
        gtrj, trj, w, h = x

        seq_i = len(gtrj)
        seq_index.append(seq_i)
        # padding = np.array([pad_index for _ in range(seq_len - seq_i)])
        gtrj_padding = np.array([0 for _ in range(seq_len - seq_i)])
        trj_padding = np.array([trj[-1] for _ in range(seq_len - seq_i)])
        mask_input.append(np.concatenate([np.ones(seq_i), gtrj_padding], axis=0))
        # gtrj = [generalID(trj[i][0], trj[i][1], args.g, args.g) for i in range(len(trj))]
        if seq_i != seq_len:
            gtrj = np.array(gtrj) + 1
            gtrj = np.concatenate([gtrj[:-1], gtrj_padding, gtrj[-1:]], axis=0)
            trj = np.concatenate([np.array(trj), trj_padding], axis=0)
        out_gtrj.append(gtrj)
        out_trj.append(trj)
        out_dest.append(y)
        out_w.append(w)
        out_h.append(h)

    mask_input = torch.tensor(mask_input, dtype=torch.bool, device=device)
    mask_input = mask_input == False
    out_gtrj = torch.tensor(out_gtrj, dtype=torch.long, device=device)
    out_trj = torch.tensor(out_trj, dtype=torch.float32, device=device)
    out_w = torch.tensor(out_w, dtype=torch.long, device=device)
    out_h = torch.tensor(out_h, dtype=torch.long, device=device)

    out_dest = np.array(out_dest)
    # out_dest = grid_coord_scalar.transform(out_dest)
    out_dest = torch.tensor(out_dest, dtype=torch.float32, device=device)

    h1_grid = F.one_hot(out_gtrj, num_classes=args.g ** 2 + 1).float()

    in_gcn = h1_grid.sum(dim=1).unsqueeze(-1)[:, 1:]

    return (out_gtrj, out_trj, h1_grid, in_gcn, out_w, out_h, mask_input), out_dest

def parse_args(argv=None):
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    parser = argparse.ArgumentParser(
        description='DeepMove Model Train and Test')
    parser.add_argument('--pr', default=0.1, type=float,
                        help='The ratio of the partial sequence in the whole')
    parser.add_argument('--data', default='sanfran', type=str,
                        help='Choose the dataset')
    parser.add_argument('--device', default='cuda:0', type=str,
                        help='Choose device')
    parser.add_argument('--g', default=60, type=int,
                        help='Choose grid granularity')
    parser.add_argument('--K', default=2, type=int,
                        help='Choose GCN layers')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Choose batch size')
    global args
    args = parser.parse_args(argv)

def process_graph(graph_data):
    N = graph_data.size(0)
    matrix_i = torch.eye(N, dtype=graph_data.dtype, device=args.device)
    graph_data += matrix_i  # A~ [N, N]

    degree_matrix = torch.sum(graph_data, dim=-1, keepdim=False)  # [N]
    degree_matrix = degree_matrix.pow(-1)
    degree_matrix[degree_matrix == float("inf")] = 0.  # [N]

    degree_matrix = torch.diag(degree_matrix)  # [N, N]

    return torch.mm(degree_matrix, graph_data)

if __name__ == '__main__':
    global args
    parse_args()

    device = args.device
    is_saveModel = True
    batch_size = args.batch_size
    PATIENCE = 10
    EPOCHES = 250
    patience = 0
    min_val_mse = 99999.0

    print('partial ratio:{}'.format(args.pr))
    if args.data == 'porto':
        path1 = 'data/porto/porto_municipality_dataset_r{}.pkl'.format(args.pr)
        path2 = 'data/porto/grid_coord_center_g{}.pkl'.format(args.g) # g = 50
    if args.data == 'chengdu':
        path1 = 'data/chengdu/chengdu_dataset_r{}.pkl'.format(args.pr)
        path2 = 'data/chengdu/grid_coord_center_g{}.pkl'.format(args.g) # g = 70
    if args.data == 'shenzhen':
        path1 = 'data/shenzhen/shenzhen_dataset_r{}.pkl'.format(args.pr)
        path2 = 'data/shenzhen/grid_coord_center_g{}.pkl'.format(args.g) # g = 70
    if args.data == 'sanfran':
        path1 = 'data/sanfran/sanfran_dataset_r{}.pkl'.format(args.pr)
        path2 = 'data/sanfran/grid_coord_center_g{}.pkl'.format(args.g) # g = 60

    with open(path1, 'rb') as fr:
        dataset = pickle.load(fr)
    with open(path2, 'rb') as fr:
        grid_coord = pickle.load(fr)

    grid_coord = torch.tensor(grid_coord, dtype=torch.float32).to(device)
    print(int(len(dataset['trainset']['dest'])))
    print(int(len(dataset['valset']['dest'])))
    print(int(len(dataset['testset']['dest'])))

    trainset = Trip_parDataset(dataset['trainset'])
    valset = Trip_parDataset(dataset['valset'])
    testset = Trip_parDataset(dataset['testset'])

    del dataset

    trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True,
                             collate_fn=collate_fn)
    valloader = DataLoader(dataset=valset, batch_size=batch_size, collate_fn=collate_fn)
    testloader = DataLoader(dataset=testset, batch_size=batch_size, collate_fn=collate_fn)

    graph_data = torch.tensor(GenAdjmatrix(args.g, args.g), dtype=torch.float, device=device)
    graph_data = process_graph(graph_data)

    model = TransGTE(model_dim=256, enc_nhead=8, dec_nhead=8, enc_layers=1, dec_layers=1,
                     gcn_hid_dim=4, matrix_adjacent=graph_data, g=args.g,
                     n_head=1, K=args.K, GCNtype='GCN', device=device).to(device)


    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4, eps=1e-9,
                                 betas=(0.9, 0.98))
    best_model_path = ''
    date = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
    model_name = date + ' TransGTE_test4_r{}_b{}_g{}'.format(args.pr, args.batch_size, args.g,
                                                               )
    record_file = None
    criterion = Criterion()

    train_time_start = time.time()
    for epoch in range(EPOCHES):
        if patience >= PATIENCE:
            patience = 0
            break
        print('----------------------------------------------')
        fit_one_epoch(epoch, optimizer)
        print('----------------------------------------------')
    print('Training finished!')
    # train_str = f'Training time is {round((time.time() - train_time_start)/3600, 2)} hour'
    train_str = f'Training time is {round((time.time() - train_time_start), 2)} second'
    print(train_str)

    record_file.write(train_str + '\n')
    test()

