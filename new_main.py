'''
Created on Oct 12, 2023
Pytorch Implementation of tempLGCN: Time-Aware Collaborative Filtering with Graph Convolutional Networks
'''
#%% 
from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
from torch import nn, optim
from utils import get_recall_at_k, minibatch, save_model, calculate_ndcg, plot_loss
import dataPrep as dp
from model import LGCN, RecSysGNN 
import time
from world import config

# Set a random seed
np.random.seed(42)
torch.manual_seed(42)

g_lr = config['lr']
g_epochs = config['epochs']
g_epochs_per_eval = config['epochs_per_eval']
g_epochs_per_lr_decay = config['epochs_per_lr_decay']
g_decay = config['decay']
g_num_layers = config['num_layers']
g_emb_dim = config['emb_dim']
g_batch_size = config['batch_size']
g_seed = config['seed']
g_top_k = config['top_k']
g_loadedModel = config['loadedModel']
g_r_beta = config['r_beta']
r_method = config['r_method']
g_a_beta = config['a_beta']
a_method = config['a_method']
g_dataset = config['dataset']
g_verbose = config['verbose']
g_model = config['model']
g_win = config['win']    
by_time = config['by_time']

print(f'loading {g_dataset} ...')

def run_experiment(rating_df, num_users, num_items, g_mean_rating, g_seed):

    rmat_data = dp.get_rmat_values(rating_df)

    split_ratio = 0.2

    rmat_train_data, rmat_val_data = dp.train_test_split_by_user2(rmat_data, test_size=split_ratio, seed=g_seed)

    edge_train_data = dp.rmat_2_adjmat(num_users, num_items, rmat_train_data)
    edge_val_data = dp.rmat_2_adjmat(num_users, num_items, rmat_val_data)

    r_mat_train_idx = rmat_train_data['rmat_index']
    r_mat_train_v = rmat_train_data['rmat_values']
    r_mat_train_rts = rmat_train_data['rmat_ts']
    r_mat_train_abs_t_decay = rmat_train_data['rmat_abs_t_decay']
    r_mat_train_rel_t_decay = rmat_train_data['rmat_rel_t_decay']
    
    r_mat_val_idx = rmat_val_data['rmat_index']
    r_mat_val_v = rmat_val_data['rmat_values']
    r_mat_val_rts = rmat_val_data['rmat_ts']
    r_mat_val_abs_t_decay = rmat_val_data['rmat_abs_t_decay']
    r_mat_val_rel_t_decay = rmat_val_data['rmat_rel_t_decay']
    
    #r_mat_train_idx, r_mat_train_v, r_mat_train_rts, r_mat_train_abs_t_decay, r_mat_train_rel_t_decay = dp.adjmat_2_rmat(num_users, num_items, adj_train_data)
    #r_mat_val_idx, r_mat_val_v, r_mat_val_rts, r_mat_val_abs_t_decay, r_mat_val_rel_t_decay = dp.adjmat_2_rmat(num_users, num_items, adj_val_data)

    # loss variables
    train_losses = []
    val_losses = []

    # evaluation variables
    val_recall = []
    val_prec = []
    val_ncdg = []
    val_rmse = []
    train_rmse = []

    # message passing data
    train_edge_index = edge_train_data['edge_index']
    val_edge_index = edge_val_data['edge_index']
    
    # supervision data
    train_src = r_mat_train_idx[0]
    train_dest = r_mat_train_idx[1]
    train_values = r_mat_train_v
    train_abs_t_decay = r_mat_train_abs_t_decay
    train_rel_t_decay = r_mat_train_rel_t_decay
    val_src = r_mat_val_idx[0]
    val_dest = r_mat_val_idx[1]
    val_values = r_mat_val_v
    val_abs_t_decay = r_mat_val_abs_t_decay
    val_rel_t_decay = r_mat_val_rel_t_decay
    
    print(f'num of users: {num_users}, num of items: {num_items}, model: {g_model}')

    #model = LGCN(num_users=num_users,
    #            num_items=num_items,
    #            num_layers=g_num_layers,
    #            embedding_dim = g_emb_dim,
    #            add_self_loops = True,
    #            mu = g_mean_rating,
    #            model=g_model,
    #            verbose=g_verbose)
    
    #self,
    #latent_dim, 
    #num_layers,
    #num_users,
    #num_items,
    #model, # 'LightGCN' or 'tempLGCN'
    #dropout=0.1, # Only used in NGCF
    #is_temp=False
      
    model = RecSysGNN(latent_dim=g_emb_dim,
                    num_layers=g_num_layers,
                    num_users=num_users,
                    num_items=num_items,
                    model='LightGCN'
                    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device is - {device}")
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr = g_lr, weight_decay=g_decay)

    train_src = train_src.to(device)
    train_dest = train_dest.to(device)
    train_values = train_values.to(device)
    train_abs_t_decay = train_abs_t_decay.to(device)
    train_rel_t_decay = train_rel_t_decay.to(device)

    val_src = val_src.to(device)
    val_dest = val_dest.to(device)
    val_values = val_values.to(device)
    val_abs_t_decay = val_abs_t_decay.to(device)
    val_rel_t_decay = val_rel_t_decay.to(device)

    train_edge_index = train_edge_index.to(device)
    val_edge_index = val_edge_index.to(device)

    loss_func = nn.MSELoss()

    all_compute_time = 0
    avg_compute_time = "{:.4f}".format(round(0, 4))
    min_RMSE = 1000
    min_RMSE_epoch = 0
    min_RECALL = 0
    min_PRECISION = 0
    min_F1 = 0
    val_epochs = 0

    if g_loadedModel:
        model.load_state_dict(torch.load('models/' + g_dataset + '_model.pt'))

    for epoch in tqdm(range(g_epochs), position=1, mininterval=1.0, ncols=100):
        start_time = time.time()
        
        if len(train_src) != g_batch_size:
            total_iterations = len(train_src) // g_batch_size + 1
        else:
            total_iterations = len(train_src) // g_batch_size
        
        model.train()
        train_loss = 0.
        
        # Generate batches of data using the minibatch function
        train_minibatches = minibatch(train_abs_t_decay, train_rel_t_decay, train_src, train_dest, train_values, batch_size=g_batch_size)
        
        # Iterate over each batch using enumerate
        for b_abs_t_decay, b_rel_t_decay, b_src, b_dest, b_values in train_minibatches:
            # forward(self, edge_index, edge_attrs, src, dest)
            b_pred_ratings = model.forward(train_edge_index, b_src, b_dest, b_abs_t_decay, b_rel_t_decay)
            b_loss = loss_func(b_pred_ratings, b_values)
            train_loss += b_loss
            
            optimizer.zero_grad()
            b_loss.backward()
            optimizer.step()
        
        train_loss = train_loss / total_iterations
        
        if epoch %  g_epochs_per_eval == 0:
            
            train_rmse.append(np.sqrt(train_loss.item()))
            
            model.eval()
            val_loss = 0.
            
            val_epochs += 1
            
            with torch.no_grad():         
                val_pred_ratings = []
                
                if len(val_src) != g_batch_size:
                    total_iterations = len(val_src) // g_batch_size + 1
                else:
                    total_iterations = len(val_src) // g_batch_size

                
                val_mini_batches = minibatch(val_abs_t_decay, val_rel_t_decay, val_src, val_dest, val_values, batch_size=g_batch_size)
                
                for b_abs_t_decay, b_rel_t_decay, b_src, b_dest, b_values in val_mini_batches:
                        
                    b_pred_ratings = model.forward(val_edge_index, b_src, b_dest, b_abs_t_decay, b_rel_t_decay)
                    
                    val_b_loss = loss_func(b_pred_ratings, b_values)
                    val_loss += val_b_loss
                    val_pred_ratings.extend(b_pred_ratings)
        
                val_loss = val_loss / total_iterations
                
                recall, prec = get_recall_at_k(r_mat_val_idx,
                                                            r_mat_val_v,
                                                            torch.tensor(val_pred_ratings),
                                                            k=g_top_k)
                
                ncdg = calculate_ndcg(r_mat_val_idx, r_mat_val_v, val_pred_ratings, k=g_top_k)
                
                recall = round(recall, 3)
                prec = round(prec, 3)
                val_recall.append(recall)
                val_prec.append(prec)
                val_ncdg.append(ncdg)
                val_rmse.append(np.sqrt(val_loss.item()))
                
                train_losses.append(train_loss.item())
                val_losses.append(val_loss.item())
                
                f_train_loss = "{:.4f}".format(round(np.sqrt(train_loss.item()), 4))
                f_val_loss = "{:.4f}".format(round(np.sqrt(val_loss.item()), 4))
                f_recall = "{:.4f}".format(round(recall, 4))
                f_precision = "{:.4f}".format(round(prec, 4))
                f_ncdg = "{:.4f}".format(round(ncdg, 4))
                
                if (recall + prec) != 0:
                    f_f1_score = "{:.4f}".format(round((2*recall*prec)/(recall + prec), 4))
                else:
                    f_f1_score = 0
                    
                f_time = "{:.2f}".format(round(time.time() - start_time, 2))
                f_epoch = "{:4.0f}".format(epoch)
                            
                if min_RMSE > np.sqrt(val_loss.item()):
                    save_model(model, 'models/' + g_dataset, '_model.pt')
                    min_RMSE = np.sqrt(val_loss.item())
                    min_RMSE_loss = f_val_loss
                    min_RMSE_epoch = epoch
                    min_RECALL_f = f_recall
                    min_PRECISION_f = f_precision
                    min_RECALL = recall
                    min_PRECISION = prec
                    min_F1 = f_f1_score
                    min_ncdg = f_ncdg

                if epoch %  (g_epochs_per_eval) == 0:
                    tqdm.write(f"[Epoch {f_epoch} - {f_time}, {avg_compute_time}]\tRMSE(train -> val): {f_train_loss}"
                            f" -> \033[1m{f_val_loss}\033[0m | "
                            f"Recall, Prec, F_score, NCDG:{f_recall, f_precision, f_f1_score, f_ncdg}")
                
        all_compute_time += (time.time() - start_time)
        avg_compute_time = "{:.4f}".format(round(all_compute_time/(epoch+1), 4)) 

    tqdm.write(f"\033[1mMinimum Seed {seed} -> RMSE: {min_RMSE_loss} at epoch {min_RMSE_epoch} with Recall, Precision, F1, NCDG: {min_RECALL_f, min_PRECISION_f, min_F1, min_ncdg}\033[0m")

    plot_loss(val_epochs, train_losses, val_losses, train_rmse, val_rmse, val_recall, val_prec)
    
    return min_RMSE, min_RECALL, min_PRECISION, min_ncdg


# load the dataset
rating_df, user_df, item_df, rating_stat = dp.load_data2(dataset=g_dataset, verbose=True)
num_users, num_items, g_mean_rating = rating_stat['num_users'], rating_stat['num_items'], rating_stat['mean_rating']

# add time distance column by calculating timestamp from the fixed minimum point
if g_model == 'lgcn_b_a' or g_model == 'lgcn_b_ar':
    rating_df = dp.add_u_abs_decay(rating_df=rating_df, beta=g_a_beta, method=a_method, verbose=g_verbose)

if g_model == 'lgcn_b_r' or g_model == 'lgcn_b_ar':
    rating_df = dp.add_u_rel_decay(rating_df=rating_df, beta=g_r_beta, win_size = g_win, method=r_method, verbose=g_verbose)

#rand_seed = [7, 12, 89, 91, 41]
rand_seed = [7]

rmses = []
recalls = []
precs = []
ncdgs = []

exp_n = 1

for seed in rand_seed:
    print(f'Experiment ({exp_n}) starts with seed:{seed}')
    rmse, recall, prec, ncdg = run_experiment(rating_df=rating_df, 
                                        num_users=num_users, 
                                        num_items=num_items, 
                                        g_mean_rating=g_mean_rating,
                                        g_seed=seed)
    rmses.append(rmse)
    recalls.append(recall)
    precs.append(prec)
    ncdgs.append(ncdg)
    
    exp_n += 1
