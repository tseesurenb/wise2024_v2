'''
Created on Oct 12, 2023
Pytorch Implementation of TempLGCN in
Tseesuren et al. tempLGCN: Temporal Collaborative Filtering with Graph Convolutional Networks

@author: Tseesuren Batsuuri (tseesuren.batsuuri@hdr.mq.edu.au)
'''
#%% 
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn, optim
from utils import get_recall_at_k, minibatch 
import dataPrep as dp
from model import LGCN 
import time
from world import config
import torch

from captum.attr import IntegratedGradients, NeuronConductance, LayerConductance

    
def run_experiment(rating_df, num_users, num_items, g_mean_rating, config, g_seed):
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
   

    # get user statistics list: userId, # of ratings, mean rating, rating time distance
    u_stats = None #dp.get_user_stats(rating_df=rating_df)
    # get item statistics list: itemId, # of ratings, mean rating, rating time distance
    i_stats = None #dp.get_item_stats(rating_df=rating_df)

    e_idx, e_vals, e_ts, e_abs_t_decay, e_rel_t_decay = dp.get_edge_values(rating_df)

    o_train_idx, o_train_vals, o_train_ts, o_train_abs_t_decay, o_train_rel_t_decay, o_val_idx, o_val_vals, o_val_ts, o_val_abs_t_decay, o_val_rel_t_decay = dp.train_test_split_by_user(e_idx=e_idx, e_vals=e_vals, e_ts=e_ts, e_abs_t_decay=e_abs_t_decay, e_rel_t_decay = e_rel_t_decay, test_size=0.1, seed=g_seed)

    train_idx, train_v, train_rts, train_abs_t_decay, train_rel_t_decay = dp.rmat_2_adjmat(num_users, num_items, o_train_idx, o_train_vals, o_train_ts, o_train_abs_t_decay, o_train_rel_t_decay)
    val_idx, val_v, val_rts, val_abs_t_decay, val_rel_t_decay = dp.rmat_2_adjmat(num_users, num_items, o_val_idx, o_val_vals, o_val_ts, o_val_abs_t_decay, o_val_rel_t_decay)

    r_mat_train_idx, r_mat_train_v, r_mat_train_rts, r_mat_train_abs_t_decay, r_mat_train_rel_t_decay = dp.adjmat_2_rmat(num_users, num_items, train_idx, train_v, train_rts, train_abs_t_decay, train_rel_t_decay)
    r_mat_val_idx, r_mat_val_v, r_mat_val_rts, r_mat_val_abs_t_decay, r_mat_val_rel_t_decay = dp.adjmat_2_rmat(num_users, num_items, val_idx, val_v, val_rts, val_abs_t_decay, val_rel_t_decay)

    train_losses = []
    val_losses = []

    val_recall_at_ks = []
    val_precision_at_k = []

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

    model = LGCN(num_users=num_users,
                num_items=num_items,
                num_layers=g_num_layers,
                embedding_dim = g_emb_dim,
                add_self_loops = True,
                mu = g_mean_rating,
                model=g_model,
                u_stats = u_stats,
                verbose=g_verbose)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr = g_lr, weight_decay=g_decay)
    #scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.95)

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

    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)

    loss_func = nn.MSELoss()

    all_compute_time = 0
    avg_compute_time = "{:.4f}".format(round(0, 4))
    min_RMSE = 1000
    min_RMSE_epoch = 0
    min_RECALL = 0
    min_PRECISION = 0
    min_F1 = 0

    if g_loadedModel:
        model.load_state_dict(torch.load('models/' + g_dataset + '_model.pt'))


    for epoch in tqdm(range(g_epochs), position=1, mininterval=5.0, ncols=100):
        start_time = time.time()
        
        model.train()    
        
        if len(train_src) != g_batch_size:
            total_iterations = len(train_src) // g_batch_size + 1
        else:
            total_iterations = len(train_src) // g_batch_size
        
        train_loss = 0.
        for (b_i,
            (b_abs_t_decay, b_rel_t_decay, b_src, b_dest, b_values)) in enumerate(minibatch(train_abs_t_decay,
                                                                            train_rel_t_decay,
                                                                        train_src,
                                                                        train_dest,
                                                                        train_values,
                                                                        batch_size=g_batch_size)):
            
            b_pred_ratings = model.forward(b_src, b_dest, train_idx, b_abs_t_decay, b_rel_t_decay)
            b_loss = loss_func(b_pred_ratings, b_values)
            train_loss += b_loss
            
            optimizer.zero_grad()
            b_loss.backward()
            optimizer.step()
        
        train_loss = train_loss / total_iterations
        
        if epoch %  g_epochs_per_eval == 0:
            model.eval()
            
            with torch.no_grad():         
                val_pred_ratings = []
                
                if len(val_src) != g_batch_size:
                    total_iterations = len(val_src) // g_batch_size + 1
                else:
                    total_iterations = len(val_src) // g_batch_size
            
                val_loss = 0.
                for (batch_i,
                    (b_abs_t_decay, b_rel_t_decay, b_src, b_dest, b_values)) in enumerate(minibatch(val_abs_t_decay,
                                                                                    val_rel_t_decay,
                                                                                val_src,
                                                                                val_dest,
                                                                                val_values,
                                                                                batch_size=g_batch_size)):
                        
                    b_pred_ratings = model.forward(b_src, b_dest, val_idx, b_abs_t_decay, b_rel_t_decay)
                    
                    val_b_loss = loss_func(b_pred_ratings, b_values)
                    val_loss += val_b_loss
                    val_pred_ratings.extend(b_pred_ratings)
                
                val_loss = val_loss / total_iterations
                
                recall_at_k, precision_at_k = get_recall_at_k(r_mat_val_idx,
                                                            r_mat_val_v,
                                                            torch.tensor(val_pred_ratings),
                                                            k=g_top_k)
                
                recall = round(recall_at_k, 3)
                precision = round(precision_at_k, 3)
                val_recall_at_ks.append(recall)
                val_precision_at_k.append(precision)
                
                train_losses.append(train_loss.item())
                val_losses.append(val_loss.item())
                
                f_train_loss = "{:.4f}".format(round(np.sqrt(train_loss.item()), 4))
                f_val_loss = "{:.4f}".format(round(np.sqrt(val_loss.item()), 4))
                f_recall = "{:.4f}".format(round(recall, 4))
                f_precision = "{:.4f}".format(round(precision, 4))
                if (recall + precision) != 0:
                    f_f1_score = "{:.4f}".format(round((2*recall*precision)/(recall + precision), 4))
                else:
                    f_f1_score = 0
                    
                f_time = "{:.2f}".format(round(time.time() - start_time, 2))
                f_epoch = "{:4.0f}".format(epoch)
                            
                if min_RMSE > np.sqrt(val_loss.item()):
                    torch.save(model.state_dict(), 'models/' + g_dataset + '_model.pt')
                    min_RMSE = np.sqrt(val_loss.item())
                    min_RMSE_loss = f_val_loss
                    min_RMSE_epoch = epoch
                    min_RECALL_f = f_recall
                    min_PRECISION_f = f_precision
                    min_RECALL = recall
                    min_PRECISION = precision
                    min_F1 = f_f1_score

                if epoch %  (g_epochs_per_eval) == 0:
                    tqdm.write(f"[Epoch {f_epoch} - {f_time}, {avg_compute_time}]\tRMSE(train -> val): {f_train_loss}"
                            f" -> \033[1m{f_val_loss}\033[0m | "
                            f"Recall, Prec, F_score:{f_recall, f_precision, f_f1_score}")
                
        all_compute_time += (time.time() - start_time)
        avg_compute_time = "{:.4f}".format(round(all_compute_time/(epoch+1), 4)) 
        #if epoch % g_epochs_per_lr_decay == 0 and epoch != 0:
        #    scheduler.step()
        
    tqdm.write(f"\033[1mMinimum Seed {seed} -> RMSE: {min_RMSE_loss} at epoch {min_RMSE_epoch} with Recall, Precision, F1: {min_RECALL_f, min_PRECISION_f, min_F1}\033[0m")
    #tqdm.write(f"The experiment is complete.")
    
    return min_RMSE, min_RECALL, min_PRECISION

g_r_beta = config['r_beta']
g_a_beta = config['a_beta']
a_method = config['a_method']
r_method = config['r_method']
g_dataset = config['dataset']
g_verbose = config['verbose']
g_model = config['model']
g_win = config['win']    
# load the dataset
rating_df, num_users, num_items, g_mean_rating = dp.load_data(dataset=g_dataset, verbose=g_verbose)
# add time distance column by calculating timestamp from the fixed minimum point
if g_model == 'lgcn_b_a' or g_model == 'lgcn_b_ar':
    rating_df = dp.add_u_abs_decay(rating_df=rating_df, beta=g_a_beta, method=a_method, verbose=g_verbose)

if g_model == 'lgcn_b_r' or g_model == 'lgcn_b_ar':
    rating_df = dp.add_u_rel_decay(rating_df=rating_df, beta=g_r_beta, win_size = g_win, method=r_method, verbose=g_verbose)

rand_seed = [7, 12, 89, 91, 41]

rmses = []
recalls = []
precs = []

exp_n = 1

for seed in rand_seed:
    print(f'Experiment ({exp_n}) starts with seed:{seed}')
    rmse, recall, prec = run_experiment(rating_df=rating_df, 
                                        num_users=num_users, 
                                        num_items=num_items, 
                                        g_mean_rating=g_mean_rating,
                                        config=config,
                                        g_seed=seed)
    
    #print(f"Seed {seed} -> RMSE: {rmse:0.4f}, reCall:{recall:0.4f}, Prec:{prec:0.4f}")
    rmses.append(rmse)
    recalls.append(recall)
    precs.append(prec)
    exp_n += 1
    
print(f"  RMSE:\t {rmses[0]:.4f}, {rmses[1]:.4f}, {rmses[2]:.4f}, {rmses[3]:.4f}, {rmses[4]:.4f} -> {sum(rmses) / len(rmses):.4f}")
print(f"Recall:\t {recalls[0]:.4f}, {recalls[1]:.4f}, {recalls[2]:.4f}, {recalls[3]:.4f}, {recalls[4]:.4f} -> {sum(recalls) / len(recalls):.4f}")
print(f"  Prec:\t {precs[0]:.4f}, {precs[1]:.4f}, {precs[2]:.4f}, {precs[3]:.4f}, {precs[4]:.4f} -> {sum(precs) / len(precs):.4f}")

#%%

model = LGCN(num_users=num_users,
            num_items=num_items,
            num_layers=4,
            embedding_dim = 64,
            add_self_loops = True,
            mu = g_mean_rating,
            model=g_model,
            verbose=g_verbose)

cond = LayerConductance(model, model.f)

cond_vals = cond.attribute(b_src, b_dest, train_idx, b_abs_t_decay, b_rel_t_decay,target=1)
            cond_vals = cond_vals.detach().numpy()
