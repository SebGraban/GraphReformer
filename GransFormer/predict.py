import sys
import numpy as np
import networkx as nx
import torch
from GransFormer.data import my_decode_adj
from GransFormer.Transformer.Models import prepare_for_MADE

def get_graph(adj):
    '''
    get a graph from zero-padded adj
    :param adj:
    :return:
    '''
    # remove all zeros rows and columns
    len_1 = adj.shape[0]
    adj = adj[~np.all(adj == 0, axis=1)]
    adj = adj[:, ~np.all(adj == 0, axis=0)]
    len_2 = adj.shape[0]
    print('       ', len_1, ' ', len_2)
    sys.stdout.flush()
    adj = np.asmatrix(adj)
    G = nx.from_numpy_array(adj)
    return G

def generate_graph_exact(gg_model, args, device=None):

    if args.feed_graph_length:
        assert args.estimate_num_nodes

    # return None
    global min_par_idx
    gg_model.eval()

    if args.input_type == 'node_based':
        src_seq = torch.zeros((32, args.max_seq_len), dtype=torch.long).to(device)
        for i in range(args.max_seq_len - 1):
            #pred = gg_model, *_ (src_seq, src_seq).max(1)[1].view([32, args.max_seq_len])
            pred_logprobs, *_ = gg_model(src_seq, src_seq) #.max(1)[1].view([32, args.max_seq_len])
            pred_probs = pred_logprobs.exp() / pred_logprobs.exp().sum(axis=-1, keepdim=True).repeat(1,pred_logprobs.size(-1))
            pred = torch.tensor([np.random.choice(np.arange(probs.size(0)),size=1,p=probs.detach().cpu().numpy())[0]
                                 for probs in pred_probs]).view([32, args.max_seq_len]).to(device)
            src_seq[:, i + 1] = pred[:, i]
    elif args.input_type == 'preceding_neighbors_vector':

        if args.use_min_num_nodes:
            assert args.use_termination_bit

        src_seq = args.src_pad_idx * torch.ones((32, args.max_seq_len, args.max_num_node + 1),
                                  dtype=torch.float32).to(device)


        adj = torch.zeros((32, args.max_seq_len, args.max_seq_len), dtype=torch.float32).to(
            device)

        if args.estimate_num_nodes:
            len_gen = np.random.choice(np.arange(1,args.max_num_node + 1), 32, True, gg_model.num_nodes_prob[1:])
            if args.feed_graph_length:
                for i in range(src_seq.size(0)):
                    src_seq[i, len_gen[i]+1, 0] = args.one_input

        not_finished_idx = torch.ones([src_seq.size(0)]).bool().to(device)
        damaged_idx = torch.zeros([src_seq.size(0)]).bool().to(device)
        if args.use_bfs_incremental_parent_idx:
            min_par_idx = torch.zeros(src_seq.size(0), src_seq.size(2), dtype=torch.int32).bool().to(device)
        for i in range(args.max_seq_len - 1):

            tmp, dec_output = gg_model(src_seq, src_seq, src_seq, adj)
            pred_probs = torch.sigmoid(tmp).view(-1, args.max_seq_len, args.max_num_node + 1)
            # if args.use_max_prev_node and i > args.max_prev_node:
            #     pred_probs[:, i, 1:i - args.max_prev_node + 1] = 0
            # if args.use_bfs_incremental_parent_idx:
            #     for j in range(pred_probs.size(0)):
            #         pred_probs[j, i, 1:min_par_idx[j]] = 0
            num_trials = 0
            remainder_idx = not_finished_idx.clone()
            src_seq[remainder_idx, i+1, i+1:] = args.dontcare_input
            while remainder_idx.sum().item() > 0:
                num_trials += 1

                if args.use_MADE:
                    # if args.separate_termination_bit:
                    #     gold = args.trg_pad_idx * torch.ones(remainder_idx.sum().item(), src_seq.size(2) - 1).to(device)
                    #     term_bits = torch.rand([remainder_idx.sum().item()], device=device) < pred_probs[
                    #         remainder_idx, i, 0]
                    #     pred_probs = pred_probs[:, :, 1:]
                    # else:
                    gold = args.trg_pad_idx * torch.ones(remainder_idx.sum().item(), src_seq.size(2)).to(device)
                    for j in range(i + 1):
                        if args.use_max_prev_node and i > args.max_prev_node and j > 0 and j < i - args.max_prev_node + 1:
                            gold[remainder_idx, j] = args.dontcare_input
                        else:
                            if j == 0 and args.estimate_num_nodes:
                                gold[:, 0] = args.zero_input
                            elif j == 0 and args.use_min_num_nodes and i < args.min_num_node:
                                gold[:, 0] = args.zero_input
                            else:
                                tmp = (torch.rand([remainder_idx.sum().item()], device=device) < pred_probs[
                                    remainder_idx,
                                    i, j]).float()
                                ind_0 = tmp == 0
                                ind_1 = tmp == 1
                                tmp[ind_0] = args.zero_input
                                tmp[ind_1] = args.one_input
                                if args.use_bfs_incremental_parent_idx:
                                    tmp[min_par_idx[remainder_idx, j]] = args.zero_input
                                gold[:, j] = tmp
                        if j < i:
                            if args.separate_termination_bit:
                                tmp = gg_model.trg_word_MADE(torch.cat([dec_output[remainder_idx, i, :], prepare_for_MADE(gold[:,1:], args)], dim=1))
                                if gg_model.scale_prj:
                                    tmp *= gg_model.d_model ** -0.5
                                pred_probs[remainder_idx, i, 1:] = torch.sigmoid(tmp)
                            else:
                                tmp = gg_model.trg_word_MADE(torch.cat([dec_output[remainder_idx, i, :], prepare_for_MADE(gold, args)], dim=1))
                                if gg_model.scale_prj:
                                    tmp *= gg_model.d_model ** -0.5
                                pred_probs[remainder_idx, i, :] = torch.sigmoid(tmp)

                    src_seq[remainder_idx, i + 1, :i + 1] = gold[:, :i + 1]
                else:
                    tmp = (torch.rand([remainder_idx.sum().item(), i + 1], device=device) < pred_probs[remainder_idx,
                                                                                         i, :i + 1]).float()
                    if args.estimate_num_nodes:
                        tmp[:, 0] = 0
                    ind_0 = tmp == 0
                    ind_1 = tmp == 1
                    tmp[ind_0] = args.zero_input
                    tmp[ind_1] = args.one_input
                    src_seq[remainder_idx, i + 1, :i + 1] = tmp
                    if args.use_bfs_incremental_parent_idx:
                        src_seq[remainder_idx, i+1,:][min_par_idx[remainder_idx, :]] = args.zero_input
                    if args.use_max_prev_node and i > args.max_prev_node:
                        src_seq[remainder_idx, i+1, 1:i - args.max_prev_node + 1] = args.dontcare_input
                if i == 0:
                    src_seq[remainder_idx, i+1, 0] = args.zero_input
                    break

                if args.use_min_num_nodes and i < args.min_num_node:
                    src_seq[remainder_idx, i + 1, 0] = args.zero_input

                if args.estimate_num_nodes:
                    tmp_new_finished_idx = remainder_idx & torch.tensor(len_gen == i).to(device)
                    src_seq[remainder_idx, i+1, 0] = args.zero_input
                    src_seq[tmp_new_finished_idx, i+1, 0] = args.one_input

                if not args.use_termination_bit:
                    if args.estimate_num_nodes:
                        raise NotImplementedError
                    remainder_idx[:] = False
                    tmp_new_finished_idx = remainder_idx & ((src_seq[:, i+1, 1:i+1] == args.one_input).sum(-1) == 0)
                    src_seq[remainder_idx, i + 1, 0] = args.zero_input
                    src_seq[tmp_new_finished_idx, i + 1, 0] = args.one_input
                if args.allow_all_zeros:
                    remainder_idx[:] = False
                else:
                    remainder_idx = remainder_idx & ((src_seq[:, i + 1, : i + 1] == args.one_input).sum(-1) == 0)
                    if num_trials >= args.max_num_generate_trials:
                        # print('   reached max_num_gen_trials   dim:', i, '   num remainder:', remainder_idx.detach().cpu().sum().item())
                        if i+2 < args.max_seq_len:
                            src_seq[remainder_idx, i+2:, 0] = args.src_pad_idx
                        src_seq[remainder_idx, i+1, 0] = args.one_input
                        damaged_idx[remainder_idx] = True
                        remainder_idx[:] = False

            new_finished_idx = not_finished_idx & (src_seq[:, i + 1, 0] == args.one_input)
            src_seq[new_finished_idx, i + 1, 1:] = args.src_pad_idx
            if i > 0 and args.use_bfs_incremental_parent_idx:
                tmp = src_seq[not_finished_idx, i + 1, :] == args.one_input
                min_par_idx[not_finished_idx, :] = tmp.cumsum(dim=1) == 0
                min_par_idx[not_finished_idx, 0] = False
            not_finished_idx = not_finished_idx & (src_seq[:, i + 1, 0] != args.one_input)
            # if num_trials > 1:
            #     print('                          ', i, '      num of trials:', num_trials)
            if not_finished_idx.sum().item() == 0:
                break

            tmp = src_seq[not_finished_idx, i + 1, 1:i + 1]
            ind_0 = tmp == args.zero_input
            ind_1 = tmp == args.one_input
            tmp[ind_0] = 0
            tmp[ind_1] = 1
            adj[not_finished_idx, i + 1, 1:i + 1] = tmp
            adj[not_finished_idx, 1:i + 1, i + 1] = tmp

        ind_0 = src_seq == args.zero_input
        ind_1 = src_seq == args.one_input
        src_seq[ind_0] = 0
        src_seq[ind_1] = 1

        src_seq = src_seq[~damaged_idx]
    else:
        raise NotImplementedError

    # save graphs as pickle
    G_pred_list = []
    for i in range(32):
        adj_pred = my_decode_adj(src_seq[i,1:].cpu().numpy(), args)
        G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
        G_pred_list.append(G_pred)

    return G_pred_list
