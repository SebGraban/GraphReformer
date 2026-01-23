import torch
import torch.nn.functional as F
import networkx as nx
from graphgen.dfscode.dfs_wrapper import graph_from_dfscode
from graph_rnn.helper import get_attributes_len_for_graph_rnn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import numpy as np 
import sys
from GransFormer.Transformer.Models import prepare_for_MADE

def compute_loss(model_type, model, data, args, device=None):
    if model_type == 'DFSGraphReformer' or model_type == 'DFSGraphTransformer':
        return compute_loss_dfs_loss(model, data, args['pad_token_id'], device)
    elif model_type == 'GraphRNN':
        return compute_loss_graph_rnn(model, data, device, args['feature_map'], args['max_prev_node'], args['max_head_and_tail'])
    elif model_type == 'graphgen':
        return compute_loss_graphgen(model, data, device, args['feature_map'])
    elif model_type == 'Gransformer':
        return compute_gransformer_loss(model, data, device, args)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def cal_loss(pred, dec_output, gold, trg_pad_idx, args, model, termination_bit_weight=None, smoothing=False):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''
    if smoothing:
        if args.input_type == 'node_based':
            gold = gold.contiguous().view(-1)
            eps = 0.1
            n_class = pred.size(1)

            one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)

            non_pad_mask = gold.ne(trg_pad_idx)
            loss = -(one_hot * log_prb).sum(dim=1)
            loss = loss.masked_select(non_pad_mask).sum()  # average later
        else:
            raise NotImplementedError
    else:
        if args.input_type == 'node_based':
            gold = gold.contiguous().view(-1)
            loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction='sum')
        elif args.input_type == 'preceding_neighbors_vector':

            if args.allow_all_zeros and (args.use_max_prev_node or args.use_bfs_incremental_parent_idx):
                raise NotImplementedError

            pred = torch.sigmoid(pred).view(-1, args.max_seq_len, pred.size(-1))
            cond_1 = gold != args.trg_pad_idx
            if args.use_max_prev_node:
                cond_mpn = torch.ones(gold.size(0), gold.size(1), gold.size(2)).to(device=args.device)
                cond_mpn = torch.tril(cond_mpn, diagonal=0)
                cond_mpn = torch.triu(cond_mpn, diagonal=-args.max_prev_node+1)
                cond_mpn[:, :, 0] = 1
                cond_1 = cond_1 * cond_mpn

            if args.use_bfs_incremental_parent_idx:
                gold_0 = gold[:, :, 1:].clone()
                ind_dontcare = gold_0 == args.dontcare_input
                ind_0 = gold_0 == args.zero_input
                ind_1 = gold_0 == args.one_input
                gold_0[ind_dontcare] = 0
                gold_0[ind_0] = 0
                gold_0[ind_1] = 1
                cond_bfs_par = gold_0.cumsum(dim=2) > 0
                cond_bfs_par = torch.cat(
                    [torch.zeros(cond_bfs_par.size(0), 1, cond_bfs_par.size(2)).bool().to(args.device),
                     cond_bfs_par[:, :-1, :]], dim=1)
                cond_bfs_par[:, 1, 0] = True
                cond_bfs_par = torch.cat(
                    [torch.ones(cond_bfs_par.size(0), cond_bfs_par.size(1), 1).bool().to(args.device), cond_bfs_par],
                    dim=2)
                cond_bfs_par = torch.tril(cond_bfs_par, diagonal=0)
                cond_1 = cond_1 * cond_bfs_par

            if (not args.use_termination_bit) or args.feed_graph_length:
                cond_1[:, :, 0] = False

            pred_1 = torch.tril(pred * cond_1, diagonal=0)
            gold_1 = torch.tril(gold * cond_1, diagonal=0)
            ind_0 = gold_1 == args.zero_input
            ind_1 = gold_1 == args.one_input
            gold_1[ind_0] = 0
            gold_1[ind_1] = 1

            assert not (args.weight_positions and (termination_bit_weight is not None))

            if args.weight_positions:
                loss_1 = F.binary_cross_entropy(pred_1, gold_1, reduction='none').sum(-1)
                loss_1 = loss_1 * model.positions_weights.view(1,-1)
                loss_1 = loss_1.sum()
            elif termination_bit_weight is not None:
                loss_1 = F.binary_cross_entropy(pred_1, gold_1, reduction='none')
                loss_1[:,:,0] = loss_1[:,:,0] * termination_bit_weight
                loss_1 = loss_1.sum()
            else:
                loss_1 = F.binary_cross_entropy(pred_1, gold_1, reduction='sum')

            if args.allow_all_zeros or not args.use_termination_bit:
                loss = loss_1
            else:
                if args.use_MADE:
                    gold_all_zeros = gold.clone()
                    gold_all_zeros[gold == args.one_input] = args.zero_input
                    if args.separate_termination_bit:
                        gold_all_zeros = gold_all_zeros[:, :, 1:]
                    tmp = model.trg_word_MADE(torch.cat([dec_output, prepare_for_MADE(gold_all_zeros, args)], dim=2))
                    if model.scale_prj:
                        tmp *= model.d_model ** -0.5
                    pred_all_zeros = torch.sigmoid(tmp)
                    if args.separate_termination_bit:
                        pred_all_zeros = torch.cat([pred[:,:,:1], pred_all_zeros], dim=2)
                else:
                    pred_all_zeros = pred

                if args.feed_graph_length:
                    cond_0 = gold[:,:,0] == args.zero_input
                else:
                    cond_0 = gold[:,:,0] != args.trg_pad_idx
                cond_0[:, 0] = False
                cond_2 = cond_0.unsqueeze(-1).repeat(1, 1, gold.size(-1))
                if args.use_max_prev_node:
                    cond_2 = cond_2 * cond_mpn
                if args.use_bfs_incremental_parent_idx:
                    cond_2 = cond_2 * cond_bfs_par
                if args.feed_graph_length:
                    cond_2[:,:,0] = False
                pred_2 = torch.tril(pred_all_zeros * cond_2, diagonal=0)
                gold_2 = torch.zeros(gold.size(0), gold.size(1), gold.size(2), device=gold.device)

                p_zero = torch.exp(-F.binary_cross_entropy(pred_2, gold_2, reduction='none').sum(-1))
                loss_2 = torch.log(torch.max(1-p_zero[cond_0], torch.tensor([1e-9]).to(args.device)))
                if args.weight_positions:
                    loss_2 = loss_2 * model.positions_weights
                loss_2 = loss_2.sum()
                loss = loss_1 + loss_2
        elif args.input_type == 'max_prev_node_neighbors_vec':

            pred = torch.sigmoid(pred).view(-1, args.max_seq_len, pred.size(-1))

            if args.allow_all_zeros:
                raise NotImplementedError

            cond_pad = gold != args.trg_pad_idx
            cond_max_prev = torch.ones(pred.size(0), pred.size(1), pred.size(2)).to(args.device)
            cond_max_prev = torch.tril(cond_max_prev, diagonal=-1)
            cond_max_prev = torch.flip( cond_max_prev, [2])
            cond_max_prev[:, :, 0] = 1

            if args.use_bfs_incremental_parent_idx:
                gold_0 = gold[:,:,1:].clone()
                ind_0 = gold_0 == args.zero_input
                ind_1 = gold_0 == args.one_input
                gold_0[ind_0] = 0
                gold_0[ind_1] = 1
                cond_bfs_par = gold_0.cumsum(dim=2) > 0
                cond_bfs_par = torch.cat(
                    [cond_bfs_par, torch.ones(cond_bfs_par.size(0), cond_bfs_par.size(1), 1).bool().to(args.device)],
                    dim=2)
                cond_bfs_par = torch.cat(
                    [torch.zeros(cond_bfs_par.size(0), 1, cond_bfs_par.size(2)).bool().to(args.device),
                     cond_bfs_par[:, :-1, :]], dim=1)
                cond_bfs_par[:, :, 0] = True
                cond_max_prev = cond_bfs_par

            pred_1 = pred * cond_pad * cond_max_prev
            gold_1 = gold * cond_pad * cond_max_prev
            ind_0 = gold_1 == args.zero_input
            ind_1 = gold_1 == args.one_input
            gold_1[ind_0] = 0
            gold_1[ind_1] = 1

            loss_1 = F.binary_cross_entropy(pred_1, gold_1, reduction='sum')

            cond_zeros_1d = gold[:, :, 0] != args.trg_pad_idx
            cond_zeros_1d[:, 0] = False
            cond_zeros_2d = cond_zeros_1d.unsqueeze(-1).repeat(1, 1, gold.size(-1))

            pred_2 = pred * cond_zeros_2d * cond_max_prev
            gold_2 = torch.zeros(gold.size(0), gold.size(1), gold.size(2), device=gold.device)

            p_zero = torch.exp(-F.binary_cross_entropy(pred_2, gold_2, reduction='none').sum(-1))
            loss_2 = torch.log(torch.max(1-p_zero[cond_zeros_1d], torch.tensor([1e-9]).to(args.device))).sum()
            loss = loss_1 + loss_2
        else:
            raise NotImplementedError
    return loss

def cal_performance(pred, dec_output, gold, trg_pad_idx, args, model, termination_bit_weight=None, smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, dec_output, gold, trg_pad_idx, args, model, termination_bit_weight, smoothing)
    if args.input_type == 'node_based':
        pred = pred.max(1)[1]
        gold = gold.contiguous().view(-1)
        non_pad_mask = gold.ne(trg_pad_idx)
        n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
        n_word = non_pad_mask.sum().item()

        return loss, n_correct, n_word
    elif args.input_type in ['preceding_neighbors_vector', 'max_prev_node_neighbors_vec']:
        return loss, None
    else:
        raise NotImplementedError

def compute_gransformer_loss(model, data, device, args):

    if args.use_MADE:
        model.trg_word_MADE.update_masks()

    sys.stdout.flush()
    src_seq = data['src_seq'].to(device)
    trg_seq = data['src_seq'].to(device)

    gold = data['trg_seq'].contiguous().to(device)
    adj = data['adj'].to(device)

    pred, dec_output = model(src_seq, trg_seq, gold, adj)
    loss, *_ = cal_performance( pred, dec_output, gold, trg_pad_idx=0, args=args, model=model, smoothing=False)

    return loss

def compute_loss_graphgen(model, data, device, feature_map):

    x_len_unsorted = data['len'].to(device)
    x_len_max = max(x_len_unsorted)
    batch_size = x_len_unsorted.size(0)

    # sort input for packing variable length sequences
    x_len, sort_indices = torch.sort(x_len_unsorted, dim=0, descending=True)

    max_nodes = feature_map['max_nodes']
    len_node_vec, len_edge_vec = len(
        feature_map['node_forward']) + 1, len(feature_map['edge_forward']) + 1
    feature_len = 2 * (max_nodes + 1) + 2 * len_node_vec + len_edge_vec

    # Prepare targets with end_tokens already there
    t1 = torch.index_select(
        data['t1'][:, :x_len_max + 1].to(device), 0, sort_indices)
    t2 = torch.index_select(
        data['t2'][:, :x_len_max + 1].to(device), 0, sort_indices)
    v1 = torch.index_select(
        data['v1'][:, :x_len_max + 1].to(device), 0, sort_indices)
    e = torch.index_select(
        data['e'][:, :x_len_max + 1].to(device), 0, sort_indices)
    v2 = torch.index_select(
        data['v2'][:, :x_len_max + 1].to(device), 0, sort_indices)
    x_t1, x_t2 = F.one_hot(t1, num_classes=max_nodes +
                           2)[:, :, :-1], F.one_hot(t2, num_classes=max_nodes + 2)[:, :, :-1]
    x_v1, x_v2 = F.one_hot(v1, num_classes=len_node_vec +
                           1)[:, :, :-1], F.one_hot(v2, num_classes=len_node_vec + 1)[:, :, :-1]
    x_e = F.one_hot(e, num_classes=len_edge_vec + 1)[:, :, :-1]

    x_target = torch.cat((x_t1, x_t2, x_v1, x_e, x_v2), dim=2).float()

    # initialize dfs_code_rnn hidden according to batch size
    model['dfs_code_rnn'].hidden = model['dfs_code_rnn'].init_hidden(
        batch_size=batch_size)

    # Teacher forcing: Feed the target as the next input
    # Start token is all zeros
    dfscode_rnn_input = torch.cat(
        (torch.zeros(batch_size, 1, feature_len, device=device), x_target[:, :-1, :]), dim=1)

    # Forward propogation
    dfscode_rnn_output = model['dfs_code_rnn'](
        dfscode_rnn_input, input_len=x_len + 1)

    # Evaluating dfscode tuple
    timestamp1 = model['output_timestamp1'](dfscode_rnn_output)
    timestamp2 = model['output_timestamp2'](dfscode_rnn_output)
    vertex1 = model['output_vertex1'](dfscode_rnn_output)
    edge = model['output_edge'](dfscode_rnn_output)
    vertex2 = model['output_vertex2'](dfscode_rnn_output)

    x_pred = torch.cat(
        (timestamp1, timestamp2, vertex1, edge, vertex2), dim=2)

    # Cleaning the padding i.e setting it to zero
    x_pred = pack_padded_sequence(x_pred, x_len.cpu() + 1, batch_first=True)
    x_pred, _ = pad_packed_sequence(x_pred, batch_first=True)

    weight = None

    loss_sum = F.binary_cross_entropy(
        x_pred, x_target, reduction='none', weight=weight)
    loss = torch.mean(
        torch.sum(loss_sum, dim=[1, 2]) / (x_len.float() + 1))

    return loss


def compute_loss_graph_rnn(model, data, device, feature_map, max_prev_node, max_head_and_tail):
    x_unsorted = data['x'].to(device)

    x_len_unsorted = data['len'].to(device)
    x_len_max = max(x_len_unsorted)
    x_unsorted = x_unsorted[:, 0:max(x_len_unsorted), :]

    len_node_vec, len_edge_vec, num_nodes_to_consider = get_attributes_len_for_graph_rnn(
        len(feature_map['node_forward']), len(feature_map['edge_forward']),
        max_prev_node, max_head_and_tail)

    batch_size = x_unsorted.size(0)
    # sort input for packing variable length sequences
    x_len, sort_indices = torch.sort(x_len_unsorted, dim=0, descending=True)
    x = torch.index_select(x_unsorted, 0, sort_indices)

    # initialize node_level_rnn hidden according to batch size
    model['node_level_rnn'].hidden = model['node_level_rnn'].init_hidden(
        batch_size=batch_size)

    # Teacher forcing: Feed the target as the next input
    # Start token for graph level RNN decoder is node feature second last bit is 1
    node_level_input = torch.cat(
        (torch.zeros(batch_size, 1, x.size(2), device=device), x), dim=1)
    node_level_input[:, 0, len_node_vec - 2] = 1

    # Forward propogation
    node_level_output = model['node_level_rnn'](
        node_level_input, input_len=x_len + 1)

    # Evaluating node predictions
    x_pred_node = model['output_node'](node_level_output)

    # Evaluating edge predictions
    # Make a 2D matrix of edge feature vectors with size = [sum(x_len)] x [min(x_len_max - 1, num_nodes_to_consider) * len_edge_vec]
    # 2D matrix will have edge vectors sorted by time_stamp in graph level RNN
    edge_mat_packed = pack_padded_sequence(
        x[:, :, len_node_vec: min(
            x_len_max - 1, num_nodes_to_consider) * len_edge_vec + len_node_vec],
        x_len.cpu(), batch_first=True)

    edge_mat, _ = edge_mat_packed.data, edge_mat_packed.batch_sizes

    # Time stamp 'i' corresponds to edge feature sequence of length i (including start token added later)
    # Reverse the matrix in dim 0 (for packing purposes)
    idx = torch.LongTensor(
        [i for i in range(edge_mat.size(0) - 1, -1, -1)]).to(device)
    edge_mat = edge_mat.index_select(0, idx)

    # Start token of edge level RNN is 1 at second last position in vector of length len_edge_vector
    # End token of edge level RNN is 1 at last position in vector of length len_edge_vector
    # Convert the edge_mat in a 3D tensor of size
    # [sum(x_len)] x [min(x_len_max, num_nodes_to_consider + 1)] x [len_edge_vec]
    edge_mat = edge_mat.reshape(edge_mat.size(0), min(
        x_len_max - 1, num_nodes_to_consider), len_edge_vec)
    edge_level_input = torch.cat(
        (torch.zeros(sum(x_len), 1, len_edge_vec, device=device), edge_mat), dim=1)
    edge_level_input[:, 0, len_edge_vec - 2] = 1

    # Compute descending list of lengths for y_edge
    x_edge_len = []
    # Histogram of y_len
    x_edge_len_bin = torch.bincount(x_len)
    for i in range(len(x_edge_len_bin) - 1, 0, -1):
        # count how many x_len is above and equal to i
        count_temp = torch.sum(x_edge_len_bin[i:]).item()

        # put count_temp of them in x_edge_len each with value min(i, num_nodes_to_consider + 1)
        x_edge_len.extend([min(i, num_nodes_to_consider + 1)] * count_temp)

    x_edge_len = torch.LongTensor(x_edge_len).to(device)

    # Get edge-level RNN hidden state from node-level RNN output at each timestamp
    # Ignore the last hidden state corresponding to END
    hidden_edge = model['embedding_node_to_edge'](
        node_level_output[:, 0:-1, :])

    # Prepare hidden state for edge level RNN similiar to edge_mat
    # Ignoring the last graph level decoder END token output (all 0's)
    hidden_edge = pack_padded_sequence(
        hidden_edge, x_len.cpu(), batch_first=True).data
    idx = torch.LongTensor(
        [i for i in range(hidden_edge.size(0) - 1, -1, -1)]).to(device)
    hidden_edge = hidden_edge.index_select(0, idx)

    # Set hidden state for edge-level RNN
    # shape of hidden tensor (num_layers, batch_size, hidden_size)
    hidden_edge = hidden_edge.view(1, hidden_edge.size(0), hidden_edge.size(1))
    hidden_edge_rem_layers = torch.zeros(model['edge_level_rnn'].num_layers - 1, hidden_edge.size(1), hidden_edge.size(2), device=device)
    model['edge_level_rnn'].hidden = torch.cat(
        (hidden_edge, hidden_edge_rem_layers), dim=0)

    # Run edge level RNN
    x_pred_edge = model['edge_level_rnn'](
        edge_level_input, input_len=x_edge_len)

    # cleaning the padding i.e setting it to zero
    x_pred_node = pack_padded_sequence(
        x_pred_node, x_len.cpu() + 1, batch_first=True)
    x_pred_node, _ = pad_packed_sequence(x_pred_node, batch_first=True)
    x_pred_edge = pack_padded_sequence(
        x_pred_edge, x_edge_len.cpu(), batch_first=True)
    x_pred_edge, _ = pad_packed_sequence(x_pred_edge, batch_first=True)

    # Loss evaluation & backprop
    x_node = torch.cat(
        (x[:, :, :len_node_vec], torch.zeros(batch_size, 1, len_node_vec, device=device)), dim=1)
    x_node[torch.arange(batch_size), x_len, len_node_vec - 1] = 1

    x_edge = torch.cat((edge_mat, torch.zeros(
        sum(x_len), 1, len_edge_vec, device=device)), dim=1)
    x_edge[torch.arange(sum(x_len)), x_edge_len - 1, len_edge_vec - 1] = 1

    loss1 = F.binary_cross_entropy(x_pred_node, x_node, reduction='sum')
    loss2 = F.binary_cross_entropy(x_pred_edge, x_edge, reduction='sum')

    # Avg (node prediction + edge prediction) error per example
    loss = (loss1 + loss2) / batch_size

    return loss

def compute_loss_dfs_loss(model, data, pad_token_id, device):

    inputs = {k: v.to(device) for k, v in data['inputs'].items()}
    targets = {k: v.to(device) for k, v in data['targets'].items()}

    pred = model(inputs)

    loss_total = 0.0

    weights = {
        'v1': 3.0,
        'v2': 3.0,
        'l1': 1.0,
        'e':  1.0,
        'l2': 1.0,
    }


    # Token-level cross entropy losses
    for key in targets:
        logits = pred[key]
        target = targets[key]
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target.view(-1),
            ignore_index=pad_token_id
        )
        loss_total += loss * weights[key]

    return loss_total 

def evaluate(model_type, model, val_loader, args, device):
    if model_type == 'DFSGraphReformer' or model_type == 'DFSGraphTransformer':
        model.eval()
    elif model_type == 'GraphRNN' or model_type == 'graphgen':
        for _, net in model.items():
            net.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            loss = compute_loss(model_type, model, batch, args, device)
            total_loss += loss.item()
    return total_loss / len(val_loader)


def train_model(model_type, model, train_dataloader, val_dataloader, optimizer, device, args, model_name, num_epochs=10):
    if model_type == 'Gransformer':
        if args.estimate_num_nodes or args.weight_positions:
            print('estimation of num_nodes_prob started')
            num_nodes_prob = np.zeros(args.max_num_node + 1)
            for epoch in range(10):
                print(epoch, ' ', end='')
                sys.stdout.flush()
                for data in train_dataloader:
                    adj = data['adj'].to(args.device)
                    for a in adj:
                        idx = a.sum(dim=0).bool().sum().item()
                        num_nodes_prob[idx] += 1
            num_nodes_prob = num_nodes_prob / num_nodes_prob.sum()
            print('estimation of num_nodes_prob finished')
            if args.estimate_num_nodes:
                model.num_nodes_prob = num_nodes_prob
            if args.weight_positions:
                tmp = np.cumsum(num_nodes_prob, axis=0)
                tmp = 1 - tmp[:-1]
                tmp = np.concatenate([np.array([1.]), tmp])
                tmp[tmp <= 0] = np.min(tmp[tmp > 0])
                position_weights = 1 / tmp
                model.positions_weights = torch.tensor(position_weights).to(args.device).view(1, -1)

    if model_type == 'DFSGraphReformer' or model_type == 'DFSGraphTransformer' or model_type == 'Gransformer':
        model.to(device)
        model.train()
    elif model_type == 'GraphRNN' or model_type == 'graphgen':
        for _, net in model.items():
            net.to(device)
            net.train()

    best_val_loss = float('inf')
    patience = 30  # Increased from 10 to account for larger batch sizes
    patience_counter = 0

    for epoch in range(num_epochs):
        total_loss = 0.0
        if model_type == 'DFSGraphReformer' or model_type == 'DFSGraphTransformer' or model_type == 'Gransformer':
            model.to(device)
            model.train()
        elif model_type == 'GraphRNN' or model_type == 'graphgen':
            for _, net in model.items():
                net.to(device)
                net.train()
        for batch in train_dataloader:
            if model_type == 'GraphRNN' or model_type == 'graphgen':
                for _, net in model.items():
                    net.zero_grad()
            else:
                optimizer.zero_grad()
            loss = compute_loss(model_type, model, batch, args, device)
            loss.backward()

            if model_type == 'GraphRNN' or model_type == 'graphgen':
                for _, opt in optimizer.items():
                    opt.step()
                total_loss += loss.data.item()
            elif model_type == 'Gransformer':
                optimizer.step_and_update_lr()
                total_loss += loss.item()
            else:
                optimizer.step()
                total_loss += loss.item()

        print(f"[Epoch {epoch+1}] Loss: {total_loss / len(train_dataloader):.4f}")

        # Validation
        val_loss = evaluate(model_type, model, val_dataloader, args, device)
        print(f"[Epoch {epoch+1}] Validation Loss: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            if model_type == 'DFSGraphReformer' or model_type == 'DFSGraphTransformer' or model_type == 'Gransformer':
                torch.save(model.state_dict(), model_name)
            elif model_type == 'GraphRNN' or model_type == 'graphgen':
                for net_name, net in model.items():
                    torch.save(net.state_dict(), f"{model_name.split('.')[0]}_{net_name}.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

def generate_dfs_code(
    model,
    node_id_vocab,
    node_label_vocab,
    edge_label_vocab,
    max_len=64,
    temperature=1.0,
    device='cuda'
):
    model.eval()
    model.to(device)

    pad_id = node_id_vocab.get_id("<PAD>")
    end_id = node_id_vocab.get_id("<END>")
    start_token = ("<START>", "<START>", "<START>", "<START>", "<START>")

    # Prepare first token
    v1_seq = [node_id_vocab.get_id(start_token[0])]
    v2_seq = [node_id_vocab.get_id(start_token[1])]
    l1_seq = [node_label_vocab.get_id(start_token[2])]
    e_seq  = [edge_label_vocab.get_id(start_token[3])]
    l2_seq = [node_label_vocab.get_id(start_token[4])]

    for _ in range(max_len - 1):  # Already have one token
        # Prepare inputs
        inputs = {
            'v1': torch.tensor([v1_seq], dtype=torch.long, device=device),
            'v2': torch.tensor([v2_seq], dtype=torch.long, device=device),
            'l1': torch.tensor([l1_seq], dtype=torch.long, device=device),
            'e':  torch.tensor([e_seq],  dtype=torch.long, device=device),
            'l2': torch.tensor([l2_seq], dtype=torch.long, device=device),
        }

        # Forward pass
        with torch.no_grad():
            logits = model(inputs)

        next_vals = {}
        for key in logits:
            last_logits = logits[key][:, -1, :] / temperature
            probs = F.softmax(last_logits, dim=-1)
            idx = torch.multinomial(probs, 1)
            next_vals[key] = idx.item()

        # Append predicted token
        v1_seq.append(next_vals['v1'])
        v2_seq.append(next_vals['v2'])
        l1_seq.append(next_vals['l1'])
        e_seq.append(next_vals['e'])
        l2_seq.append(next_vals['l2'])

        # Check if we reached the end token
        if any(next_vals[k] == end_id for k in ['v1', 'v2', 'l1', 'e', 'l2']):
            break

        # Optional stopping condition
        if any(next_vals[k] == pad_id for k in ['v1', 'v2', 'l1', 'e', 'l2']):
            break
        

    # Decode sequence to DFS code
    dfs_code = []
    for i in range(1, len(v1_seq) - 1):  # Skip the first token (start token) and last token (end token)
        try:
            v1 = int(node_id_vocab.get_token(v1_seq[i]))
            v2 = int(node_id_vocab.get_token(v2_seq[i]))
            l1 = node_label_vocab.get_token(l1_seq[i])
            e  = edge_label_vocab.get_token(e_seq[i])
            l2 = node_label_vocab.get_token(l2_seq[i])
            dfs_code.append((v1, v2, l1, e, l2))
        except:
            break

    # Convert to NetworkX graph
    graph = graph_from_dfscode(dfs_code)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    if len(graph.nodes):
        max_comp = max(nx.connected_components(graph), key=len)
        graph = nx.Graph(graph.subgraph(max_comp))

    return graph