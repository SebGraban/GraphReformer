import torch
import numpy as np

import networkx as nx


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def blanket_seq(G, start_id):
    '''
    get a blanket node sequence
    :param G:
    :param start_id:
    :return:
    '''
    visited = set()
    blanket = set([start_id])
    output = []

    while len(blanket) > 0:
        next = np.random.choice(list(blanket))
        output.append(next)
        blanket.remove(next)
        visited.update([next])
        blanket.update( set(G[next].keys()) - visited)

    return output



def bfs_seq(G, start_id):
    '''
    get a bfs node sequence
    :param G:
    :param start_id:
    :return:
    '''
    dictionary = dict(nx.bfs_successors(G, start_id))
    start = [start_id]
    output = [start_id]
    while len(start) > 0:
        next = []
        while len(start) > 0:
            current = start.pop(0)
            neighbor = dictionary.get(current)
            if neighbor is not None:
                #### a wrong example, should not permute here!
                # shuffle(neighbor)
                next = next + neighbor
        output = output + next
        start = next
    return output



def encode_adj(adj, max_prev_node=10, is_full = False):
    '''

    :param adj: n*n, rows means time step, while columns are input dimension
    :param max_degree: we want to keep row number, but truncate column numbers
    :return:
    '''
    if is_full:
        max_prev_node = adj.shape[0]-1

    # pick up lower tri
    adj = np.tril(adj, k=-1)
    n = adj.shape[0]
    adj = adj[1:n, 0:n-1]

    # use max_prev_node to truncate
    # note: now adj is a (n-1)*(n-1) matrix
    adj_output = np.zeros((adj.shape[0], max_prev_node))
    for i in range(adj.shape[0]):
        input_start = max(0, i - max_prev_node + 1)
        input_end = i + 1
        output_start = max_prev_node + input_start - input_end
        output_end = max_prev_node
        adj_output[i, output_start:output_end] = adj[i, input_start:input_end]
        adj_output[i,:] = adj_output[i,:][::-1] # reverse order

    return adj_output

def decode_adj(adj_output):
    '''
        recover to adj from adj_output
        note: here adj_output have shape (n-1)*m
    '''
    max_prev_node = adj_output.shape[1]
    adj = np.zeros((adj_output.shape[0], adj_output.shape[0]))
    for i in range(adj_output.shape[0]):
        input_start = max(0, i - max_prev_node + 1)
        input_end = i + 1
        output_start = max_prev_node + max(0, i - max_prev_node + 1) - (i + 1)
        output_end = max_prev_node
        adj[i, input_start:input_end] = adj_output[i,::-1][output_start:output_end] # reverse order
    adj_full = np.zeros((adj_output.shape[0]+1, adj_output.shape[0]+1))
    n = adj_full.shape[0]
    adj_full[1:n, 0:n-1] = np.tril(adj, 0)
    adj_full = adj_full + adj_full.T

    return adj_full


def my_decode_adj(generated_seq, args):
    '''
        recover to adj from nodes and edges
    '''

    if args.input_type == 'node_based':
        print('@@ ', generated_seq)
        adj = np.zeros([args.max_num_node, args.max_num_node])
        n = 0
        #assert generated_seq[0] == args.max_num_node + 1 ## add_node
        if generated_seq[0] != args.max_num_node + 1:
            print('      __ERR: first word is ', generated_seq[0])
            generated_seq[0] = args.max_num_node + 1
        if generated_seq[1] != args.max_num_node + 1:
            print('      __ERR: second word is ', generated_seq[1])
            generated_seq[1] = args.max_num_node + 1

        for i in range(generated_seq.size):
            if generated_seq[i] == args.max_num_node + 2: ## terminate
                break
            # assert not (i > 1 and generated_seq[i] == generated_seq[i-1] == args.max_num_node + 1)
            if (i > 1 and generated_seq[i] == generated_seq[i-1] == args.max_num_node + 1):
                print('      __ERR: ignoring an orphan node at position', i)
                continue
            if generated_seq[i] ==  args.max_num_node + 1: ## add_node
                n += 1
                if n == args.max_num_node:
                    break
                continue
            ## assert generated_seq[i] > 0
            if generated_seq[i] == 0:
                print('      __ERR: ignoring word 0 at position', i)
                continue

            j = generated_seq[i] - 1
            ## assert j < n-1
            ## assert adj[n-1,j] == 0
            if j >= n-1:
                print('      __ERR: edge to a prospective node changed to edge to a preceding node.')
                j = np.random.randint(0, n-1)
            elif adj[n-1,j] != 0:
                print('      __ERR: ignoring duplicate edge')
            adj[n-1,j] = adj[j,n-1] = 1
        adj = adj[:n, :n]
    elif args.input_type == 'preceding_neighbors_vector':
        if generated_seq[0, 0] == 1:
            print('       __ERR: empty graph.')
            generated_seq[0, 0] = 0
        if generated_seq[1, 0] == 1:
            print('       __ERR: single node graph.')
            generated_seq[1, 0] = 0
        if generated_seq[1, 1] == 0:
            print('       __ERR: no edge.')
            generated_seq[1, 1] = 1
        adj = np.zeros([args.max_num_node, args.max_num_node])
        for i in range(generated_seq.shape[0] - 1):
            if generated_seq[i, 0] == 1:
                adj = adj[:i, :i]
                break
            adj[i, :i] = generated_seq[i, 1:i+1]
            adj[:i, i] = generated_seq[i, 1:i+1]
    else:
        raise NotImplementedError

    return adj


def encode_adj_flexible(adj):
    '''
    return a flexible length of output
    note that here there is no loss when encoding/decoding an adj matrix
    :param adj: adj matrix
    :return:
    '''
    # pick up lower tri
    adj = np.tril(adj, k=-1)
    n = adj.shape[0]
    adj = adj[1:n, 0:n-1]

    adj_output = []
    input_start = 0
    for i in range(adj.shape[0]):
        input_end = i + 1
        adj_slice = adj[i, input_start:input_end]
        adj_output.append(adj_slice)
        non_zero = np.nonzero(adj_slice)[0]
        input_start = input_end-len(adj_slice)+np.amin(non_zero)

    return adj_output



def decode_adj_flexible(adj_output):
    '''
    return a flexible length of output
    note that here there is no loss when encoding/decoding an adj matrix
    :param adj: adj matrix
    :return:
    '''
    adj = np.zeros((len(adj_output), len(adj_output)))
    for i in range(len(adj_output)):
        output_start = i+1-len(adj_output[i])
        output_end = i+1
        adj[i, output_start:output_end] = adj_output[i]
    adj_full = np.zeros((len(adj_output)+1, len(adj_output)+1))
    n = adj_full.shape[0]
    adj_full[1:n, 0:n-1] = np.tril(adj, 0)
    adj_full = adj_full + adj_full.T

    return adj_full


def encode_adj_full(adj):
    '''
    return a n-1*n-1*2 tensor, the first dimension is an adj matrix, the second show if each entry is valid
    :param adj: adj matrix
    :return:
    '''
    # pick up lower tri
    adj = np.tril(adj, k=-1)
    n = adj.shape[0]
    adj = adj[1:n, 0:n-1]
    adj_output = np.zeros((adj.shape[0],adj.shape[1],2))
    adj_len = np.zeros(adj.shape[0])

    for i in range(adj.shape[0]):
        non_zero = np.nonzero(adj[i,:])[0]
        input_start = np.amin(non_zero)
        input_end = i + 1
        adj_slice = adj[i, input_start:input_end]
        # write adj
        adj_output[i,0:adj_slice.shape[0],0] = adj_slice[::-1] # put in reverse order
        # write stop token (if token is 0, stop)
        adj_output[i,0:adj_slice.shape[0],1] = 1 # put in reverse order
        # write sequence length
        adj_len[i] = adj_slice.shape[0]

    return adj_output,adj_len

def decode_adj_full(adj_output):
    '''
    return an adj according to adj_output
    :param
    :return:
    '''
    # pick up lower tri
    adj = np.zeros((adj_output.shape[0]+1,adj_output.shape[1]+1))

    for i in range(adj_output.shape[0]):
        non_zero = np.nonzero(adj_output[i,:,1])[0] # get valid sequence
        input_end = np.amax(non_zero)
        adj_slice = adj_output[i, 0:input_end+1, 0] # get adj slice
        # write adj
        output_end = i+1
        output_start = i+1-input_end-1
        adj[i+1,output_start:output_end] = adj_slice[::-1] # put in reverse order
    adj = adj + adj.T
    return adj


class MyGraph_sequence_sampler_pytorch(torch.utils.data.Dataset):
    def __init__(self, G_list, args, max_num_node=None, max_prev_node=None, iteration=20000):
        self.input_type = args.input_type
        self.adj_all = []
        self.len_all = []
        self.args = args
        for G in G_list:
            self.adj_all.append(np.asarray(nx.to_numpy_array(G)))
            self.len_all.append(G.number_of_nodes())
        if max_num_node is None:
            self.n = max(self.len_all)
        else:
            self.n = max_num_node
        if max_prev_node is None:
            print('calculating max previous node, total iteration: {}'.format(iteration))
            self.max_prev_node = max(self.calc_max_prev_node(iter=iteration))
            print('max previous node: {}'.format(self.max_prev_node))
        else:
            self.max_prev_node = max_prev_node

        num_e = 0
        for i in range(self.n):
            num_e += min(self.max_prev_node, i)
        self.e = num_e

        if self.input_type == 'node_based':
            self.max_seq_len = self.n + self.e + 2 # self.n add_node charachters, self.e node_idx charachters,
                                                   # 1 termination charachter and 1 for positional shift of the source sequence
        elif self.input_type in ['preceding_neighbors_vector', 'max_prev_node_neighbors_vec' ]:
            self.max_seq_len = self.n + 2 # 1 for positional shift of the source sequence and 1 for termination bit
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.adj_all)
    def __getitem__(self, idx):
        adj_copy = self.adj_all[idx].copy()
        # generate input x, y pairs
        len_batch = adj_copy.shape[0]
        x_idx = np.random.permutation(adj_copy.shape[0])
        adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
        adj_copy_matrix = np.asmatrix(adj_copy)
        G = nx.from_numpy_array(adj_copy_matrix)
        # then do bfs in the permuted G
        start_idx = np.random.randint(adj_copy.shape[0])
        if self.args.node_ordering == 'bfs':
            x_idx = np.array(bfs_seq(G, start_idx))
        elif self.args.node_ordering == 'blanket':
            x_idx = np.array(blanket_seq(G, start_idx))
        else:
            raise NotImplementedError
        adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
        if adj_copy.shape[0] != len_batch:
            len_batch = adj_copy.shape[0]
        adj_copy_z = adj_copy.copy()
        np.fill_diagonal(adj_copy_z, 0)

        if self.input_type == 'node_based':
            trg_seq = np.zeros(self.max_seq_len, dtype=np.long)
            src_seq = np.zeros(self.max_seq_len, dtype=np.long)

            head = 0
            for i in range(len_batch):
                trg_seq[head] = self.n + 1  # add node
                head += 1
                nd_idx = np.where(adj_copy[:i,i])[0] + 1
                sz = nd_idx.size
                if sz > 0:
                    trg_seq[head:head+sz] = nd_idx
                    head += sz

            trg_seq[head] = self.n + 2  # terminate
            head += 1

            src_seq[1:] = trg_seq[:-1].copy()
        elif self.input_type == 'preceding_neighbors_vector':
            trg_seq = self.args.trg_pad_idx * np.ones([self.max_seq_len, self.n + 1], dtype=np.float32)
            src_seq = self.args.src_pad_idx * np.ones([self.max_seq_len, self.n + 1], dtype=np.float32)
            assert self.args.src_pad_idx == self.args.trg_pad_idx
            for i in range(len_batch):
                tmp = adj_copy[i, :].copy()
                ind_0 = tmp == 0
                ind_1 = tmp == 1
                tmp[ind_0] = self.args.zero_input
                tmp[ind_1] = self.args.one_input
                trg_seq[i,1:adj_copy.shape[1] + 1] = tmp
                trg_seq[i, 0] = self.args.zero_input     # termination bit
                trg_seq[i, i+1:] = self.args.dontcare_input
                if self.args.use_max_prev_node and i > self.args.max_prev_node:
                    trg_seq[i, 1:i - self.args.max_prev_node + 1] = self.args.dontcare_input
            if not self.args.use_termination_bit:
                trg_seq[len_batch, :len_batch + 1] = self.args.zero_input
                trg_seq[len_batch, len_batch + 1:] = self.args.dontcare_input
            else:
                trg_seq[len_batch, :] = self.args.trg_pad_idx
                trg_seq[len_batch, 0] = self.args.one_input     # termination bit
            src_seq[1:, :] = trg_seq[:-1, :].copy()
            if self.args.ensemble_input_type == 'negative':
                ind_0 = src_seq == self.args.zero_input
                ind_1 = src_seq == self.args.one_input
                src_seq = np.tile(src_seq.reshape([self.max_seq_len, 1, self.n + 1]), [1, 2, 1] )
                src_seq[:, 1, :][ind_0] = self.args.one_input
                src_seq[:, 1, :][ind_1] = self.args.zero_input
                src_seq[:, 1, 0] = src_seq[:, 0, 0]
            elif self.args.ensemble_input_type in ['multihop', 'multihop-single']:
                if self.args.ensemble_input_type == 'multihop':
                    assert self.args.n_ensemble == len(self.args.ensemble_multihop) + 1
                if self.args.ensemble_input_type == 'multihop-single':
                    assert self.args.n_ensemble == 1
                src_seq = np.tile(src_seq.reshape([self.max_seq_len, 1, self.n + 1]), [1, len(self.args.ensemble_multihop) + 1, 1])
                hops = sorted(self.args.ensemble_multihop)
                k = 1
                p = np.tril(adj_copy_z, -1)
                for j in range(len(hops)):
                    h = hops[j]
                    while k < h:
                        k += 1
                        p = np.tril(np.matmul(p, adj_copy_z), 0)
                    for i in range(len_batch):
                        src_seq[i+1, j + 1, 0] = self.args.zero_input
                        tmp = p[i,:i+1].copy()
                        ind_0 = tmp == 0
                        tmp[ind_0] = self.args.zero_input
                        src_seq[i+1, j + 1, 1:i+2] = tmp

            if self.args.ensemble_input_type == 'multihop-single':
                src_seq = src_seq.reshape([self.max_seq_len, -1])
        elif self.input_type == 'max_prev_node_neighbors_vec':
            trg_seq = self.args.trg_pad_idx * np.ones([self.max_seq_len, self.args.max_prev_node + 1], dtype=np.float32)
            src_seq = self.args.src_pad_idx * np.ones([self.max_seq_len, self.args.max_prev_node + 1], dtype=np.float32)
            assert self.args.src_pad_idx == self.args.trg_pad_idx
            for i in range(len_batch):
                tmp_1 = adj_copy[i, max(0, i-self.args.max_prev_node) : i].copy()
                ind_0 = tmp_1 == 0
                ind_1 = tmp_1 == 1
                tmp_1[ind_0] = self.args.zero_input
                tmp_1[ind_1] = self.args.one_input

                tmp = self.args.dontcare_input * np.ones([self.args.max_prev_node])
                if tmp_1.size > 0:
                    tmp[-tmp_1.size:] = tmp_1

                trg_seq[i,1:] = tmp
                trg_seq[i, 0] = self.args.zero_input     # termination bit
            trg_seq[len_batch, :] = self.args.trg_pad_idx
            trg_seq[len_batch, 0] = self.args.one_input     # termination bit
            src_seq[1:, :] = trg_seq[:-1, :].copy()

            if self.args.ensemble_input_type == 'negative':
                ind_0 = src_seq == self.args.zero_input
                ind_1 = src_seq == self.args.one_input
                src_seq = np.tile(src_seq.reshape([self.max_seq_len, 1, self.args.max_prev_node + 1]), [1, 2, 1] )
                src_seq[:, 1, :][ind_0] = self.args.one_input
                src_seq[:, 1, :][ind_1] = self.args.zero_input
                src_seq[:, 1, 0] = src_seq[:, 0, 0]
            elif self.args.ensemble_input_type == 'multihop-single':
                assert self.args.n_ensemble == 1
                extra_src_seq = self.args.src_pad_idx * \
                                np.ones([self.max_seq_len, len(self.args.ensemble_multihop), self.n], dtype=np.float32)
                hops = sorted(self.args.ensemble_multihop)
                k = 1
                p = np.tril(adj_copy_z, -1)
                for j in range(len(hops)):
                    h = hops[j]
                    while k < h:
                        k += 1
                        p = np.tril(np.matmul(p, adj_copy_z), 0)
                    for i in range(len_batch):
                        tmp = p[i,:i+1].copy()
                        ind_0 = tmp == 0
                        #ind_1 = tmp > 0  ### what to do with positive elements?
                        tmp[ind_0] = self.args.zero_input
                        extra_src_seq[i+1, j, :i+1] = tmp
                        extra_src_seq[i+1, j, i+1:] = self.args.dontcare_input
                src_seq = np.concatenate([src_seq, extra_src_seq.reshape([self.max_seq_len, -1])], axis=1)

        else:
            raise NotImplementedError

        if self.args.input_bfs_depth:
            depth = np.zeros([len_batch], dtype=np.long)
            for i in range(1, len_batch):
                ind = adj_copy[i,:i].astype(bool)
                depth[i] = depth[:i][ind].min() + 1

            dp_seq = np.zeros([self.n + 1, self.n], dtype=np.float32)
            for i in range(len_batch):
                dp_seq[i+1, depth[i]] = 1

            src_seq = np.concatenate([src_seq, dp_seq], axis=1)

        adj_expanded = np.zeros([src_seq.shape[0], src_seq.shape[0]], dtype=np.float32)
        adj_expanded[1:len_batch+1, 1:len_batch+1] = adj_copy_z

        return {'src_seq': src_seq, 'trg_seq': trg_seq, 'adj': adj_expanded}

    def calc_max_prev_node(self, iter=20000,topk=10):
        max_prev_node = []
        for i in range(iter):
            if i % (iter / 5) == 0:
                print('iter {} times'.format(i))
            adj_idx = np.random.randint(len(self.adj_all))
            adj_copy = self.adj_all[adj_idx].copy()
            # print('Graph size', adj_copy.shape[0])
            x_idx = np.random.permutation(adj_copy.shape[0])
            adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
            adj_copy_matrix = np.asmatrix(adj_copy)
            G = nx.from_numpy_array(adj_copy_matrix)
            # then do bfs in the permuted G
            start_idx = np.random.randint(adj_copy.shape[0])
            x_idx = np.array(bfs_seq(G, start_idx))
            adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
            # encode adj
            adj_encoded = encode_adj_flexible(adj_copy.copy())
            max_encoded_len = max([len(adj_encoded[i]) for i in range(len(adj_encoded))])
            max_prev_node.append(max_encoded_len)
        max_prev_node = sorted(max_prev_node)[-1*topk:]
        return max_prev_node
