import faiss
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import torch


def build_knn(feats, k):
    # faiss.omp_set_num_threads(threads)

    feats = feats.astype('float32')
    size, dim = feats.shape
    index = faiss.IndexFlatIP(dim)
    index.add(feats)
    sims, nbrs = index.search(feats, k=k)
    knns = [(np.array(nbr, dtype=np.int32),
             1 - np.minimum(np.maximum(np.array(sim, dtype=np.float32), 0), 1))
            for nbr, sim in zip(nbrs, sims)]
    return knns


def row_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    # if rowsum <= 0, keep its previous value
    rowsum[rowsum <= 0] = 1
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_indices_values(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    values = sparse_mx.data
    shape = np.array(sparse_mx.shape)
    return indices, values, shape


def label2spmat(label, load=True):
    from scipy.sparse import csr_matrix
    n = len(label)

    row = np.array([])
    col = np.array([])
    if load != True:
        row_temp = np.array([])
        col_temp = np.array([])
        for idx in range(n):
            row_nodeid = np.where(label[idx] == label)[0]
            row_temp = np.concatenate((row_temp, row_nodeid), axis=-1)

            col_nodeid = np.full((len(row_nodeid)), idx)
            col_temp = np.concatenate((col_temp, col_nodeid), axis=-1)
            if idx % 10000 == 0 or idx + 1 == n:
                row = np.concatenate((row, row_temp), axis=-1)
                col = np.concatenate((col, col_temp), axis=-1)
                row_temp = np.array([])
                col_temp = np.array([])
                print(idx)

        np.save("./../data/row.npy", row)
        np.save("./../data/col.npy", col)
        print("Save Finish")
    else:
        row = np.load("./../data/row.npy")
        col = np.load("./../data/col.npy")

    data = np.ones(shape=(row.shape))
    spmat = csr_matrix((data, (row, col)), shape=(n, n))

    return spmat


def knn2spmat(dists, nbrs, k, th_sim, use_sim, self_loop):
    eps = 1e-2
    n = len(nbrs)
    if use_sim:
        sims = 1. - dists
    row, col = np.where(sims >= th_sim)
    # remove self-loop
    idxs = np.where(row != nbrs[row, col])
    row = row[idxs]
    col = col[idxs]
    data = sims[row, col]
    col = nbrs[row, col]
    adj = csr_matrix((data, (row, col)), shape=(n, n))
    # make symmetric adj
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    if self_loop:
        adj = adj + sp.eye(adj.shape[0])
    # div row sum to norm
    adj = row_normalize(adj)
    # sparse_mx2torch_sparse
    indices, values, shape = sparse_mx_to_indices_values(adj)
    indices = torch.from_numpy(indices)
    values = torch.from_numpy(values)
    shape = torch.Size(shape)

    return torch.sparse.FloatTensor(indices, values, shape)


def fast_knns2spmat(knns, k, th_sim=0.7, active_connection=10, use_sim=False, fill_value=None):
    # convert knns to symmetric sparse matrix
    from scipy.sparse import csr_matrix
    eps = 1e-5
    n = len(knns)
    if isinstance(knns, list):
        knns = np.array(knns)
    if len(knns.shape) == 2:
        # knns saved by hnsw has different shape
        n = len(knns)
        ndarr = np.ones([n, 2, k])
        ndarr[:, 0, :] = -1  # assign unknown dist to 1 and nbr to -1
        for i, (nbr, dist) in enumerate(knns):
            size = len(nbr)
            assert size == len(dist)
            ndarr[i, 0, :size] = nbr[:size]
            ndarr[i, 1, :size] = dist[:size]
        knns = ndarr
    nbrs = knns[:, 0, :active_connection]
    dists = knns[:, 1, :active_connection]
    assert -eps <= dists.min() <= dists.max(
    ) <= 1 + eps, "min: {}, max: {}".format(dists.min(), dists.max())
    if use_sim:
        sims = 1. - dists
    else:
        sims = dists
    if fill_value is not None:
        print('[fast_knns2spmat] edge fill value:', fill_value)
        sims.fill(fill_value)
    row, col = np.where(sims >= th_sim)
    # remove the self-loop
    idxs = np.where(row != nbrs[row, col])
    row = row[idxs]
    col = col[idxs]
    data = sims[row, col]
    col = nbrs[row, col]  # convert to absolute column
    assert len(row) == len(col) == len(data)
    spmat = csr_matrix((data, (row, col)), shape=(n, n))
    return spmat


def knn2mat(knns, k, use_sim, self_loop):
    eps = 1e-2
    n = len(knns)
    if isinstance(knns, list):
        knns = np.array(knns)
    nbrs = knns[:, 0, :]
    dists = knns[:, 1, :]
    if use_sim:
        sims = 1. - dists
    row, col = np.where(sims >= 0)
    idxs = np.where(row != nbrs[row, col])
    row = row[idxs]
    col = col[idxs]
    data = sims[row, col]
    col = nbrs[row, col]
    adj = csr_matrix((data, (row, col)), shape=(n, n))
    # make symmetric adj
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    if self_loop:
        adj = adj + sp.eye(adj.shape[0])
    # div row sum to norm
    adj = row_normalize(adj)
    adj = adj.todense().A

    return adj


def build_symmetric_adj(adj, self_loop=True):
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    if self_loop:
        adj = adj + sp.eye(adj.shape[0])
    return adj


def knns2ordered_nbrs(knns, sort=True):
    if isinstance(knns, list):
        knns = np.array(knns)
    nbrs = knns[:, 0, :].astype(np.int32)
    dists = knns[:, 1, :]
    if sort:
        # sort dists from low to high
        nb_idx = np.argsort(dists, axis=1)
        idxs = np.arange(nb_idx.shape[0]).reshape(-1, 1)
        dists = dists[idxs, nb_idx]
        nbrs = nbrs[idxs, nb_idx]
    return dists, nbrs
