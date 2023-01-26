# coding:utf-8
import sys
import numpy as np
from multiprocessing import Pool, Manager
import copy
from functools import partial
import os.path as osp
import gc


def get_docnodeid_list(query_nodeid):
    hops = list()
    hops.append(set(I[query_nodeid, :k]))
    hops.append(set())
    for h in hops[-2]:
        hops[-1].update(set(I[h, :10]))

    hops_set = set([h for hop in hops for h in hop])
    node_list = list(hops_set)
    return node_list


def worker(k, feature, queue, query_nodeid):
    docnodeid_list = get_docnodeid_list(query_nodeid)

    query_Rstarset = Rstarset_list[query_nodeid]
    outlist = []
    for idx, doc_nodeid in enumerate(docnodeid_list):
        doc_Rstarset = Rstarset_list[doc_nodeid]
        sim = 1.0 * len(query_Rstarset & doc_Rstarset) / len(query_Rstarset | doc_Rstarset)
        jd = sim
        nd = jd
        tpl = (doc_nodeid, nd)
        outlist.append(tpl)
    outlist = sorted(outlist, key=lambda x: x[1], reverse=True)[:k]
    queue.put(query_nodeid)

    fn_name = sys._getframe().f_code.co_name
    if queue.qsize() % 1000 == 0:
        print("==>", fn_name, queue.qsize())
    return list(zip(*outlist))


def get_Kngbr(query_nodeid, k):
    Kngbr = I[int(query_nodeid), :k]
    return set(Kngbr)


def get_Rset(k, queue, query_nodeid):
    docnodeid_set = get_Kngbr(query_nodeid, k)
    Rset = set()
    for doc_nodeid in docnodeid_set:
        if query_nodeid not in get_Kngbr(doc_nodeid, k):
            continue
        Rset.add(doc_nodeid)
    queue.put(query_nodeid)
    fn_name = sys._getframe().f_code.co_name
    if queue.qsize() % 1000 == 0:
        print("==>", fn_name, queue.qsize())
    return Rset


def get_Rstarset(queue, query_nodeid):
    Rset = Rset_list[query_nodeid]
    Rstarset = copy.deepcopy(Rset)
    for doc_nodeid in Rset:
        doc_Rset = half_Rset_list[int(doc_nodeid)]
        if len(doc_Rset & Rset) >= len(doc_Rset) * 2 / 3:
            Rstarset |= doc_Rset
    queue.put(query_nodeid)
    fn_name = sys._getframe().f_code.co_name
    if queue.qsize() % 1000 == 0:
        print("==>", fn_name, queue.qsize())
    return Rstarset


def read_probs(path, inst_num, feat_dim, dtype=np.float32, verbose=False):
    assert (inst_num > 0 or inst_num == -1) and feat_dim > 0
    count = -1
    if inst_num > 0:
        count = inst_num * feat_dim
    probs = np.fromfile(path, dtype=dtype, count=count)
    if feat_dim > 1:
        probs = probs.reshape(inst_num, feat_dim)
    if verbose:
        print('[{}] shape: {}'.format(path, probs.shape))
    return probs


if __name__ == "__main__":
    # train_name = 'part0_train'
    # k = 80

    k, train_name = sys.argv[1], sys.argv[2]
    print("use topk", k)
    print("use name", train_name)

    data_path = './data'
    feat_path = osp.join(data_path, 'features', '{}.bin'.format(train_name))
    label_path = osp.join(data_path, 'labels', '{}.meta'.format(train_name))
    knn_path = osp.join(data_path, 'knns', train_name, '{}_k_{}.npz'.format('faiss', 80))
    feature_dim = 256

    K80 = np.load(knn_path)['data']
    I = K80[:, 0, :].astype(int)
    D = K80[:, 1, :]

    feature = read_probs(feat_path, len(I), feature_dim)
    workers = 20

    queue1 = Manager().Queue()
    queue2 = Manager().Queue()
    queue3 = Manager().Queue()
    queue4 = Manager().Queue()
    debug = True
    debug = False
    if debug:
        for query_nodeid in range(len(I)):
            res = worker(k, query_nodeid)
    else:
        pool = Pool(workers)
        get_Rset_partial = partial(get_Rset, k, queue1)
        Rset_list = pool.map(get_Rset_partial, range(len(I)))
        pool.close()
        pool.join()

        pool = Pool(workers)
        k2 = k // 2
        get_Rset_partial = partial(get_Rset, k2, queue2)
        half_Rset_list = pool.map(get_Rset_partial, range(len(I)))
        pool.close()
        pool.join()

        pool = Pool(workers)
        get_Rstarset_partial = partial(get_Rstarset, queue3)
        Rstarset_list = pool.map(get_Rstarset_partial, range(len(I)))
        pool.close()
        pool.join()

        pool = Pool(workers)
        worker_partial = partial(worker, k, feature, queue4)
        res = pool.map(worker_partial, range(len(I)))
        pool.close()
        pool.join()

        newI, newD = list(zip(*res))
        newI = np.array(newI)
        newD = np.array(newD)
        newdata = np.concatenate((newI[:, None, :], newD[:, None, :]), axis=1)
        knn_path = osp.join(data_path, 'reranking_knn_{}.npy'.format(train_name))
        sim_path = osp.join(data_path, 'reranking_sim_{}.npy'.format(train_name))
        np.save(knn_path, newI)
        np.save(sim_path, newD)
        print("Finish")