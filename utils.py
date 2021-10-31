import numpy as np
import scipy.io as sio
from scipy.sparse import csc_matrix
from scipy.sparse import lil_matrix
import scipy
from sklearn.metrics import f1_score
import scipy.sparse as sp
from sklearn.decomposition import PCA
from warnings import filterwarnings
filterwarnings('ignore')


def load_network(file):
    net = sio.loadmat(file)
    x, a, y = net['attrb'], net['network'], net['group']
    if not isinstance(x, scipy.sparse.lil_matrix):
        x = lil_matrix(x)
    return a, x, y


def feature_compression(features, dim=200):
    """Preprcessing of features"""
    features = features.toarray()
    feat = lil_matrix(PCA(n_components=dim, random_state=0).fit_transform(features))
    return feat


def my_scale_sim_mat(w):
    """L1 row norm of a matrix"""
    rowsum = np.array(np.sum(w, axis=1), dtype=np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    w = r_mat_inv.dot(w)
    return w


def agg_tran_prob_mat(g, step):
    """aggregated K-step transition probality"""
    g = my_scale_sim_mat(g)
    g = csc_matrix.toarray(g)
    a_k = g
    a = g
    for k in np.arange(2, step+1):
        a_k = np.matmul(a_k, g)
        a = a+a_k/k
    return a


def compute_ppmi(a):
    """compute PPMI, given aggregated K-step transition probality matrix as input"""
    np.fill_diagonal(a, 0)
    a = my_scale_sim_mat(a)
    (p, q) = np.shape(a)
    col = np.sum(a, axis=0)
    col[col == 0] = 1
    ppmi = np.log((float(p)*a)/col[None, :])
    idx_nan = np.isnan(ppmi)
    ppmi[idx_nan] = 0
    ppmi[ppmi < 0] = 0
    return ppmi


def shuffle_aligned_list(data):
    num = data[0].shape[0]
    shuffle_index = np.random.permutation(num)
    return shuffle_index, [d[shuffle_index] for d in data]


def batch_generator(data, batch_size, shuffle=True):
    shuffle_index = None
    if shuffle:
        shuffle_index, data = shuffle_aligned_list(data)
    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size >= data[0].shape[0]:
            batch_count = 0
            if shuffle:
                shuffle_index, data = shuffle_aligned_list(data)
        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield [d[start:end] for d in data], shuffle_index[start:end]


def batch_ppmi(batch_size, shuffle_index_s, shuffle_index_t, ppmi_s, ppmi_t):
    """return the PPMI matrix between nodes in each batch"""
    # #proximity matrix between source network nodes in each mini-batch
    # noinspection DuplicatedCode
    a_s = np.zeros((int(batch_size / 2), int(batch_size / 2)))
    for ii in range(int(batch_size / 2)):
        for jj in range(int(batch_size / 2)):
            if ii != jj:
                a_s[ii, jj] = ppmi_s[shuffle_index_s[ii], shuffle_index_s[jj]]
    # #proximity matrix between target network nodes in each mini-batch
    # noinspection DuplicatedCode
    a_t = np.zeros((int(batch_size / 2), int(batch_size / 2)))
    for ii in range(int(batch_size / 2)):
        for jj in range(int(batch_size / 2)):
            if ii != jj:
                a_t[ii, jj] = ppmi_t[shuffle_index_t[ii], shuffle_index_t[jj]]
    return my_scale_sim_mat(a_s), my_scale_sim_mat(a_t)


def f1_scores(y_pred, y_true):
    def predict(y_tru, y_pre):
        top_k_list = np.array(np.sum(y_tru, 1), np.int32)
        prediction = []
        for i in range(y_tru.shape[0]):
            pred_i = np.zeros(y_tru.shape[1])
            pred_i[np.argsort(y_pre[i, :])[-top_k_list[i]:]] = 1
            prediction.append(np.reshape(pred_i, (1, -1)))
        prediction = np.concatenate(prediction, axis=0)
        return np.array(prediction, np.int32)
    results = {}
    predictions = predict(y_true, y_pred)
    averages = ["micro", "macro"]
    for average in averages:
        results[average] = f1_score(y_true, predictions, average=average)
    return results["micro"], results["macro"]
