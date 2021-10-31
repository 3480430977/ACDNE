import numpy as np
import torch
import utils
from scipy.sparse import vstack
from evalModel import train_and_evaluate
from scipy.sparse import lil_matrix
torch.manual_seed(0)
torch.cuda.manual_seed(0)if torch.cuda.is_available()else None
np.random.seed(0)
source = 'Blog1'
target = 'Blog2'
emb_filename = str(source)+'_'+str(target)
Kstep = 3
####################
# Load source data
####################
A_s, X_s, Y_s = utils.load_network('./input/'+str(source)+'.mat')
num_nodes_S = X_s.shape[0]
####################
# Load target data
####################
A_t, X_t, Y_t = utils.load_network('./input/'+str(target)+'.mat')
num_nodes_T = X_t.shape[0]
features = vstack((X_s, X_t))
features = utils.feature_compression(features, dim=1000)
X_s = features[0:num_nodes_S, :]
X_t = features[-num_nodes_T:, :]
'''compute PPMI'''
A_k_s = utils.agg_tran_prob_mat(A_s, Kstep)
PPMI_s = utils.compute_ppmi(A_k_s)
n_PPMI_s = utils.my_scale_sim_mat(PPMI_s)  # row normalized PPMI
X_n_s = np.matmul(n_PPMI_s, lil_matrix.toarray(X_s))  # neibors' attribute matrix
# noinspection DuplicatedCode
'''compute PPMI'''
A_k_t = utils.agg_tran_prob_mat(A_t, Kstep)
PPMI_t = utils.compute_ppmi(A_k_t)
n_PPMI_t = utils.my_scale_sim_mat(PPMI_t)  # row normalized PPMI
X_n_t = np.matmul(n_PPMI_t, lil_matrix.toarray(X_t))  # neibors' attribute matrix
# #input data
input_data = dict()
input_data['PPMI_S'] = PPMI_s
input_data['PPMI_T'] = PPMI_t
input_data['attrb_S'] = X_s
input_data['attrb_T'] = X_t
input_data['attrb_nei_S'] = X_n_s
input_data['attrb_nei_T'] = X_n_t
input_data['label_S'] = Y_s
input_data['label_T'] = Y_t
# ##model config
config = dict()
config['clf_type'] = 'multi-class'
config['dropout'] = 0.6
config['num_epoch'] = 100  # maximum training iteration
config['batch_size'] = 100
config['n_hidden'] = [512, 128]  # dimensionality for each hidden layer of FE1 and FE2
config['n_emb'] = 128  # embedding dimension d
config['l2_w'] = 1e-3  # weight of L2-norm regularization
config['lr_ini'] = 0.02  # initial learning rate
config['net_pro_w'] = 1e-3  # weight of pairwise constraint
config['emb_filename'] = emb_filename  # output file name to save node representations
numRandom = 5
microAllRandom = []
macroAllRandom = []
print('source and target networks:', str(source), str(target))
for random_state in range(numRandom):
    print("%d-th random initialization" % (random_state+1))
    micro_t, macro_t = train_and_evaluate(input_data, config, random_state)
    microAllRandom.append(micro_t)
    macroAllRandom.append(macro_t)
'''avg F1 scores over 5 random splits'''
micro = np.mean(microAllRandom)
macro = np.mean(macroAllRandom)
micro_sd = np.std(microAllRandom)
macro_sd = np.std(macroAllRandom)
print("The avergae micro and macro F1 scores over {} random initializations are:  {} +/- {} and {} +/- {}: ".format(
    numRandom, micro, micro_sd, macro, macro_sd))
