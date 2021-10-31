import numpy as np
import torch
import utils
from evalModel import train_and_evaluate
from scipy.sparse import lil_matrix
torch.manual_seed(0)
torch.cuda.manual_seed(0)if torch.cuda.is_available()else None
np.random.seed(0)
source = 'citationv1'
target = 'dblpv7'
emb_filename = str(source)+'_'+str(target)
Kstep = 3
####################
# Load source data
####################
A_s, X_s, Y_s = utils.load_network('./input/'+str(source)+'.mat')
'''compute PPMI'''
A_k_s = utils.agg_tran_prob_mat(A_s, Kstep)
PPMI_s = utils.compute_ppmi(A_k_s)
n_PPMI_s = utils.my_scale_sim_mat(PPMI_s)  # row normalized PPMI
X_n_s = np.matmul(n_PPMI_s, lil_matrix.toarray(X_s))  # neibors' attribute matrix
####################
# Load target data
####################
# noinspection DuplicatedCode
A_t, X_t, Y_t = utils.load_network('./input/'+str(target)+'.mat')
'''compute PPMI'''
A_k_t = utils.agg_tran_prob_mat(A_t, Kstep)
PPMI_t = utils.compute_ppmi(A_k_t)
n_PPMI_t = utils.my_scale_sim_mat(PPMI_t)  # row normalized PPMI
X_n_t = np.matmul(n_PPMI_t, lil_matrix.toarray(X_t))  # neibors' attribute matrix
# #input data
# noinspection DuplicatedCode
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
config['clf_type'] = 'multi-label'
config['dropout'] = 0.5
config['num_epoch'] = 30  # maximum training iteration
config['batch_size'] = 100
config['n_hidden'] = [512, 128]  # dimensionality for each k-th hidden layer of FE1 and FE2
config['n_emb'] = 128  # embedding dimension d
config['l2_w'] = 1e-3  # weight of L2-norm regularization
config['net_pro_w'] = 0.1  # weight of pairwise constraint
config['emb_filename'] = emb_filename  # output file name to save node representations
config['lr_ini'] = 0.02  # initial learning rate
numRandom = 5
microAllRandom = []
macroAllRandom = []
print('source and target networks:', str(source), str(target))
for random_state in range(numRandom):
    print("%d-th random initialization " % (random_state+1))
    micro_t, macro_t = train_and_evaluate(input_data, config, random_state)
    microAllRandom.append(micro_t)
    macroAllRandom.append(macro_t)
'''avg F1 scores over 5 random splits'''
micro = np.mean(microAllRandom)
macro = np.mean(macroAllRandom)
micro_sd = np.std(microAllRandom)
macro_sd = np.std(macroAllRandom)
print("The avergae micro and macro F1 scores over {} random initializations are:  {} +/- {} and {} +/- {}: " .format(
    numRandom, micro, micro_sd, macro, macro_sd))
