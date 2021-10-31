import numpy as np
import torch
import torch.nn.functional as f
import utils
from torch import nn
from scipy.sparse import vstack
from scipy.sparse import lil_matrix
from ACDNE_model import ACDNE
from flip_gradient import GradReverse


def train_and_evaluate(input_data, config, random_state=0):
    # ##get input data
    ppmi_s = input_data['PPMI_S']
    ppmi_t = input_data['PPMI_T']
    x_s = input_data['attrb_S']
    x_t = input_data['attrb_T']
    x_n_s = input_data['attrb_nei_S']
    x_n_t = input_data['attrb_nei_T']
    y_s = input_data['label_S']
    y_t = input_data['label_T']
    y_t_o = np.zeros(np.shape(y_t))  # observable label matrix of target network, all zeros
    x_s_new = lil_matrix(np.concatenate((lil_matrix.toarray(x_s), x_n_s), axis=1))
    x_t_new = lil_matrix(np.concatenate((lil_matrix.toarray(x_t), x_n_t), axis=1))
    n_input = x_s.shape[1]
    num_class = y_s.shape[1]
    num_nodes_s = x_s.shape[0]
    num_nodes_t = x_t.shape[0]
    # ##model config
    clf_type = config['clf_type']
    dropout = config['dropout']
    num_epoch = config['num_epoch']
    batch_size = config['batch_size']
    n_hidden = config['n_hidden']
    n_emb = config['n_emb']
    l2_w = config['l2_w']
    net_pro_w = config['net_pro_w']
    # emb_filename = config['emb_filename']
    lr_ini = config['lr_ini']
    whole_xs_xt_stt = torch.FloatTensor(vstack([x_s, x_t]).toarray())
    whole_xs_xt_stt_nei = torch.FloatTensor(vstack([x_n_s, x_n_t]).toarray())
    torch.manual_seed(random_state)
    torch.cuda.manual_seed(random_state)if torch.cuda.is_available()else None
    np.random.seed(random_state)
    model = ACDNE(n_input, n_hidden, n_emb, num_class, clf_type, batch_size, dropout)
    clf_loss_f = nn.BCEWithLogitsLoss(reduction='none')if model.node_classifier.clf_type == 'multi-label'\
        else nn.CrossEntropyLoss()
    domain_loss_f = nn.CrossEntropyLoss()
    f1_t = None
    for cEpoch in range(num_epoch):
        s_batches = utils.batch_generator([x_s_new, y_s], int(batch_size / 2), shuffle=True)
        t_batches = utils.batch_generator([x_t_new, y_t_o], int(batch_size / 2), shuffle=True)
        num_batch = round(max(num_nodes_s / (batch_size / 2), num_nodes_t / (batch_size / 2)))
        # Adaptation param and learning rate schedule as described in the DANN paper
        p = float(cEpoch) / num_epoch
        lr = lr_ini / (1. + 10 * p) ** 0.75
        grl_lambda = 2. / (1. + np.exp(-10. * p)) - 1  # gradually change from 0 to 1
        GradReverse.rate = grl_lambda
        optimizer = torch.optim.SGD(model.parameters(), lr, 0.9, weight_decay=l2_w/2)
        # optimizer = torch.optim.SGD(model.parameters(), lr, weight_decay=l2_w/2)
        # optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=l2_w/2)
        # #in each epoch, train all the mini batches
        for cBatch in range(num_batch):
            # ## each batch, half nodes from source network, and half nodes from target network
            xs_ys_batch, shuffle_index_s = next(s_batches)
            xs_batch = xs_ys_batch[0]
            ys_batch = xs_ys_batch[1]
            xt_yt_batch, shuffle_index_t = next(t_batches)
            xt_batch = xt_yt_batch[0]
            yt_batch = xt_yt_batch[1]
            x_batch = vstack([xs_batch, xt_batch])
            batch_csr = x_batch.tocsr()
            xb = torch.FloatTensor(batch_csr[:, 0:n_input].toarray())
            xb_nei = torch.FloatTensor(batch_csr[:, -n_input:].toarray())
            yb = np.vstack([ys_batch, yt_batch])
            mask_l = np.sum(yb, axis=1) > 0
            # 1 if the node is with observed label, 0 if the node is without label
            domain_label = np.vstack([np.tile([1., 0.], [batch_size // 2, 1]), np.tile([0., 1.], [
                batch_size // 2, 1])])  # [1,0] for source, [0,1] for target
            # #topological proximity matrix between nodes in each mini-batch
            a_s, a_t = utils.batch_ppmi(batch_size, shuffle_index_s, shuffle_index_t, ppmi_s, ppmi_t)
            model.train()
            optimizer.zero_grad()
            emb, pred_logit, d_logit = model(xb, xb_nei)
            emb_s, emb_t = model.network_embedding.pairwise_constraint(emb)
            net_pro_loss_s = model.network_embedding.net_pro_loss(emb_s, a_s)
            net_pro_loss_t = model.network_embedding.net_pro_loss(emb_t, a_t)
            net_pro_loss = net_pro_w*(net_pro_loss_s+net_pro_loss_t)
            if clf_type == 'multi-class':
                clf_loss = clf_loss_f(pred_logit[mask_l], torch.argmax(torch.FloatTensor(yb[mask_l]), 1))
            else:
                clf_loss = clf_loss_f(pred_logit[mask_l], torch.FloatTensor(yb[mask_l]))
                clf_loss = torch.sum(clf_loss)/np.sum(mask_l)
            domain_loss = domain_loss_f(d_logit, torch.argmax(torch.FloatTensor(domain_label), 1))
            total_loss = clf_loss+domain_loss+net_pro_loss
            total_loss.backward()
            optimizer.step()
        '''Compute evaluation on test data by the end of each epoch'''
        model.eval()  # deactivates dropout during validation run.
        with torch.no_grad():
            GradReverse.rate = 1.0
            _, pred_logit_xs_xt, _ = model(whole_xs_xt_stt, whole_xs_xt_stt_nei)
        pred_prob_xs_xt = f.sigmoid(pred_logit_xs_xt)if clf_type == 'multi-label'else f.softmax(pred_logit_xs_xt)
        pred_prob_xs = pred_prob_xs_xt[0:num_nodes_s, :]
        pred_prob_xt = pred_prob_xs_xt[-num_nodes_t:, :]
        print('epoch: ', cEpoch + 1)
        f1_s = utils.f1_scores(pred_prob_xs, y_s)
        print('Source micro-F1: %f, macro-F1: %f' % (f1_s[0], f1_s[1]))
        f1_t = utils.f1_scores(pred_prob_xt, y_t)
        print('Target testing micro-F1: %f, macro-F1: %f' % (f1_t[0], f1_t[1]))
    ''' save final evaluation on test data by the end of all epoches'''
    micro = float(f1_t[0])
    macro = float(f1_t[1])
    # #save embedding features
    # #emb= sess.run(model.emb, feed_dict={model.X: whole_xs_xt_stt, model.X_nei:whole_xs_xt_stt_nei,
    # model.Ada_lambda:1.0, model.dropout:0.})
    # #hs=emb[0:num_nodes_S,:]
    # #ht=emb[-num_nodes_T:,:]
    # #print(np.shape(hs))
    # #print(np.shape(ht))
    # #scipy.io.savemat(emb_filename+'_emb.mat', {'rep_S':hs, 'rep_T':ht})
    return micro, macro
