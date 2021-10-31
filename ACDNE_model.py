from torch import nn
import torch
import torch.nn.functional as f
from flip_gradient import GRL


class FE1(nn.Module):
    def __init__(self, n_input, n_hidden, drop):
        super(FE1, self).__init__()
        self.drop = drop
        self.h1_self = nn.Linear(n_input, n_hidden[0])
        self.h2_self = nn.Linear(n_hidden[0], n_hidden[1])
        std = 1/(n_input/2)**0.5
        nn.init.trunc_normal_(self.h1_self.weight, std=std, a=-2*std, b=2*std)
        # nn.init.xavier_normal_(self.h1_self.weight, 1.414)
        nn.init.constant_(self.h1_self.bias, 0.1)
        std = 1/(n_hidden[0]/2)**0.5
        nn.init.trunc_normal_(self.h2_self.weight, std=std, a=-2*std, b=2*std)
        # nn.init.xavier_normal_(self.h2_self.weight, 1.414)
        nn.init.constant_(self.h2_self.bias, 0.1)

    def forward(self, x):
        x = f.dropout(f.relu(self.h1_self(x)), self.drop)
        return f.relu(self.h2_self(x))


class FE2(nn.Module):
    def __init__(self, n_input, n_hidden, drop):
        super(FE2, self).__init__()
        self.drop = drop
        self.h1_nei = nn.Linear(n_input, n_hidden[0])
        self.h2_nei = nn.Linear(n_hidden[0], n_hidden[1])
        std = 1/(n_input/2)**0.5
        nn.init.trunc_normal_(self.h1_nei.weight, std=std, a=-2*std, b=2*std)
        # nn.init.xavier_normal_(self.h1_nei.weight, 1.414)
        nn.init.constant_(self.h1_nei.bias, 0.1)
        std = 1/(n_hidden[0]/2)**0.5
        nn.init.trunc_normal_(self.h2_nei.weight, std=std, a=-2*std, b=2*std)
        # nn.init.xavier_normal_(self.h2_nei.weight, 1.414)
        nn.init.constant_(self.h2_nei.bias, 0.1)

    def forward(self, x_nei):
        x_nei = f.dropout(f.relu(self.h1_nei(x_nei)), self.drop)
        return f.relu(self.h2_nei(x_nei))


class NetworkEmbedding(nn.Module):
    def __init__(self, n_input, n_hidden, n_emb, drop, batch_size):
        super(NetworkEmbedding, self).__init__()
        self.drop = drop
        self.batch_size = batch_size
        self.fe1 = FE1(n_input, n_hidden, drop)
        self.fe2 = FE2(n_input, n_hidden, drop)
        self.emb = nn.Linear(n_hidden[-1]*2, n_emb)
        std = 1/(n_hidden[-1]*2)**0.5
        nn.init.trunc_normal_(self.emb.weight, std=std, a=-2*std, b=2*std)
        # nn.init.xavier_normal_(self.emb.weight, 1.414)
        nn.init.constant_(self.emb.bias, 0.1)

    def forward(self, x, x_nei):
        h2_self = self.fe1(x)
        h2_nei = self.fe2(x_nei)
        return f.relu(self.emb(torch.cat((h2_self, h2_nei), 1)))

    def pairwise_constraint(self, emb):
        emb_s = emb[:int(self.batch_size/2), :]
        emb_t = emb[int(self.batch_size/2):, :]
        return emb_s, emb_t

    @staticmethod
    def net_pro_loss(emb, a):
        r = torch.sum(emb*emb, 1)
        r = torch.reshape(r, (-1, 1))
        dis = r-2*torch.matmul(emb, emb.T)+r.T
        return torch.mean(torch.sum(torch.FloatTensor(a).__mul__(dis), 1))


class NodeClassifier(nn.Module):
    def __init__(self, n_emb, num_class, clf_type):
        super(NodeClassifier, self).__init__()
        self.clf_type = clf_type
        self.layer = nn.Linear(n_emb, num_class)
        std = 1/(n_emb/2)**0.5
        nn.init.trunc_normal_(self.layer.weight, std=std, a=-2*std, b=2*std)
        # nn.init.xavier_normal_(self.layer.weight, 1.414)
        nn.init.constant_(self.layer.bias, 0.1)

    def forward(self, emb):
        pred_logit = self.layer(emb)
        return pred_logit


class DomainDiscriminator(nn.Module):
    def __init__(self, n_emb):
        super(DomainDiscriminator, self).__init__()
        self.h_dann_1 = nn.Linear(n_emb, 128)
        self.h_dann_2 = nn.Linear(128, 128)
        self.output_layer = nn.Linear(128, 2)
        std = 1/(n_emb/2)**0.5
        nn.init.trunc_normal_(self.h_dann_1.weight, std=std, a=-2*std, b=2*std)
        # nn.init.xavier_normal_(self.h_dann_1.weight, 1.414)
        nn.init.constant_(self.h_dann_1.bias, 0.1)
        nn.init.trunc_normal_(self.h_dann_2.weight, std=0.125, a=-0.25, b=0.25)
        # nn.init.xavier_normal_(self.h_dann_2.weight, 1.414)
        nn.init.constant_(self.h_dann_2.bias, 0.1)
        nn.init.trunc_normal_(self.output_layer.weight, std=0.125, a=-0.25, b=0.25)
        # nn.init.xavier_normal_(self.output_layer.weight, 1.414)
        nn.init.constant_(self.output_layer.bias, 0.1)

    def forward(self, h_grl):
        h_grl = f.relu(self.h_dann_1(h_grl))
        h_grl = f.relu(self.h_dann_2(h_grl))
        d_logit = self.output_layer(h_grl)
        return d_logit


class ACDNE(nn.Module):
    def __init__(self, n_input, n_hidden, n_emb, num_class, clf_type, batch_size, drop):
        super(ACDNE, self).__init__()
        self.network_embedding = NetworkEmbedding(n_input, n_hidden, n_emb, drop, batch_size)
        self.node_classifier = NodeClassifier(n_emb, num_class, clf_type)
        self.domain_discriminator = DomainDiscriminator(n_emb)
        self.grl = GRL()

    def forward(self, x, x_nei):
        # Network_Embedding
        emb = self.network_embedding(x, x_nei)
        # Node_Classifier
        pred_logit = self.node_classifier(emb)
        # Domain_Discriminator
        h_grl = self.grl(emb)
        d_logit = self.domain_discriminator(h_grl)
        return emb, pred_logit, d_logit
