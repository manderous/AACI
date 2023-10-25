import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from model.tree import Tree, head_to_tree, tree_to_adj
from utils import constant, torch_utils
from model.counterfactual import get_shortest_path_tmp, get_neighbor
from model.CausalNormClassifier_la import Causal_Norm_Classifier


class GCNClassifier(nn.Module):
    """ A wrapper classifier for GCNRelationModel. """
    def __init__(self, opt, emb_matrix=None):
        super().__init__()
        self.gcn_model = GCNRelationModel(opt, emb_matrix=emb_matrix)
        self.opt = opt
        self.classifiers = self.gcn_model.classifiers

    def weighted_loss(self, l1, l2, l3, l4, loss_weight=0.9):
        return l1 + torch.sigmoid(self.loss_weight1)*l2 + torch.sigmoid(self.loss_weight2)*l3 + l4

    def conv_l2(self):
        return self.gcn_model.gcn.conv_l2()

    def forward(self, inputs, eval):
        main_logits, logits_dict, original_logits, inventions_logits, pooling_output = self.gcn_model(inputs, eval)
        return main_logits, logits_dict, original_logits, inventions_logits, pooling_output


class GCNRelationModel(nn.Module):
    def __init__(self, opt, emb_matrix=None):
        super().__init__()
        self.opt = opt
        self.emb_matrix = emb_matrix

        # init moving average
        self.embed_mean = torch.zeros(int(opt['feature_dim'])).numpy()
        self.mu = 0.9
        self.Causal_Norm_Classifier = Causal_Norm_Classifier(num_classes=opt['num_class'],
                                                             feat_dim=opt['feature_dim'],
                                                             use_effect=True,
                                                             num_head=5, tau=16.0, alpha=0.85, gamma=0.125)

        # create embedding layers
        self.emb = nn.Embedding(opt['vocab_size'], opt['emb_dim'], padding_idx=constant.PAD_ID)
        self.pos_emb = nn.Embedding(len(constant.POS_TO_ID), opt['pos_dim']) if opt['pos_dim'] > 0 else None
        self.ner_emb = nn.Embedding(len(constant.NER_TO_ID), opt['ner_dim']) if opt['ner_dim'] > 0 else None
        embeddings = (self.emb, self.pos_emb, self.ner_emb)
        self.init_embeddings()

        # gcn layer
        self.gcn = GCN(opt, embeddings, opt['hidden_dim'], opt['num_layers'])
        self.in_drop = nn.Dropout(opt['input_dropout'])

        # output mlp layers
        in_dim = opt['hidden_dim']*3
        if self.opt.get('no_gcn', False):
            layers = [nn.Linear(in_dim*2, opt['hidden_dim']), nn.ReLU()]
        else:
            layers = [nn.Linear(in_dim, opt['hidden_dim']), nn.PReLU()]
        for _ in range(self.opt['mlp_layers']-1):
            layers += [nn.Linear(opt['hidden_dim'], opt['hidden_dim']), nn.PReLU()]

        self.out_mlp = nn.Sequential(*layers)
        self.effect_type = self.opt['effect_type']
        print("Casual Effect is:{}".format(self.effect_type))
        self.criterion = nn.CrossEntropyLoss()
        self.ner_classifier = nn.Linear(opt['hidden_dim'], len(constant.NER_TO_ID))
        self.rnn = nn.LSTM(opt['emb_dim'] + opt['ner_dim'] + opt['pos_dim'], opt['rnn_hidden'], opt['rnn_layers'],
                           batch_first=True,dropout=opt['rnn_dropout'], bidirectional=True)
        self.in_dim = opt['rnn_hidden'] * 2
        self.rnn_drop = nn.Dropout(opt['rnn_dropout'])
        self.ctx_I = opt['ctx_I']
        self.opt['context'] = self.opt.get('context', False)
        self.main_classifier = nn.Linear(opt['hidden_dim'], opt['num_class'])

        ### baseline
        self.classifiers = [self.main_classifier, self.ner_classifier]
        if self.effect_type != "None":
            self.classifiers += [self.tags2rel, self.ents2rel, self.sent2rel, 
                            self.gate2rel]
        if self.opt.get('lws', False):
            self.tau = nn.Parameter(torch.tensor(self.opt['tau'], dtype=float))
        else:
            self.tau = self.opt.get('tau', 0)

    def moving_average(self, holder, input):
            assert len(input.shape) == 2
            with torch.no_grad():
                holder = holder * (1 - self.average_ratio) + self.average_ratio * input.mean(0).view(-1)
            return holder

    ### baseline
    def adaptive_classify(self, classifier, x):
        from utils.bert_utils import tau_classify
        if not self.opt.get('use_tau', False):
            return classifier(x)
        return tau_classify(x, classifier, self.tau)

    def init_embeddings(self):
        if self.emb_matrix is None:
            self.emb.weight.data[1:,:].uniform_(-1.0, 1.0)
        else:
            self.emb_matrix = torch.from_numpy(self.emb_matrix)
            self.emb.weight.data.copy_(self.emb_matrix)
        # decide finetuning
        if self.opt['topn'] <= 0:
            print("Do not finetune word embedding layer.")
            self.emb.weight.requires_grad = False
        elif self.opt['topn'] < self.opt['vocab_size']:
            print("Finetune top {} word embeddings.".format(self.opt['topn']))
            self.emb.weight.register_hook(lambda x: \
                    torch_utils.keep_partial_grad(x, self.opt['topn']))
        else:
            print("Finetune all embeddings.")

    def encode_with_rnn(self, rnn_inputs, masks, batch_size):
        new_mask =  masks.data.eq(constant.PAD_ID).long().sum(1).squeeze()
        if len(new_mask.shape) == 0:
            seq_lens = list(masks.data.eq(constant.PAD_ID).long().sum(1))
        else:
            seq_lens = list(masks.data.eq(constant.PAD_ID).long().sum(1).squeeze())
        h0, c0 = rnn_zero_state(batch_size, self.opt['rnn_hidden'], self.opt['rnn_layers'])
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens, batch_first=True)
        rnn_outputs, (ht, ct) = self.rnn(rnn_inputs, (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        return rnn_outputs

    def forward(self, inputs, eval):
        words, masks, pos, ner, deprel, head, subj_pos, obj_pos, subj_type, obj_type = inputs # unpack
        l = (masks.data.cpu().numpy() == 0).astype(np.int64).sum(1)
        pool_type = self.opt['pooling']
        maxlen = max(l)
        batch_size = words.size(0)

        def inputs_to_tree_reps(head, words, l, prune, subj_pos, obj_pos):
            head, words, subj_pos, obj_pos = head.cpu().numpy(), words.cpu().numpy(), subj_pos.cpu().numpy(), obj_pos.cpu().numpy()
            trees = [head_to_tree(head[i], words[i], l[i], prune, subj_pos[i], obj_pos[i]) for i in range(len(l))]
            adj = [tree_to_adj(maxlen, tree, directed=False, self_loop=False).reshape(1, maxlen, maxlen) for tree in trees]
            cf_paths = None
            if self.opt['context']:
                cf_paths = [get_shortest_path_tmp(adj[i].reshape(maxlen, maxlen), subj_pos[i], obj_pos[i], l[i]) for i in range(len(l))]
                cf_paths = [torch.LongTensor(i) for i in cf_paths]
            adj = np.concatenate(adj, axis=0)
            adj = torch.from_numpy(adj)
            adj = Variable(adj.cuda()) if self.opt['cuda'] else Variable(adj)
            return adj, cf_paths

        # deprecated mr

        adj, cf_paths = inputs_to_tree_reps(head.data, words.data, l, self.opt['prune_k'], subj_pos.data, obj_pos.data)
        word_embs = self.emb(words)

        if self.effect_type == 'None':
            if self.opt['ner_dim'] > 0:
                word_embs = torch.cat([word_embs,self.ner_emb(ner), self.pos_emb(pos)], dim=-1)
            word_embs = self.in_drop(word_embs)
        
        ctx_inputs = self.rnn_drop(self.encode_with_rnn(word_embs, masks, words.size()[0]))
        h, pool_mask = self.gcn(adj, ctx_inputs)
        h_out = pool(h, pool_mask, type=pool_type)
        subj_mask, obj_mask = subj_pos.eq(0).eq(0).unsqueeze(2), obj_pos.eq(0).eq(0).unsqueeze(2)  # invert mask
        subj_out = pool(h, subj_mask, type=pool_type)
        obj_out = pool(h, obj_mask, type=pool_type)
        logitsdict = None
        original_logits  = intervention_logits = None
        outputs = torch.cat([h_out, subj_out, obj_out], dim=-1)
        outputs = self.out_mlp(outputs)
        # update moving average
        if not eval:
            self.embed_mean = self.mu * self.embed_mean + outputs.detach().mean(0).view(-1).cpu().numpy()
        main_logits, _ = self.Causal_Norm_Classifier(outputs, self.embed_mean, eval)
        return main_logits, logitsdict, original_logits, intervention_logits, h_out


class GCN(nn.Module):
    """ A GCN/Contextualized GCN module operated on dependency graphs. """
    def __init__(self, opt, embeddings, mem_dim, num_layers):
        super(GCN, self).__init__()

        self.opt = opt
        self.effect_type = self.opt['effect_type']
        self.layers = num_layers
        self.use_cuda = opt['cuda']
        self.mem_dim = mem_dim
        self.opt['use_bert'] = self.opt.get('use_bert', False)
        if self.opt['use_bert']:
            self.in_dim = opt['bert_dim'] + opt['pos_dim'] + opt['ner_dim']
        else:
            self.in_dim = opt['emb_dim'] + opt['pos_dim'] + opt['ner_dim']

        self.emb, self.pos_emb, self.ner_emb = embeddings

        # rnn layer
        if self.opt.get('rnn', False):
            input_size = self.in_dim
            self.rnn = nn.LSTM(input_size, opt['rnn_hidden'], opt['rnn_layers'], batch_first=True, \
                    dropout=opt['rnn_dropout'], bidirectional=True)
            self.in_dim = opt['rnn_hidden'] * 2
            self.rnn_drop = nn.Dropout(opt['rnn_dropout']) # use on last layer output

        self.in_drop = nn.Dropout(opt['input_dropout'])
        self.gcn_drop = nn.Dropout(opt['gcn_dropout'])

        # gcn layer
        self.W = nn.ModuleList()
        for layer in range(self.layers):
            input_dim = self.in_dim if layer == 0 else self.mem_dim
            self.W.append(nn.Linear(input_dim, self.mem_dim))

    def conv_l2(self):
        conv_weights = []
        for w in self.W:
            conv_weights += [w.weight, w.bias]
        return sum([x.pow(2).sum() for x in conv_weights])

    def encode_with_rnn(self, rnn_inputs, masks, batch_size):
        new_mask =  masks.data.eq(constant.PAD_ID).long().sum(1).squeeze()
        if len(new_mask.shape) == 0:
            seq_lens = list(masks.data.eq(constant.PAD_ID).long().sum(1))
        else:
            seq_lens = list(masks.data.eq(constant.PAD_ID).long().sum(1).squeeze())
        h0, c0 = rnn_zero_state(batch_size, self.opt['rnn_hidden'], self.opt['rnn_layers'])
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens, batch_first=True)
        rnn_outputs, (ht, ct) = self.rnn(rnn_inputs, (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        return rnn_outputs

    def forward(self, adj, inputs):
        gcn_inputs = inputs
        denom = adj.sum(2).unsqueeze(2) + 1
        mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2)
        # zero out adj for ablation
        if self.opt.get('no_adj', False):
            adj = torch.zeros_like(adj)

        for l in range(self.layers):
            Ax = adj.bmm(gcn_inputs)
            AxW = self.W[l](Ax)
            AxW = AxW + self.W[l](gcn_inputs) # self loop
            AxW = AxW / denom
            gAxW = F.relu(AxW)
            gcn_inputs = self.gcn_drop(gAxW) if l < self.layers - 1 else gAxW
        return gcn_inputs, mask


def pool(h, mask, type='max'):
    if type == 'max':
        h = h.masked_fill(mask, -constant.INFINITY_NUMBER)
        return torch.max(h, 1)[0]
    elif type == 'avg':
        h = h.masked_fill(mask, 0)
        return h.sum(1) / (mask.size(1) - mask.float().sum(1))
    else:
        h = h.masked_fill(mask, 0)
        return h.sum(1)


def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True, use_cuda=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    if use_cuda:
        return h0.cuda(), c0.cuda()
    else:
        return h0, c0

