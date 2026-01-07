# model.py

import io

from nltk.translate import bleu_score
import numpy as np


import chainer as ch
import chainer.functions as F
import chainer.links as L


class DocAndQuesEncoder(ch.Chain):
    # def __init__(self, n_layers, n_units, n_target_vocab, emb_mat, dropout, hid_size):
    def __init__(self, emb_mat, dropout, hid_size):
        super(DynamicCoattentionNW, self).__init__()
        with self.init_scope():
            self.emb = L.EmbedID(emb_mat.shape[0], emb_mat.shape[1], initialW=emb_mat)
            self.emb.disable_update()
            # self.encLSTM = L.LSTM(emb_mat.shape[1], hid_size)
            self.encLSTM = L.NStepLSTM(1, emb_mat.shape[1], hid_size, dropout)
            # self.projectionLayer = ch.Sequential(L.Linear(n_units, n_target_vocab), F.tanh)
            self.projectionLayer = ch.Sequential(L.Linear(hid_size, hid_size), F.tanh)
            self.sentielD = ch.Parameter(np.random.randn(emb_mat.shape[1], 1)) #TODO: shape?
            self.sentielQ = ch.Parameter(np.random.randn(emb_mat.shape[1], 1)) #TODO: shape?
            # self.sentiel = L.Parameter(np.random.randn(1, emb_mat.shape[1]))

        self.dropout = dropout

    def forward(self, x_D, x_Q, hx_D=None, cx_D=None, hx_Q=None, cx_Q=None):

        x_D_emb = F.dropout(self.emb(x_D), self.dropout)

        _, _, D = self.encLSTM(hx_D, cx_D, x_D_emb)
        # D = F.dropout(self.encLSTM(x_D_emb), self.dropout)
        #TODO: append sentinel Q

        x_Q_emb = F.dropout(self.emb(x_D), self.dropout)

        # Q_ = F.dropout(self.encLSTM(x_D_emb), self.dropout)
        _, _, Q_ = self.encLSTM(hx_Q, cx_Q, x_Q_emb)
        Q = self.projectionLayer(Q_)

        #TODO: append sentinel Q


        return (D, Q)

class CoattentionEncoder(ch.Chain):
    def __init__(self, dropout, hid_size):
        super(CoattentionEncoder, self).__init__()
        with self.init_scope():
            self.biLSTM = L.NStepBiLSTM(1, 3*hid_size, 2*hid_size, dropout)

        self.dropout = dropout

    def forward(self, D, Q, hx=None, cx=None):
        L = F.matmul(F.transpose(D), Q)

        Aq = F.softmax(L)
        Ad = F.softmax(F.transpose(L))

        Cq = F.matmul(D, Aq)
        Cd = F.matmul(F.concat([Q, Cq], axis=0), Ad)

        # U = bi-directional LSTM
        _, _, U = self.biLSTM(hx, cx, F.concat([D, Cd], axis=0))

        #TODO: remove sentinel vect - before or after U?
        return U

class HighwayMaxout(ch.Chain):
    def __init__(self, dropout, hid_size, maxout_pool_size):
        super(DynamicPointingDecoder, self).__init__()
        with self.init_scope():
            self.projectionLayer_alpha = ch.Sequential(L.Linear(5*hid_size, hid_size, nobias=True), F.tanh)
            self.max1 = L.Maxout(3*hid_size, hid_size, maxout_pool_size)
            self.max2 = L.Maxout(hid_size, hid_size, maxout_pool_size)
            self.max3 = L.Maxout(2*hid_size, 1, maxout_pool_size)

        self.dropout = dropout

    def forward(self, U):
        r = None #TODO
        m_t_1 = self.max1(F.concat([U, r], axis=0))
        m_t_2 = self.max2(m_t_1)
        hmn = self.max3(F.concat([m_t_1, m_t_2], axis=0))
        return hmn


class DynamicPointingDecoder(ch.Chain):
    def __init__(self, dropout, hid_size, maxout_pool_size, dyn_dec_max_it):
        super(DynamicPointingDecoder, self).__init__()
        with self.init_scope():
            self.encLSTM = L.NStepLSTM(1, hid_size, hid_size, dropout)

            self.hmn_start = HighwayMaxout(dropout, hid_size, maxout_pool_size)
            self.hmn_end = HighwayMaxout(dropout, hid_size, maxout_pool_size)

        self.dropout = dropout
        self.dyn_dec_max_it = dyn_dec_max_it

    def forward(self, U):
        #TODO
        for i in range(self.dyn_dec_max_it):
            alpha = self.hmn_start(U)
            # s = F.max(alpha)
            beta = self.hmn_end(U)
            # e = F.max(beta)
        

        # return s[-1], e[-1]
        


class DynamicCoattentionNW(ch.Chain):
    # def __init__(self, n_layers, n_source_vocab, n_target_vocab, n_units):
    def __init__(self, max_seq_length, hid_state_size, dyn_dec_max_it, maxout_pool_size, dropout, emb_mat):
        super(DynamicCoattentionNW, self).__init__()
        with self.init_scope():
            # self.embed_D = None
            self.docQuesEncoder = DocAndQuesEncoder(emb_mat, dropout, hid_state_size)
            self.coAttEncoder = CoattentionEncoder(dropout, hid_state_size)
            self.decoder = DynamicPointingDecoder(dropout, hid_state_size, maxout_pool_size)

        self.max_seq_length = max_seq_length
        self.hid_state_size = hid_state_size
        self.dyn_dec_max_it = dyn_dec_max_it
        self.maxout_pool_size = maxout_pool_size
        self.dropout = dropout

    def forward(self, d, q):
        # TODO: pad d, q to max seq?

        D, Q = self.docQuesEncoder(d, q)

        U = self.coAttEncoder(D, Q)

        s, e = self.decoder(U)
