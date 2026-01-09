# model.py

import io

from nltk.translate import bleu_score
import numpy as np


import chainer as ch
import chainer.functions as F
import chainer.links as L



class DocAndQuesEncoder(ch.Chain):
    def __init__(self, emb_mat, dropout, hid_size):
        super(DocAndQuesEncoder, self).__init__()
        with self.init_scope():
            self.emb = L.EmbedID(emb_mat.shape[0], emb_mat.shape[1], initialW=emb_mat)
            self.emb.disable_update()
            self.encLSTM = L.NStepLSTM(1, emb_mat.shape[1], hid_size, dropout)
            # self.projectionLayer = ch.Sequential(L.Linear(hid_size, hid_size), F.tanh)
            self.projectionLayer = L.Linear(hid_size, hid_size)

            # Sentinel vectors for D, Q
            self.sentielD = ch.Parameter(np.random.randn(emb_mat.shape[1], 1)) #TODO: shape?
            self.sentielQ = ch.Parameter(np.random.randn(emb_mat.shape[1], 1)) #TODO: shape?

        self.dropout = dropout

    def forward(self, x_D, x_Q, hx_D=None, cx_D=None, hx_Q=None, cx_Q=None):

        x_D_emb = F.dropout(self.emb(x_D), self.dropout)

        # nstep LSTM needs it split into a list
        x_D_emb_ = [x_D_emb[i] for i in range(x_D_emb.shape[0])]
        _, _, D = self.encLSTM(hx_D, cx_D, x_D_emb_)
        D = F.stack(D, axis=0)
        # print(D.shape)

        #TODO: append sentinel D

        x_Q_emb = F.dropout(self.emb(x_Q), self.dropout)

        x_Q_emb_ = [x_Q_emb[i] for i in range(x_Q_emb.shape[0])]
        _, _, Q_ = self.encLSTM(hx_Q, cx_Q, x_Q_emb_)
        Q_ = F.stack(Q_, axis=0)

        Q = F.tanh(self.projectionLayer(Q_, n_batch_axes=2))

        #TODO: append sentinel Q

        return (D, Q)

class CoattentionEncoder(ch.Chain):
    def __init__(self, dropout, hid_size):
        super(CoattentionEncoder, self).__init__()
        with self.init_scope():
            # self.biLSTM = L.NStepBiLSTM(1, 3*hid_size, 2*hid_size, dropout)
            self.biLSTM = L.NStepBiLSTM(1, 3*hid_size, hid_size, dropout)

        self.dropout = dropout

    def forward(self, D, Q, hx=None, cx=None):
        # TODO Note: in our case m = n = max_seq_length due to padding 
        # l is our hid_size
        # also, matmuls & dims are reversed

        # L = D^TQ \in \R^{(m + 1) x (n + 1)}
        # L = F.matmul(F.transpose(D, axes=(0,2,1)), Q)
        L = F.matmul(Q, F.transpose(D, axes=(0,2,1)))

        # A^Q = softmax(L) \in \R^{(m + 1) x (n + 1)}
        Aq = F.softmax(L, axis=2)
        # print(Aq.shape)
        # A^Q = softmax(L^T) \in \R^{(n + 1) x (m + 1)}
        Ad = F.softmax(F.transpose(L, axes=(0,2,1)))

        # C^Q = DA^Q \in \R^{l x (n + 1)}
        # Cq = F.matmul(D, Aq) 
        Cq = F.matmul(Aq, D) 
        # C^D = [Q; C^Q]A^D \in \R^{2l x (m + 1)}, that is the simultaneous multiplication of QA^D and C^QA^D
        QCq = F.concat([Q, Cq], axis=2)
        # Cd = F.matmul(QCq, Ad)
        Cd = F.matmul(Ad, QCq)

        # U = [u_1, ..., u_m] for u_t = bi-directional LSTM(u_{t-1}, u_{t+1}, [d_t, c^D_t]) \in \R^{2l} 
        DCd = F.concat([D, Cd], axis=2)
        # print(DCd.shape)
        DCd = [DCd[i] for i in range(DCd.shape[0])]
        _, _, U = self.biLSTM(hx, cx, DCd)
        U = F.stack(U, axis=0)
        # print(U.shape)


        #TODO: remove sentinel vect - before or after calc U?
        return U

class HighwayMaxout(ch.Chain):
    def __init__(self, dropout, hid_size, maxout_pool_size):
        super(HighwayMaxout, self).__init__()
        with self.init_scope():
            # self.projectionLayer = ch.Sequential(L.Linear(5*hid_size, hid_size, nobias=True), F.tanh)
            self.projectionLayer = L.Linear(hid_size)
            # self.max1 = L.Maxout((3*hid_size)**2, hid_size, maxout_pool_size)
            self.max1 = L.Maxout((3*hid_size)**2, hid_size, maxout_pool_size)
            self.max2 = L.Maxout(hid_size, hid_size, maxout_pool_size)
            self.max3 = L.Maxout(2*hid_size, 1, maxout_pool_size)

        self.dropout = dropout

    def forward(self, U, h_i, u_s_i, u_e_i):
        # print(h_i.shape)
        # print(u_s_i.shape)
        # print(u_e_i.shape)
        # r_in = F.concat([h_i, u_s_i, u_e_i], axis=2)
        r_in = F.concat([h_i, u_s_i, u_e_i], axis=1)
        print(r_in.shape)
        r = F.tanh(self.projectionLayer(r_in))
        r = F.swapaxes(F.tile(r, (U.shape[1],1,1)), 0, 1)
        print(r.shape)
        print(U.shape)
        Ur = F.concat([U, r], axis=2)
        m_t_1 = self.max1(Ur)
        m_t_2 = self.max2(m_t_1)
        hmn = self.max3(F.concat([m_t_1, m_t_2], axis=1))
        return hmn


class DynamicPointingDecoder(ch.Chain):
    def __init__(self, dropout, hid_size, maxout_pool_size, dyn_dec_max_it):
        super(DynamicPointingDecoder, self).__init__()
        with self.init_scope():
            # self.decLSTM = L.NStepLSTM(1, hid_size, hid_size, dropout)
            # self.decLSTM = L.LSTM(hid_size, hid_size)
            self.decLSTM = L.LSTM(None, hid_size)

            self.hmn_start = HighwayMaxout(dropout, hid_size, maxout_pool_size)
            self.hmn_end = HighwayMaxout(dropout, hid_size, maxout_pool_size)

        self.dropout = dropout
        self.dyn_dec_max_it = dyn_dec_max_it

    def forward(self, U, hx=None, cx=None):
        # h_i = U
        s_i = 0
        e_i = U.shape[1]-1

        # u_s_i = U[:,s_i:s_i+1,:]
        u_s_i = U[:,s_i,:]
        # u_e_i = U[:,e_i:e_i+1,:]
        u_e_i = U[:,e_i,:]
        
        #TODO
        for i in range(self.dyn_dec_max_it):
            
            # h_i = [U[i] for i in range(h_i.shape[0])]
            
            # hx, cx, h_i = self.decLSTM(hx, cx, h_i) #TODO: reuse hx, cx?
            # h_i = F.stack(h_i, axis=0)
            # u_se_i = F.concat([u_s_i, u_e_i], axis=2) #TODO: should axis=1 instead?
            u_se_i = F.concat([u_s_i, u_e_i], axis=1) #TODO: should axis=1 instead?
            h_i = self.decLSTM(u_se_i) #TODO: unsure if and how h_i should also be fed in

            alpha = self.hmn_start(U, h_i, u_s_i, u_e_i)
            # s = F.max(alpha)
            beta = self.hmn_end(U, h_i, u_s_i, u_e_i)
            # e = F.max(beta)

            u_s_i = U[:,s_i:s_i+1,:]
            u_e_i = U[:,e_i:e_i+1,:]
        

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



# class Seq2seq(ch.Chain):

#     def __init__(self, n_layers, n_source_vocab, n_target_vocab, n_units):
#         super(Seq2seq, self).__init__()
#         with self.init_scope():
#             self.embed_x = L.EmbedID(n_source_vocab, n_units)
#             self.embed_y = L.EmbedID(n_target_vocab, n_units)
#             self.encoder = L.NStepLSTM(n_layers, n_units, n_units, 0.1)
#             self.decoder = L.NStepLSTM(n_layers, n_units, n_units, 0.1)
#             self.W = L.Linear(n_units, n_target_vocab)

#         self.n_layers = n_layers
#         self.n_units = n_units

#     def forward(self, xs, ys):
#         xs = [x[::-1] for x in xs]

#         eos = self.xp.array([EOS], np.int32)
#         ys_in = [F.concat([eos, y], axis=0) for y in ys]
#         ys_out = [F.concat([y, eos], axis=0) for y in ys]

#         # Both xs and ys_in are lists of arrays.
#         exs = sequence_embed(self.embed_x, xs)
#         eys = sequence_embed(self.embed_y, ys_in)

#         batch = len(xs)
#         # None represents a zero vector in an encoder.
#         hx, cx, _ = self.encoder(None, None, exs)
#         _, _, os = self.decoder(hx, cx, eys)

#         # It is faster to concatenate data before calculating loss
#         # because only one matrix multiplication is called.
#         concat_os = F.concat(os, axis=0)
#         concat_ys_out = F.concat(ys_out, axis=0)
#         loss = F.sum(F.softmax_cross_entropy(
#             self.W(concat_os), concat_ys_out, reduce='no')) / batch

#         ch.report({'loss': loss}, self)
#         n_words = concat_ys_out.shape[0]
#         perp = self.xp.exp(loss.array * batch / n_words)
#         ch.report({'perp': perp}, self)
#         return loss

#     def translate(self, xs, max_length=100):
#         batch = len(xs)
#         with ch.no_backprop_mode(), ch.using_config('train', False):
#             xs = [x[::-1] for x in xs]
#             exs = sequence_embed(self.embed_x, xs)
#             h, c, _ = self.encoder(None, None, exs)
#             ys = self.xp.full(batch, EOS, np.int32)
#             result = []
#             for i in range(max_length):
#                 eys = self.embed_y(ys)
#                 eys = F.split_axis(eys, batch, 0)
#                 h, c, ys = self.decoder(h, c, eys)
#                 cys = F.concat(ys, axis=0)
#                 wy = self.W(cys)
#                 ys = self.xp.argmax(wy.array, axis=1).astype(np.int32)
#                 result.append(ys)

#         # Using `xp.concatenate(...)` instead of `xp.stack(result)` here to
#         # support NumPy 1.9.
#         result = ch.get_device('@numpy').send(
#             self.xp.concatenate([x[None, :] for x in result]).T)

#         # Remove EOS taggs
#         outs = []
#         for y in result:
#             inds = np.argwhere(y == EOS)
#             if len(inds) > 0:
#                 y = y[:inds[0, 0]]
#             outs.append(y)
