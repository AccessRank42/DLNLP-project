# model.py

import io

from nltk.translate import bleu_score
import numpy as np


import chainer as ch
import chainer.functions as F
import chainer.links as L


"""
# 1. tokenize DS/input via stanford toeknizer

# 2. token2vec
for token in input:
    if token in glove_vectors:
        vectorized_input.append(appropriate_glove_vector)
    else:
        vectorized_input.append(zero_vector)
"""
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
        # self.n_layers = n_layers

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

        # packed_context_output, (_) = self.encoder(packed_context)
        # D, (_) = pad_packed_sequence(packed_context_output)
        # D = D.contiguous()
        # packed_question_output, (_) = self.encoder(packed_question)
        # Q_intermediate, (_) = pad_packed_sequence(packed_question_output)
        # Q_intermediate = Q_intermediate.contiguous()
        # # Non-linear projection on question encoding space
        # Q = F.tanh(self.ques_projection(Q_intermediate))
        # # Append the sentinel vector, shape = B x 1 x l
        # sentinel_c = self.sentinel_c.unsqueeze(0).expand(
        #     batch_size, config.hidden_dim).unsqueeze(1).contiguous()
        # sentinel_q = self.sentinel_q.unsqueeze(0).expand(
        #     batch_size, config.hidden_dim).unsqueeze(1)
        # # shape changes to B x m+1 x l
        # D = torch.cat((D, sentinel_c), 1)
        # # shape changes to B x n+1 x l
        # Q = torch.cat((Q, sentinel_q), 1)
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



class Seq2seq(ch.Chain):

    def __init__(self, n_layers, n_source_vocab, n_target_vocab, n_units):
        super(Seq2seq, self).__init__()
        with self.init_scope():
            self.embed_x = L.EmbedID(n_source_vocab, n_units)
            self.embed_y = L.EmbedID(n_target_vocab, n_units)
            self.encoder = L.NStepLSTM(n_layers, n_units, n_units, 0.1)
            self.decoder = L.NStepLSTM(n_layers, n_units, n_units, 0.1)
            self.W = L.Linear(n_units, n_target_vocab)

        self.n_layers = n_layers
        self.n_units = n_units

    def forward(self, xs, ys):
        xs = [x[::-1] for x in xs]

        eos = self.xp.array([EOS], np.int32)
        ys_in = [F.concat([eos, y], axis=0) for y in ys]
        ys_out = [F.concat([y, eos], axis=0) for y in ys]

        # Both xs and ys_in are lists of arrays.
        exs = sequence_embed(self.embed_x, xs)
        eys = sequence_embed(self.embed_y, ys_in)

        batch = len(xs)
        # None represents a zero vector in an encoder.
        hx, cx, _ = self.encoder(None, None, exs)
        _, _, os = self.decoder(hx, cx, eys)

        # It is faster to concatenate data before calculating loss
        # because only one matrix multiplication is called.
        concat_os = F.concat(os, axis=0)
        concat_ys_out = F.concat(ys_out, axis=0)
        loss = F.sum(F.softmax_cross_entropy(
            self.W(concat_os), concat_ys_out, reduce='no')) / batch

        ch.report({'loss': loss}, self)
        n_words = concat_ys_out.shape[0]
        perp = self.xp.exp(loss.array * batch / n_words)
        ch.report({'perp': perp}, self)
        return loss

    def translate(self, xs, max_length=100):
        batch = len(xs)
        with ch.no_backprop_mode(), ch.using_config('train', False):
            xs = [x[::-1] for x in xs]
            exs = sequence_embed(self.embed_x, xs)
            h, c, _ = self.encoder(None, None, exs)
            ys = self.xp.full(batch, EOS, np.int32)
            result = []
            for i in range(max_length):
                eys = self.embed_y(ys)
                eys = F.split_axis(eys, batch, 0)
                h, c, ys = self.decoder(h, c, eys)
                cys = F.concat(ys, axis=0)
                wy = self.W(cys)
                ys = self.xp.argmax(wy.array, axis=1).astype(np.int32)
                result.append(ys)

        # Using `xp.concatenate(...)` instead of `xp.stack(result)` here to
        # support NumPy 1.9.
        result = ch.get_device('@numpy').send(
            self.xp.concatenate([x[None, :] for x in result]).T)

        # Remove EOS taggs
        outs = []
        for y in result:
            inds = np.argwhere(y == EOS)
            if len(inds) > 0:
                y = y[:inds[0, 0]]
            outs.append(y)

# """
# Minimal model and tokenizer utilities used by agent.py / the notebook.

# Students must implement the neural architecture so that checkpoints
# trained in the notebook load and decode with the same code here.

# Keep the public API stable:

# - SPECIAL_TOKENS : Dict[str, str]
# - simple_tokenize(s: str) -> List[str]
# - encode(tokens: List[str], stoi: Dict[str, int], add_sos_eos: bool=False) -> List[int]
# - class Encoder(nn.Module): forward(src, src_lens)
# - class Decoder(nn.Module): forward(tgt_in, hidden)
# - class Seq2Seq(nn.Module):
#     - forward(src, src_lens, tgt_in) -> logits [B,T,V]
#     - greedy_decode(src, src_lens, max_len, sos_id, eos_id) -> LongTensor[B, max_len]
# """

# from __future__ import annotations

# from typing import List, Dict, Tuple, Optional
# import torch
# import torch.nn as nn


# # -------------------------
# # Tokenization utilities
# # -------------------------

# SPECIAL_TOKENS = {
#     "pad": "<pad>",
#     "sos": "<sos>",
#     "eos": "<eos>",
#     "unk": "<unk>",
# }


# def simple_tokenize(s: str) -> List[str]:
#     """Lowercase whitespace tokenizer used by both training and inference."""
#     return s.strip().lower().split()


# def encode(tokens: List[str], stoi: Dict[str, int], add_sos_eos: bool = False) -> List[int]:
#     """Map tokens to ids using `stoi`. Optionally wrap with <sos>/<eos>."""
#     ids = [stoi.get(t, stoi[SPECIAL_TOKENS["unk"]]) for t in tokens]
#     if add_sos_eos:
#         ids = [stoi[SPECIAL_TOKENS["sos"]]] + ids + [stoi[SPECIAL_TOKENS["eos"]]]
#     return ids


# # -------------------------
# # Model scaffolding
# # -------------------------

# class Encoder(nn.Module):
#     """
#     Student-implemented encoder.
#     Expected behavior:
#       forward(src: LongTensor[B, S], src_lens: LongTensor[B]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
#     Returns:
#       - outputs: Tensor[B, S, H] (padded time-major outputs)
#       - hidden:  RNN-style tuple (h, c) or similar state your decoder expects
#     """
#     def __init__(self, vocab_size: int, emb_dim: int, hid_dim: int,
#                  num_layers: int = 1, dropout: float = 0.1, nw_type='att_gru'):
#         super().__init__()
#         self.nw_type = nw_type
#         # # TODO: define embeddings and encoder layers to match your notebook training
        
    
#         if nw_type == 'lstm': 
#             # #initial notebook
#             self.emb=nn.Embedding(vocab_size, emb_dim, padding_idx=0)
#             self.rnn=nn.LSTM(emb_dim, hid_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers>1 else 0.0)
#             return

#         if nw_type in ['gru', 'att_gru']:
#             # tutorial
#             self.hidden_size = hid_dim
#             self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
#             self.gru = nn.GRU(emb_dim, hid_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers>1 else 0.0)
#             self.proj=nn.Linear(hid_dim, emb_dim)
#             return
        
#         # if nw_type == 'simple_transf':
#         #     self.emb=nn.Embedding(vocab_size, emb_dim, padding_idx=0)
#         #     transf_layer = nn.TransformerEncoderLayer(vocab_size, 8, dim_feedforward=hid_dim, batch_first=True, dropout=dropout if num_layers>1 else 0.0)
#         #     self.transf=nn.TransformerEncoder(transf_layer, num_layers)
#         #     return

#         raise NotImplementedError("Implement Encoder __init__. for " + nw_type)

#     def forward(self, src: torch.Tensor, src_lens: torch.Tensor):
#         # # TODO: implement packed sequence handling and return (outputs, hidden)
        
#         if self.nw_type == 'lstm':
#             #initial notebook
#             emb=self.emb(src)
#             packed=nn.utils.rnn.pack_padded_sequence(emb, src_lens.cpu(), batch_first=True, enforce_sorted=False)
#             out,(h,c)=self.rnn(packed)
#             out,_=nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
#             return out,(h,c)
#         ###

#         if self.nw_type in ['gru', 'att_gru']:
#             #initial tutorial
#             embedded = self.embedding(src)#
#             packed=nn.utils.rnn.pack_padded_sequence(embedded, src_lens.cpu(), batch_first=True, enforce_sorted=False)
#             output, hidden = self.gru(packed)#
#             output,_=nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
            
#             if self.nw_type == 'att_gru':
#                 hidden = self.proj(hidden)
#                 output = self.proj(output)
#             return output, hidden
        
#         # if self.nw_type == 'simple_transf':
#         #     #transformer attempt
#         #     embedded = self.embedding(src)#
#         #     packed=nn.utils.rnn.pack_padded_sequence(embedded, src_lens.cpu(), batch_first=True, enforce_sorted=False)
#         #     output, hidden = self.transf(packed)#
#         #     output,_=nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        
#         raise NotImplementedError("Implement Encoder.forward.")

# class BahdanauAttention(nn.Module):
#     def __init__(self, hidden_size):
#         super().__init__()
#         self.Wa = nn.Linear(hidden_size, hidden_size)
#         self.Ua = nn.Linear(hidden_size, hidden_size)
#         self.Va = nn.Linear(hidden_size, 1)

#     def forward(self, query, keys):
#         scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
#         scores = scores.squeeze(2).unsqueeze(1)

#         weights = torch.nn.functional.softmax(scores, dim=-1)
#         context = torch.bmm(weights, keys)

#         return context, weights

# class Decoder(nn.Module):
#     """
#     Student-implemented decoder.
#     Expected behavior:
#       forward(tgt_in: LongTensor[B, T], hidden) -> Tuple[Tensor, Any]
#     Returns:
#       - logits: Tensor[B, T, V] (distributions before softmax over target vocab)
#       - hidden: updated recurrent state
#     """
#     def __init__(self, vocab_size: int, emb_dim: int, hid_dim: int,
#                  num_layers: int = 1, dropout: float = 0.1, nw_type='att_gru'):
#         super().__init__()
#         self.nw_type = nw_type
#         # # TODO: define embeddings, recurrent layers, and output projection
        
    
#         if nw_type == 'lstm':
#             #initial notebook
#             self.emb=nn.Embedding(vocab_size, emb_dim, padding_idx=0)
#             self.rnn=nn.LSTM(emb_dim, hid_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers>1 else 0.0)
#             self.proj=nn.Linear(hid_dim, vocab_size)
#             return

#         if nw_type == 'gru':
#             #initial tutorial
#             self.embedding = nn.Embedding(vocab_size, emb_dim)
#             self.gru = nn.GRU(emb_dim, hid_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers>1 else 0.0)
#             self.out = nn.Linear(hid_dim, vocab_size)
#             return

#         if nw_type == 'att_gru':
#             self.embedding = nn.Embedding(vocab_size, emb_dim)
#             self.attention = BahdanauAttention(emb_dim)
#             self.gru = nn.GRU(2 * emb_dim, hid_dim, batch_first=True, dropout=dropout if num_layers>1 else 0.0)
#             self.out = nn.Linear(hid_dim, vocab_size)
#             return

#         raise NotImplementedError("Implement Decoder __init__.")

#     def forward(self, tgt_in: torch.Tensor, hidden, encoder_outputs=None):
#         # # TODO: return (logits, hidden)
        
    
#         if self.nw_type == 'lstm':
#             #initial notebook
#             emb=self.emb(tgt_in)
#             out,hidden=self.rnn(emb, hidden)
#             return self.proj(out), hidden

#         if self.nw_type == 'gru':
#             #initial tutorial
#             output = self.embedding(tgt_in)
#             output = torch.nn.functional.relu(output)
#             output, hidden = self.gru(output, hidden)
#             output = self.out(output)
#             return output, hidden

#         if self.nw_type == 'att_gru':
#             return self.forward_step_att(tgt_in, hidden, encoder_outputs)
    
#         raise NotImplementedError("Implement Decoder.forward.")

    
    
#     def forward_step_att(self, input, hidden, encoder_outputs):
#         embedded =  self.embedding(input)

#         query = hidden.permute(1, 0, 2)
#         context, attn_weights = self.attention(query, encoder_outputs)

#         #shape missmatch here
#         # print('embd, contxt')
#         # print(embedded.size())
#         # print(context.size())
#         input_gru = torch.cat((embedded, context), dim=2)

#         output, hidden = self.gru(input_gru, hidden)
#         output = self.out(output)

#         return output, (hidden, attn_weights)

        


# class Seq2Seq(nn.Module):
#     """
#     Student-implemented Seq2Seq wrapper that ties Encoder and Decoder.

#     Required methods:
#       - forward(src, src_lens, tgt_in) -> logits [B, T, V]
#       - greedy_decode(src, src_lens, max_len, sos_id, eos_id) -> LongTensor[B, max_len]
#         Greedy decoding should stop at <eos> per sequence and pad remainder with <eos>.
#     """
#     def __init__(self, encoder: Encoder, decoder: Decoder, nw_type='gru'):
#         super().__init__()
#         self.encoder = encoder
#         self.decoder = decoder
#         self.nw_type = nw_type

#     def forward(self, src: torch.Tensor, src_lens: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
#         # # TODO: encode, then decode with teacher forcing; return logits [B,T,V]
    
#         if self.nw_type in ['lstm', 'gru']:
#             #initial notebook
#             _,h=self.encoder(src, src_lens)
#             logits,_=self.decoder(tgt_in, h)
#             return logits
#             ###

#         if self.nw_type == 'att_gru':
#             #attention
#             encoder_output,h=self.encoder(src, src_lens)
#             print(h.size())
#             print(encoder_output.size())
#             print(tgt_in.size())
#             logits,_=self.decoder(tgt_in, h, encoder_outputs=encoder_output)
#             return logits
    
#         raise NotImplementedError("Implement Seq2Seq.forward for "+self.nw_type)

#     @torch.no_grad()
#     def greedy_decode(
#         self,
#         src: torch.Tensor,
#         src_lens: torch.Tensor,
#         max_len: int,
#         sos_id: int,
#         eos_id: int,
#     ) -> torch.Tensor:
#         # """
#         # TODO: implement token-by-token greedy decoding.
#         # Must return LongTensor[B, max_len]. If <eos> is emitted at step t,
#         # set positions > t to <eos> for that sequence.
#         # """
        
#         if self.nw_type in ['lstm', 'gru']:
#             # initial notebook
#             B=src.size(0)
#             _,h=self.encoder(src, src_lens)
#             inputs=torch.full((B,1), sos_id, dtype=torch.long, device=src.device)
#             outs=[]
#             for _ in range(max_len):
#                 logits,h=self.decoder(inputs[:,-1:].contiguous(), h)
#                 nxt=logits[:,-1,:].argmax(-1, keepdim=True) #determine next token
#                 outs.append(nxt)
#                 inputs=torch.cat([inputs,nxt], dim=1) #add token to recurrent input
#             seqs=torch.cat(outs, dim=1)
#             for i in range(B):
#                 row=seqs[i]
#                 if (row==eos_id).any(): 
#                     idx=(row==eos_id).nonzero(as_tuple=False)[0].item()
#                     row[idx+1:]=eos_id
#             return seqs
#         ###

#         if self.nw_type == 'att_gru':
#             #attention
#             batch_size = src.size(0)
#             decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=src.device).fill_(sos_id)
#             encoder_outputs,hidden=self.encoder(src, src_lens)
#             decoder_outputs = []
#             attentions = []

#             for i in range(max_len):
#                 decoder_output, (hidden, attn_weights) = self.decoder(decoder_input[:,-1:].contiguous(), hidden, encoder_outputs=encoder_outputs)
#                 nxt=decoder_output[:,-1,:].argmax(-1, keepdim=True)
#                 decoder_outputs.append(nxt)
#                 decoder_input=torch.cat([decoder_input,nxt], dim=1)
#                 attentions.append(attn_weights)
                
        
#             seqs=torch.cat(decoder_outputs, dim=1)
#             for i in range(batch_size):
#                 row=seqs[i]
#                 if (row==eos_id).any(): 
#                     idx=(row==eos_id).nonzero(as_tuple=False)[0].item()
#                     row[idx+1:]=eos_id
#             return seqs

#         raise NotImplementedError("Implement Seq2Seq.greedy_decode for"+self.nw_type)
