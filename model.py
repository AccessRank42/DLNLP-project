# model.py
"""
Minimal model and tokenizer utilities used by agent.py / the notebook.

Students must implement the neural architecture so that checkpoints
trained in the notebook load and decode with the same code here.

Keep the public API stable:

- SPECIAL_TOKENS : Dict[str, str]
- simple_tokenize(s: str) -> List[str]
- encode(tokens: List[str], stoi: Dict[str, int], add_sos_eos: bool=False) -> List[int]
- class Encoder(nn.Module): forward(src, src_lens)
- class Decoder(nn.Module): forward(tgt_in, hidden)
- class Seq2Seq(nn.Module):
    - forward(src, src_lens, tgt_in) -> logits [B,T,V]
    - greedy_decode(src, src_lens, max_len, sos_id, eos_id) -> LongTensor[B, max_len]
"""

from __future__ import annotations

from typing import List, Dict, Tuple, Optional
import torch
import torch.nn as nn


# -------------------------
# Tokenization utilities
# -------------------------

SPECIAL_TOKENS = {
    "pad": "<pad>",
    "sos": "<sos>",
    "eos": "<eos>",
    "unk": "<unk>",
}


def simple_tokenize(s: str) -> List[str]:
    """Lowercase whitespace tokenizer used by both training and inference."""
    return s.strip().lower().split()


def encode(tokens: List[str], stoi: Dict[str, int], add_sos_eos: bool = False) -> List[int]:
    """Map tokens to ids using `stoi`. Optionally wrap with <sos>/<eos>."""
    ids = [stoi.get(t, stoi[SPECIAL_TOKENS["unk"]]) for t in tokens]
    if add_sos_eos:
        ids = [stoi[SPECIAL_TOKENS["sos"]]] + ids + [stoi[SPECIAL_TOKENS["eos"]]]
    return ids


# -------------------------
# Model scaffolding
# -------------------------

class Encoder(nn.Module):
    """
    Student-implemented encoder.
    Expected behavior:
      forward(src: LongTensor[B, S], src_lens: LongTensor[B]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
    Returns:
      - outputs: Tensor[B, S, H] (padded time-major outputs)
      - hidden:  RNN-style tuple (h, c) or similar state your decoder expects
    """
    def __init__(self, vocab_size: int, emb_dim: int, hid_dim: int,
                 num_layers: int = 1, dropout: float = 0.1, nw_type='att_gru'):
        super().__init__()
        self.nw_type = nw_type
        # # TODO: define embeddings and encoder layers to match your notebook training
        
    
        if nw_type == 'lstm': 
            # #initial notebook
            self.emb=nn.Embedding(vocab_size, emb_dim, padding_idx=0)
            self.rnn=nn.LSTM(emb_dim, hid_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers>1 else 0.0)
            return

        if nw_type in ['gru', 'att_gru']:
            # tutorial
            self.hidden_size = hid_dim
            self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
            self.gru = nn.GRU(emb_dim, hid_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers>1 else 0.0)
            self.proj=nn.Linear(hid_dim, emb_dim)
            return
        
        # if nw_type == 'simple_transf':
        #     self.emb=nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        #     transf_layer = nn.TransformerEncoderLayer(vocab_size, 8, dim_feedforward=hid_dim, batch_first=True, dropout=dropout if num_layers>1 else 0.0)
        #     self.transf=nn.TransformerEncoder(transf_layer, num_layers)
        #     return

        raise NotImplementedError("Implement Encoder __init__. for " + nw_type)

    def forward(self, src: torch.Tensor, src_lens: torch.Tensor):
        # # TODO: implement packed sequence handling and return (outputs, hidden)
        
        if self.nw_type == 'lstm':
            #initial notebook
            emb=self.emb(src)
            packed=nn.utils.rnn.pack_padded_sequence(emb, src_lens.cpu(), batch_first=True, enforce_sorted=False)
            out,(h,c)=self.rnn(packed)
            out,_=nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
            return out,(h,c)
        ###

        if self.nw_type in ['gru', 'att_gru']:
            #initial tutorial
            embedded = self.embedding(src)#
            packed=nn.utils.rnn.pack_padded_sequence(embedded, src_lens.cpu(), batch_first=True, enforce_sorted=False)
            output, hidden = self.gru(packed)#
            output,_=nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
            
            if self.nw_type == 'att_gru':
                hidden = self.proj(hidden)
                output = self.proj(output)
            return output, hidden
        
        # if self.nw_type == 'simple_transf':
        #     #transformer attempt
        #     embedded = self.embedding(src)#
        #     packed=nn.utils.rnn.pack_padded_sequence(embedded, src_lens.cpu(), batch_first=True, enforce_sorted=False)
        #     output, hidden = self.transf(packed)#
        #     output,_=nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        
        raise NotImplementedError("Implement Encoder.forward.")

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = torch.nn.functional.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights

class Decoder(nn.Module):
    """
    Student-implemented decoder.
    Expected behavior:
      forward(tgt_in: LongTensor[B, T], hidden) -> Tuple[Tensor, Any]
    Returns:
      - logits: Tensor[B, T, V] (distributions before softmax over target vocab)
      - hidden: updated recurrent state
    """
    def __init__(self, vocab_size: int, emb_dim: int, hid_dim: int,
                 num_layers: int = 1, dropout: float = 0.1, nw_type='att_gru'):
        super().__init__()
        self.nw_type = nw_type
        # # TODO: define embeddings, recurrent layers, and output projection
        
    
        if nw_type == 'lstm':
            #initial notebook
            self.emb=nn.Embedding(vocab_size, emb_dim, padding_idx=0)
            self.rnn=nn.LSTM(emb_dim, hid_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers>1 else 0.0)
            self.proj=nn.Linear(hid_dim, vocab_size)
            return

        if nw_type == 'gru':
            #initial tutorial
            self.embedding = nn.Embedding(vocab_size, emb_dim)
            self.gru = nn.GRU(emb_dim, hid_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers>1 else 0.0)
            self.out = nn.Linear(hid_dim, vocab_size)
            return

        if nw_type == 'att_gru':
            self.embedding = nn.Embedding(vocab_size, emb_dim)
            self.attention = BahdanauAttention(emb_dim)
            self.gru = nn.GRU(2 * emb_dim, hid_dim, batch_first=True, dropout=dropout if num_layers>1 else 0.0)
            self.out = nn.Linear(hid_dim, vocab_size)
            return

        raise NotImplementedError("Implement Decoder __init__.")

    def forward(self, tgt_in: torch.Tensor, hidden, encoder_outputs=None):
        # # TODO: return (logits, hidden)
        
    
        if self.nw_type == 'lstm':
            #initial notebook
            emb=self.emb(tgt_in)
            out,hidden=self.rnn(emb, hidden)
            return self.proj(out), hidden

        if self.nw_type == 'gru':
            #initial tutorial
            output = self.embedding(tgt_in)
            output = torch.nn.functional.relu(output)
            output, hidden = self.gru(output, hidden)
            output = self.out(output)
            return output, hidden

        if self.nw_type == 'att_gru':
            return self.forward_step_att(tgt_in, hidden, encoder_outputs)
    
        raise NotImplementedError("Implement Decoder.forward.")

    
    
    def forward_step_att(self, input, hidden, encoder_outputs):
        embedded =  self.embedding(input)

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)

        #shape missmatch here
        # print('embd, contxt')
        # print(embedded.size())
        # print(context.size())
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, (hidden, attn_weights)

        


class Seq2Seq(nn.Module):
    """
    Student-implemented Seq2Seq wrapper that ties Encoder and Decoder.

    Required methods:
      - forward(src, src_lens, tgt_in) -> logits [B, T, V]
      - greedy_decode(src, src_lens, max_len, sos_id, eos_id) -> LongTensor[B, max_len]
        Greedy decoding should stop at <eos> per sequence and pad remainder with <eos>.
    """
    def __init__(self, encoder: Encoder, decoder: Decoder, nw_type='gru'):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.nw_type = nw_type

    def forward(self, src: torch.Tensor, src_lens: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        # # TODO: encode, then decode with teacher forcing; return logits [B,T,V]
    
        if self.nw_type in ['lstm', 'gru']:
            #initial notebook
            _,h=self.encoder(src, src_lens)
            logits,_=self.decoder(tgt_in, h)
            return logits
            ###

        if self.nw_type == 'att_gru':
            #attention
            encoder_output,h=self.encoder(src, src_lens)
            print(h.size())
            print(encoder_output.size())
            print(tgt_in.size())
            logits,_=self.decoder(tgt_in, h, encoder_outputs=encoder_output)
            return logits
    
        raise NotImplementedError("Implement Seq2Seq.forward for "+self.nw_type)

    @torch.no_grad()
    def greedy_decode(
        self,
        src: torch.Tensor,
        src_lens: torch.Tensor,
        max_len: int,
        sos_id: int,
        eos_id: int,
    ) -> torch.Tensor:
        # """
        # TODO: implement token-by-token greedy decoding.
        # Must return LongTensor[B, max_len]. If <eos> is emitted at step t,
        # set positions > t to <eos> for that sequence.
        # """
        
        if self.nw_type in ['lstm', 'gru']:
            # initial notebook
            B=src.size(0)
            _,h=self.encoder(src, src_lens)
            inputs=torch.full((B,1), sos_id, dtype=torch.long, device=src.device)
            outs=[]
            for _ in range(max_len):
                logits,h=self.decoder(inputs[:,-1:].contiguous(), h)
                nxt=logits[:,-1,:].argmax(-1, keepdim=True) #determine next token
                outs.append(nxt)
                inputs=torch.cat([inputs,nxt], dim=1) #add token to recurrent input
            seqs=torch.cat(outs, dim=1)
            for i in range(B):
                row=seqs[i]
                if (row==eos_id).any(): 
                    idx=(row==eos_id).nonzero(as_tuple=False)[0].item()
                    row[idx+1:]=eos_id
            return seqs
        ###

        if self.nw_type == 'att_gru':
            #attention
            batch_size = src.size(0)
            decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=src.device).fill_(sos_id)
            encoder_outputs,hidden=self.encoder(src, src_lens)
            decoder_outputs = []
            attentions = []

            for i in range(max_len):
                decoder_output, (hidden, attn_weights) = self.decoder(decoder_input[:,-1:].contiguous(), hidden, encoder_outputs=encoder_outputs)
                nxt=decoder_output[:,-1,:].argmax(-1, keepdim=True)
                decoder_outputs.append(nxt)
                decoder_input=torch.cat([decoder_input,nxt], dim=1)
                attentions.append(attn_weights)
                
        
            seqs=torch.cat(decoder_outputs, dim=1)
            for i in range(batch_size):
                row=seqs[i]
                if (row==eos_id).any(): 
                    idx=(row==eos_id).nonzero(as_tuple=False)[0].item()
                    row[idx+1:]=eos_id
            return seqs

        raise NotImplementedError("Implement Seq2Seq.greedy_decode for"+self.nw_type)
