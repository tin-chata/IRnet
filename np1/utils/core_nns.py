"""
Created on 2018-11-27
@author: duytinvo
"""
import math
import torch
import numpy as np
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Embs(nn.Module):
    """
    This module take (characters or words) indices as inputs and outputs (characters or words) embedding
    """

    def __init__(self, HPs):
        super(Embs, self).__init__()
        [size, dim, pre_embs, drop_rate, zero_padding] = HPs
        self.zero_padding = zero_padding
        self.embeddings = nn.Embedding(size, dim, padding_idx=0)
        if pre_embs is not None:
            self.embeddings.weight.data.copy_(torch.from_numpy(pre_embs))
        else:
            self.embeddings.weight.data.copy_(torch.from_numpy(self.random_embedding(size, dim)))
        self.drop = nn.Dropout(drop_rate)

    def forward(self, inputs, auxiliary_embs=None):
        return self.get_embs(inputs, auxiliary_embs)

    def get_embs(self, inputs, auxiliary_embs=None):
        """
        embs.shape([0, 1]) == auxiliary_embs.shape([0, 1])
        """
        if self.zero_padding:
            # set zero vector for padding, unk, eot, sot
            self.set_zeros([0, 1, 2, 3])
        # embs = tensor(batch_size, seq_length,input_dim)
        embs = self.embeddings(inputs)
        embs_drop = self.drop(embs)
        if auxiliary_embs is not None:
            assert embs_drop.shape[:-1] == auxiliary_embs.shape[:-1]
            embs_drop = torch.cat([embs_drop, auxiliary_embs], -1)
        return embs_drop

    def random_embedding(self, size, dim):
        pre_embs = np.empty([size, dim])
        scale = np.sqrt(3.0 / dim)
        for index in range(size):
            pre_embs[index, :] = np.random.uniform(-scale, scale, [1, dim])
        return pre_embs

    def set_zeros(self, idx):
        for i in idx:
            self.embeddings.weight.data[i].fill_(0)


class NN_Embs(nn.Module):
    """
    This module take embedding inputs (characters or words) feeding to an RNN layer to extract:
        - all hidden features
        - last hidden features
        - all attentional hidden features
        - last attentional hidden features
    """

    def __init__(self, HPs, dropout=0.2):
        super(NN_Embs, self).__init__()
        [emb_size, emb_dim, pre_embs, emb_drop_rate, emb_zero_padding,
         nn_mode, nn_inp_dim, nn_out_dim, nn_layers, nn_bidirect, nn_attention] = HPs

        emb_HPs = [emb_size, emb_dim, pre_embs, emb_drop_rate, emb_zero_padding]
        self.emb_layer = Embs(emb_HPs)

        nn_rnn_dim = nn_out_dim // 2 if nn_bidirect else nn_out_dim
        if nn_mode == "rnn":
            self.hidden_layer = nn.RNN(nn_inp_dim, nn_rnn_dim, num_layers=nn_layers, dropout=dropout,
                                       batch_first=True, bidirectional=nn_bidirect)
        elif nn_mode == "gru":
            self.hidden_layer = nn.GRU(nn_inp_dim, nn_rnn_dim, num_layers=nn_layers, dropout=dropout,
                                       batch_first=True, bidirectional=nn_bidirect)
        else:
            self.hidden_layer = nn.LSTM(nn_inp_dim, nn_rnn_dim, num_layers=nn_layers, dropout=dropout,
                                        batch_first=True, bidirectional=nn_bidirect)
            # Set the bias of forget gate to 1.0
            for names in self.hidden_layer._all_weights:
                for name in filter(lambda n: "bias" in n, names):
                    bias = getattr(self.hidden_layer, name)
                    n = bias.size(0)
                    start, end = n // 4, n // 2
                    bias.data[start:end].fill_(1.)

        self.nn_attention = nn_attention
        if nn_attention:
            self.transform_layer = nn.Linear(nn_out_dim, nn_out_dim)
            self.attention_layer = nn.Linear(nn_out_dim, 1, bias=False)
            self.norm_layer = nn.Softmax(-1)

    def forward(self, inputs, input_lengths, auxiliary_embs=None, init_hidden=None):
        return self.get_all_hiddens(inputs, input_lengths, auxiliary_embs, init_hidden)

    def get_last_hiddens(self, inputs, input_lengths, auxiliary_embs=None, init_hidden=None):
        """
            input:
                inputs: tensor(batch_size, seq_length)
                input_lengths: tensor(batch_size,  1)
            output:
                tensor(batch_size, hidden_dim)
        """
        batch_size = inputs.size(0)
        seq_length = inputs.size(1)
        embs_drop = self.emb_layer(inputs, auxiliary_embs)
        pack_input = pack_padded_sequence(embs_drop, input_lengths.data.cpu().numpy(), True)
        # rnn_out = tensor(batch_size, seq_length, rnn_dim * 2)
        # hc_n = (h_n,c_n); h_n = tensor(2, batch_size, rnn_dim)
        rnn_out, hc_n = self.hidden_layer(pack_input, init_hidden)
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)
        if self.nn_attention:
            # h_transform = tensor(batch_size, seq_length, rnn_dim * 2)
            h_transform = torch.relu(self.transform_layer(rnn_out))
            # a_raw = tensor(batch_size, seq_length, 1)
            a_raw = torch.relu(self.attention_layer(h_transform))
            # a_alpha = tensor(batch_size, seq_length)
            a_raw.squeeze_()
            # a_norm = tensor(batch_size, seq_length)
            a_norm = self.norm_layer(a_raw)
            # h_attention = tensor(batch_size, seq_length, rnn_dim * 2)
            h_attention = rnn_out * a_norm.view(batch_size, seq_length, 1)
            # hn_attention = tensor(batch_size, rnn_dim * 2)
            hn_attention = h_attention.sum(1)
            return hn_attention
        else:
            # concatenate forward and backward h_n; h_n = tensor(batch_size, rnn_dim*2)
            if type(hc_n) == tuple:
                h_n = torch.cat([hc_n[0][0, :, :], hc_n[0][1, :, :]], -1)
            else:
                h_n = torch.cat([hc_n[0, :, :], hc_n[1, :, :]], -1)
            return h_n

    def get_all_hiddens(self, inputs, input_lengths=None, auxiliary_embs=None, init_hidden=None):
        """
            input:
                inputs: tensor(batch_size, seq_length)
                input_lengths: tensor(batch_size,  1)
            output:
                tensor(batch_size, seq_length, hidden_dim)
        """
        batch_size = inputs.size(0)
        seq_length = inputs.size(1)
        embs_drop = self.emb_layer(inputs, auxiliary_embs)
        # rnn_out = tensor(batch_size, seq_length, rnn_dim * 2)
        # hc_n = (h_n,c_n); h_n = tensor(2, batch_size, rnn_dim)
        pack_input = pack_padded_sequence(embs_drop, input_lengths.data.cpu().numpy(), True)
        rnn_out, hc_n = self.hidden_layer(pack_input, init_hidden)
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)
        if self.nn_attention:
            # h_transform = tensor(batch_size, seq_length, rnn_dim * 2)
            h_transform = torch.relu(self.transform_layer(rnn_out))
            # a_raw = tensor(batch_size, seq_length, 1)
            a_raw = torch.relu(self.attention_layer(h_transform))
            # a_alpha = tensor(batch_size, seq_length)
            a_raw.squeeze_()
            # a_norm = tensor(batch_size, seq_length)
            a_norm = self.norm_layer(a_raw)
            # h_attention = tensor(batch_size, seq_length, rnn_dim * 2)
            h_attention = rnn_out * a_norm.view(batch_size, seq_length, 1)
            # # concatenate forward and backward h_n; h_n = tensor(batch_size, rnn_dim*2)
            # if type(hc_n) == tuple:
            #     h_n = torch.cat([hc_n[0][0, :, :], hc_n[0][1, :, :]], -1)
            # else:
            #     h_n = torch.cat([hc_n[0, :, :], hc_n[1, :, :]], -1)
            # att_out = h_n.view(batch_size, 1, -1) * a_norm.view(batch_size, seq_length, 1)
            return h_attention, hc_n
        else:
            return rnn_out, hc_n


class Char_Word_Encoder(nn.Module):
    """
    The model builds character biLSTM, concatenated by word embeddings with attentional mechanism
    to pass through another biLSTM for extracting final features for affine layers
    """

    def __init__(self, word_HPs, char_HPs=None):
        super(Char_Word_Encoder, self).__init__()
        self.use_char = True if char_HPs is not None else False
        if self.use_char:
            self.char_nn_embs = NN_Embs(char_HPs)
            word_HPs[6] += char_HPs[7]

        self.word_nn_embs = NN_Embs(word_HPs)

    def forward(self, word_inputs, word_lengths,
                char_inputs=None, char_lengths=None, char_seq_recover=None,
                init_hidden=None):
        return self.get_all_hiddens(word_inputs, word_lengths,
                                    char_inputs, char_lengths, char_seq_recover,
                                    init_hidden)

    def get_all_hiddens(self, word_inputs, word_lengths,
                        char_inputs=None, char_lengths=None, char_seq_recover=None,
                        init_hidden=None):
        word_batch = word_inputs.size(0)
        seq_length = word_inputs.size(1)
        char_embs = None
        if self.use_char:
            char_embs = self.char_nn_embs.get_last_hiddens(char_inputs, char_lengths)
            char_embs = char_embs[char_seq_recover]
            char_embs = char_embs.view(word_batch, seq_length, -1)
        rnn_out, hidden_out = self.word_nn_embs.get_all_hiddens(word_inputs, word_lengths, char_embs, init_hidden)
        return rnn_out, hidden_out

    def get_last_hiddens(self, word_inputs, word_lengths,
                         char_inputs=None, char_lengths=None, char_seq_recover=None,
                         init_hidden=None):
        word_batch = word_inputs.size(0)
        seq_length = word_inputs.size(1)
        char_embs = None
        if self.use_char:
            char_embs = self.char_nn_embs.get_last_hiddens(char_inputs, char_lengths)
            char_embs = char_embs[char_seq_recover]
            char_embs = char_embs.view(word_batch, seq_length, -1)
        h_n = self.word_nn_embs.get_last_hiddens(word_inputs, word_lengths, char_embs, init_hidden)
        return h_n


class Word_alignment(nn.Module):
    def __init__(self, in_features, out_features, hidden_dim):
        super(Word_alignment, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()
        self.hidden_layer = nn.Linear(out_features + in_features, hidden_dim)

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, input1, input2, input_mask=None):
        """

        :param input1: [batch, seq_length1, in_features]
        :param input2: [batch, seq_length2, out_features]
        :param input_mask: mask of input1
        :return:
        """
        out1 = F.linear(input1, self.weight)
        # out1: [batch, seq_length1, out_features]
        # input2: [batch, seq_length2, out_features]
        out2 = torch.bmm(out1, input2.transpose(1, -1))
        # TODO: use mask tensor to filter out padding in out2[:,seq_length1,:]
        if input_mask:
            out2[~input_mask] = -100
        # out2: [batch, seq_length1, seq_length2]
        # input1: [batch, seq_length1, in_features]
        satt = torch.bmm(F.softmax(out2, dim=1).transpose(1, -1), input1)
        # satt: [batch, seq_length2, in_features]
        con = torch.cat((input2, satt), dim=-1)
        # con: [batch, seq_length2, in_features + out_features]
        hidden_features = F.relu(self.hidden_layer(con))
        return hidden_features


class Char_Word_Decoder(nn.Module):
    """
    The model builds character biLSTM, concatenated by word embeddings with attentional mechanism
    to pass through another biLSTM for extracting final features for affine layers
    """

    def __init__(self, word_HPs, char_HPs=None, drop_rate=0.5, num_labels=None, enc_att=False):
        super(Char_Word_Decoder, self).__init__()
        self.decoder = Char_Word_Encoder(word_HPs, char_HPs)
        self.num_labels = num_labels
        self.enc_att = enc_att
        hidden_dim = word_HPs[7]
        att_dim = 2*hidden_dim
        self.finaldrop_layer = nn.Dropout(drop_rate)
        if num_labels > 2:
            self.lossF = nn.CrossEntropyLoss()
            if enc_att:
                self.attention = Word_alignment(hidden_dim, hidden_dim, att_dim)
                self.hidden2tag_layer = nn.Linear(att_dim, num_labels)
            else:
                self.hidden2tag_layer = nn.Linear(hidden_dim, num_labels)
        else:
            self.lossF = nn.BCEWithLogitsLoss()
            if enc_att:
                self.attention = Word_alignment(hidden_dim, hidden_dim, att_dim)
                self.hidden2tag_layer = nn.Linear(att_dim, 1)
            else:
                self.hidden2tag_layer = nn.Linear(hidden_dim, 1)

    def forward(self, word_inputs, word_lengths,
                char_inputs=None, char_lengths=None, char_seq_recover=None,
                init_hidden=None, enc_out=None, enc_mask=None):
        rnn_out, _ = self.get_all_hiddens(word_inputs, word_lengths,
                                          char_inputs, char_lengths, char_seq_recover,
                                          init_hidden, enc_out, enc_mask)
        label_score = self.scoring(rnn_out)
        return label_score

    def get_all_hiddens(self, word_inputs, word_lengths,
                        char_inputs, char_lengths, char_seq_recover,
                        init_hidden, enc_out, enc_mask):
        rnn_out, hidden_out = self.decoder.get_all_hiddens(word_inputs, word_lengths,
                                                           char_inputs, char_lengths, char_seq_recover, init_hidden)
        if self.enc_att:
            rnn_out = self.attention(enc_out, rnn_out, enc_mask)
        return rnn_out, hidden_out

    def scoring(self, rnn_out):
        label_score = self.hidden2tag_layer(rnn_out)
        label_score = self.finaldrop_layer(label_score)
        return label_score

    def NLL_loss(self, label_score, label_tensor):
        if self.num_labels > 2:
            # label_score = [B, C]; label_tensor = [B, ]
            batch_loss = self.lossF(label_score.view(-1, self.num_labels), label_tensor.view(-1,))
        else:
            # label_score = [B, *]; label_tensor = [B, *]
            batch_loss = self.lossF(label_score, label_tensor.float().view(-1, 1))
        return batch_loss

    def inference(self, label_score, k=1):
        if self.num_labels > 2:
            label_prob = torch.softmax(label_score, dim=-1)
            label_prob, label_pred = label_prob.data.topk(k)
        else:
            label_prob = torch.sigmoid(label_score.squeeze())
            label_pred = (label_prob >= 0.5).data.long()
        return label_prob, label_pred

    def logsm_inference(self, label_score, k=1):
        if self.num_labels > 2:
            label_prob = torch.nn.functional.log_softmax(label_score, dim=-1)
            label_prob, label_pred = label_prob.data.topk(k)
        else:
            label_prob = torch.sigmoid(label_score.squeeze())
            label_pred = (label_prob >= 0.5).data.long()
        return label_prob, label_pred


if __name__ == '__main__':
    import random
    from data_utils import *

    device = torch.device("cpu")
    filename = "../../data/nl2fr/train_set.csv"
    # train_data = Csvfile(filename, firstline=True)
    # source_data, target_data = [], []
    # for nl, fr in train_data:
    #     source_data.append(nl)
    #     target_data.append(fr)

    s_paras = [None, 1, None, 1]
    t_paras = [None, 1, None, 1]
    vocab = Vocab(s_paras, t_paras)
    vocab.build([filename], firstline=False)
    source2idx = vocab.wd2idx(vocab_words=vocab.sw2i, unk_words=True, sos=False, eos=True,
                              vocab_chars=vocab.sc2i, unk_chars=True, sow=False, eow=True)
    target2idx = vocab.wd2idx(vocab_words=vocab.tw2i, unk_words=True, sos=False, eos=True,
                              vocab_chars=vocab.tc2i, unk_chars=True, sow=False, eow=True,
                              reverse=True)

    train_data = Csvfile(filename, firstline=True, source2idx=source2idx, target2idx=target2idx)

    train_iters = Vocab.minibatches(train_data, batch_size=5)
    source_data, target_data = [], []
    for source, target in train_iters:
        schar_ids, sword_ids = list(zip(*source))
        sword_pad_ids, sseq_lens = seqPAD.pad_sequences(sword_ids, pad_tok=vocab.sw2i[PAD])
        schar_pad_ids, swd_lens = seqPAD.pad_sequences(schar_ids, pad_tok=vocab.sc2i[PAD], nlevels=2)

        tchar_ids, tword_ids = list(zip(*target))
        tword_pad_ids, tseq_lens = seqPAD.pad_sequences(tword_ids, pad_tok=vocab.tw2i[PAD])
        tchar_pad_ids, twd_lens = seqPAD.pad_sequences(tchar_ids, pad_tok=vocab.tc2i[PAD], nlevels=2)

    source_tensors = Data2tensor.sorted_tensors(sword_pad_ids, sseq_lens, schar_pad_ids, swd_lens, False)
    sseq_tensor, sseq_len_tensor, sseqord_tensor, sseqord_recover_tensor, \
    swd_tensor, swd_len_tensor, swdord_recover_tensor = source_tensors

    target_tensors = Data2tensor.sorted_tensors(tword_pad_ids, tseq_lens, tchar_pad_ids, twd_lens, True)
    ttag_tensor, tseq_tensor, tseq_len_tensor, tseqord_tensor, tseqord_recover_tensor, \
    twd_tensor, twd_len_tensor, twdord_recover_tensor = target_tensors

    # a test on decompose a sequence of indices into a sequence of words
    ssents = Vocab.idx2text(sseq_tensor.tolist(), vocab.i2sw, 2)
    # a test on decompose a sequence of indices into a sequence of characters
    swords = Vocab.idx2text(swd_tensor[swdord_recover_tensor].tolist(), vocab.i2sc, 2)

    # a test on decompose `word` to `characters`
    tchar2idx = vocab.wd2idx(vocab_words=None, unk_words=True, sos=False, eos=False,
                             vocab_chars=vocab.tc2i, unk_chars=True, sow=True, eow=True)
    target_char_idx = tchar2idx([vocab.i2tw[60], vocab.i2tw[40]])
    chars = Vocab.idx2text(target_char_idx, vocab.i2tc, 2)

    emb_size = len(vocab.sw2i)
    emb_dim = 50
    emb_pretrained = None
    emb_drop_rate = 0.5
    emb_zero_padding = True
    nn_mode = "lstm"
    nn_inp_dim = 50
    nn_out_dim = 100
    nn_layers = 1
    nn_bidirect = True
    nn_attention = False
    drop_rate = 0.5

    HPs = [emb_size, emb_dim, emb_pretrained, emb_drop_rate, emb_zero_padding,
           nn_mode, nn_inp_dim, nn_out_dim, nn_layers, nn_bidirect, nn_attention]

    encoder = Char_Word_Encoder(HPs)
    en_output, en_hidden = encoder(sseq_tensor, sseq_len_tensor)

    if isinstance(en_hidden, tuple):
        en_hidden_re = tuple(hidden[:, sseqord_recover_tensor, :] for hidden in en_hidden)
        en_hidden_re_sort = tuple(hidden[:, tseqord_tensor, :] for hidden in en_hidden_re)
    else:
        en_hidden_re = en_hidden[:, sseqord_recover_tensor, :]
        en_hidden_re_sort = en_hidden_re[:, tseqord_tensor, :]

    num_labels = len(vocab.tw2i)
    HPs2 = [num_labels, emb_dim, emb_pretrained, emb_drop_rate, emb_zero_padding,
           nn_mode, nn_inp_dim, nn_out_dim, nn_layers, nn_bidirect, nn_attention]

    decoder = Char_Word_Decoder(HPs2, num_labels=num_labels)
    score_tensor = decoder(tseq_tensor, tseq_len_tensor)
    label_mask = ttag_tensor > 0
    batch_loss = decoder.NLL_loss(score_tensor[label_mask], ttag_tensor[label_mask])
