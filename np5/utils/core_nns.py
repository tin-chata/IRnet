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


class Emb_layer(nn.Module):
    """
    This module take (characters or words) indices as inputs and outputs (characters or words) embedding
    """

    def __init__(self, HPs):
        super(Emb_layer, self).__init__()
        [size, dim, pre_embs, drop_rate, zero_padding, requires_grad] = HPs
        self.zero_padding = zero_padding
        self.embeddings = nn.Embedding(size, dim, padding_idx=0)
        if pre_embs is not None:
            self.embeddings.weight.data.copy_(torch.from_numpy(pre_embs))
        else:
            self.embeddings.weight.data.copy_(torch.from_numpy(self.random_embedding(size, dim)))
        if not requires_grad:
            print("Fixed pre-trained embeddings")
            self.embeddings.weight.requires_grad = requires_grad
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
            embs_drop = torch.cat((embs_drop, auxiliary_embs), -1)
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


class RNN_layer(nn.Module):
    """
    This module take embedding inputs (characters or words) feeding to an RNN layer to extract:
        - all hidden features
        - last hidden features
        - all attentional hidden features
        - last attentional hidden features
    """
    def __init__(self, HPs):
        super(RNN_layer, self).__init__()
        [nn_mode, nn_inp_dim, nn_out_dim, nn_layers, nn_bidirect, nn_dropout] = HPs
        nn_rnn_dim = nn_out_dim // 2 if nn_bidirect else nn_out_dim
        if nn_mode == "rnn":
            self.hidden_layer = nn.RNN(nn_inp_dim, nn_rnn_dim, num_layers=nn_layers, dropout=nn_dropout,
                                       batch_first=True, bidirectional=nn_bidirect)
        elif nn_mode == "gru":
            self.hidden_layer = nn.GRU(nn_inp_dim, nn_rnn_dim, num_layers=nn_layers, dropout=nn_dropout,
                                       batch_first=True, bidirectional=nn_bidirect)
        else:
            self.hidden_layer = nn.LSTM(nn_inp_dim, nn_rnn_dim, num_layers=nn_layers, dropout=nn_dropout,
                                        batch_first=True, bidirectional=nn_bidirect)
            # # Set the bias of forget gate to 1.0
            # for names in self.hidden_layer._all_weights:
            #     for name in filter(lambda n: "bias" in n, names):
            #         bias = getattr(self.hidden_layer, name)
            #         n = bias.size(0)
            #         start, end = n // 4, n // 2
            #         bias.data[start:end].fill_(1.)

    def forward(self, emb_inputs, input_lengths, init_hidden=None):
        return self.get_all_hiddens(emb_inputs, input_lengths, init_hidden)

    def get_last_hiddens(self, emb_inputs, input_lengths, init_hidden=None):
        """
            input:
                inputs: tensor(batch_size, seq_length)
                input_lengths: tensor(batch_size,  1)
            output:
                tensor(batch_size, hidden_dim)
        """
        # rnn_out = tensor(batch_size, seq_length, rnn_dim * num_directions)
        # hc_n = (h_n,c_n)
        # h_n = tensor(num_layers *num_directions, batch_size, rnn_dim)
        rnn_out, hc_n = self.get_all_hiddens(emb_inputs, input_lengths, init_hidden=init_hidden)
        # concatenate forward and backward h_n; h_n = tensor(batch_size, rnn_dim*2)
        if type(hc_n) == tuple:
            h_n = torch.cat((hc_n[0][-2, :, :], hc_n[0][-1, :, :]), -1)
        else:
            h_n = torch.cat((hc_n[-2, :, :], hc_n[-1, :, :]), -1)
        return h_n

    def get_all_hiddens(self, emb_inputs, input_lengths=None, init_hidden=None):
        """
            input:
                inputs: tensor(batch_size, seq_length)
                input_lengths: tensor(batch_size,  1)
            output:
                tensor(batch_size, seq_length, hidden_dim)
        """
        # rnn_out = tensor(batch_size, seq_length, rnn_dim * num_directions)
        # hc_n = (h_n,c_n);
        # h_n = tensor(num_layers*num_directions, batch_size, rnn_dim)
        pack_input = pack_padded_sequence(emb_inputs, input_lengths.data.cpu().numpy(), True)
        rnn_out, hc_n = self.hidden_layer(pack_input, init_hidden)
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)
        return rnn_out, hc_n


class ColPredictor(nn.Module):
    """
    This model is used to simultaneously predict the number of elements and a set of these elements.
    It could be used in:
        - col (Column) classifier model
            + #num = 6
                -- {[], [vl_i], [vl_i, vl_j], [vl_i, vl_j, vl_k], [vl_i, vl_jk, vl_l], [vl_i, vl_jkl, vl_m]};
                -- Note: no [] in corpus
            + #labels = 1
                -- directly and dynamically pass #columns (126 --> [0, 125])
    """
    def __init__(self,  HPs, use_hs=True, num_labels=4, labels=1,
                 dtype=torch.long, device=torch.device("cpu")):
        super(ColPredictor, self).__init__()
        self.dtype = dtype
        self.device = device
        self.use_hs = use_hs

        self.q_lstm = RNN_layer(HPs)
        self.hs_lstm = RNN_layer(HPs)
        self.col_lstm = RNN_layer(HPs)

        N_h = HPs[2]
        self.q_num_att = nn.Linear(N_h, N_h)
        self.hs_num_att = nn.Linear(N_h, N_h)
        self.col_num_out_q = nn.Linear(N_h, N_h)
        self.col_num_out_hs = nn.Linear(N_h, N_h)

        self.col_num_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, num_labels))  # num of cols: 1-4

        self.q_att = nn.Linear(N_h, N_h)
        self.hs_att = nn.Linear(N_h, N_h)
        self.col_out_q = nn.Linear(N_h, N_h)
        self.col_out_c = nn.Linear(N_h, N_h)
        self.col_out_hs = nn.Linear(N_h, N_h)

        self.col_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, labels))  # labels = 1

        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        # Multi-label classification
        self.CE = nn.CrossEntropyLoss()
        # Binary classification
        self.BCE = nn.BCEWithLogitsLoss()

    def forward(self, q_emb_var, q_len, q_recover,
                hs_emb_var, hs_len, hs_recover,
                col_emb_var, col_len, col_recover,
                gt_col=None):
        # q_enc: (B, max_q_len, N_h)
        q_enc, _ = self.q_lstm(q_emb_var, q_len)
        q_enc = q_enc[q_recover]
        # hs_enc: (B, max_hs_len, N_h)
        hs_enc, _ = self.hs_lstm(hs_emb_var, hs_len)
        hs_enc = hs_enc[hs_recover]
        # col_enc: (B, max_col_len, N_h)
        col_enc, _ = self.col_lstm(col_emb_var, col_len)
        col_enc = col_enc[col_recover]

        q_mask = torch.arange(q_enc.size(1), dtype=self.dtype, device=self.device)[None, :] < q_len[q_recover][:, None]
        hs_mask = torch.arange(hs_enc.size(1), dtype=self.dtype, device=self.device)[None, :] < hs_len[hs_recover][:, None]
        col_mask = torch.arange(col_enc.size(1), dtype=self.dtype, device=self.device)[None, :] < col_len[col_recover][:, None]
        # ---------------------------------------------------------------------------------
        #                       Predict column number
        # ---------------------------------------------------------------------------------
        # att_val_qc_num: (B, max_col_len, max_q_len)
        # col_enc: (B, max_col_len, N_h); q_enc: (B, max_q_len, N_h) --MLP(T(1,2))--> (B, N_h, max_q_len)
        att_val_qc_num = torch.bmm(col_enc, self.q_num_att(q_enc).transpose(1, 2))
        # _qc_mask: (B, max_col_len, max_q_len)
        _qc_mask = torch.bmm(col_mask.to(dtype=torch.float).unsqueeze(-1), q_mask.to(dtype=torch.float).unsqueeze(1))

        # assign empty slots to -100 --> softmax(-100) = 0
        att_val_qc_num[~_qc_mask.to(dtype=torch.uint8)] = -100
        # att_prob_qc_num: (B, max_col_len, max_q_len)
        att_prob_qc_num = self.softmax(att_val_qc_num)
        # q_weighted_num: (B, N_h)
        # q_enc.unsqueeze(1): (B, 1, max_q_len, N_h); att_prob_qc_num.unsqueeze(3): (B, max_col_len, max_q_len, 1)
        # TODO: instead of using .sum(1) on max_col_len, could use one more attention layer
        q_weighted_num = (q_enc.unsqueeze(1) * att_prob_qc_num.unsqueeze(3)).sum(2).sum(1)

        # Same as the above, compute SQL history embedding weighted by column attentions
        # att_val_hc_num: (B, max_col_len, max_hs_len)
        # col_enc: (B, max_col_len, N_h); hs_enc: (B, max_hs_len, N_h) --MLP(T(1,2))--> (B, N_h, max_hs_len)
        att_val_hc_num = torch.bmm(col_enc, self.hs_num_att(hs_enc).transpose(1, 2))
        # _hc_mask: (B, max_col_len, max_hs_len)
        _hc_mask = torch.bmm(col_mask.to(dtype=torch.float).unsqueeze(-1), hs_mask.to(dtype=torch.float).unsqueeze(1))

        # assign empty slots to -100 --> softmax(-100) = 0
        att_val_hc_num[~_hc_mask.to(dtype=torch.uint8)] = -100
        # att_prob_hc_num: (B, max_col_len, max_hs_len)
        att_prob_hc_num = self.softmax(att_val_hc_num)
        # TODO: instead of using .sum(1) on max_col_len, could use one more attention layer
        # (hs_enc.unsqueeze(1): (B, 1, max_hs_len, N_h)
        # att_prob_hc_num.unsqueeze(3): (B, max_col_len, max_hs_len, 1);
        # hs_weighted_num: (B, N_h)
        hs_weighted_num = (hs_enc.unsqueeze(1) * att_prob_hc_num.unsqueeze(3)).sum(2).sum(1)
        # col_num_score: (B, num_labels)
        col_num_score = self.col_num_out(self.col_num_out_q(q_weighted_num) +
                                         int(self.use_hs) * self.col_num_out_hs(hs_weighted_num))

        # ---------------------------------------------------------------------------------
        #                               Predict columns
        # ---------------------------------------------------------------------------------
        # att_val_qc: (B, max_col_len, max_q_len)
        # col_enc: (B, max_col_len, N_h); q_enc: (B, max_q_len, N_h) --MLP(T(1,2))--> (B, N_h, max_q_len)
        att_val_qc = torch.bmm(col_enc, self.q_att(q_enc).transpose(1, 2))
        # assign empty slots to -100 --> softmax(-100) = 0
        att_val_qc[~_qc_mask.to(dtype=torch.uint8)] = -100

        # att_prob_qc: (B, max_col_len, max_q_len)
        att_prob_qc = self.softmax(att_val_qc)
        # q_weighted: (B, max_col_len, N_h)
        # q_enc.unsqueeze(1): (B, 1, max_q_len, N_h)
        # att_prob_qc.unsqueeze(3): (B, max_col_len, max_q_len, 1)
        q_weighted = (q_enc.unsqueeze(1) * att_prob_qc.unsqueeze(3)).sum(2)

        # Same as the above, compute SQL history embedding weighted by column attentions
        # att_val_hc: (B, max_col_len, max_hs_len)
        # col_enc: (B, max_col_len, N_h); hs_enc: (B, max_hs_len, N_h) --MLP(T(1,2))--> (B, N_h, max_hs_len)
        att_val_hc = torch.bmm(col_enc, self.hs_att(hs_enc).transpose(1, 2))
        # assign empty slots to -100 --> softmax(-100) = 0
        att_val_hc[~_hc_mask.to(dtype=torch.uint8)] = -100

        # att_prob_hc: (B, max_col_len, max_hs_len)
        att_prob_hc = self.softmax(att_val_hc)
        # hs_weighted: (B, max_col_len, N_h)
        # hs_enc.unsqueeze(1): (B, 1, max_hs_len, N_h); att_prob_hc.unsqueeze(3): (B, max_col_len, max_hs_len, 1)
        hs_weighted = (hs_enc.unsqueeze(1) * att_prob_hc.unsqueeze(3)).sum(2)
        # Compute prediction scores
        # col_score: (B, max_col_len)
        # q_weighted --MLP--> (B, max_col_len, N_h)
        # hs_weighted --MLP--> (B, max_col_len, N_h)
        # col_enc --MLP--> (B, max_col_len, N_h)
        col_score = self.col_out(self.col_out_q(q_weighted) +
                                 int(self.use_hs) * self.col_out_hs(hs_weighted) +
                                 self.col_out_c(col_enc)).squeeze(-1)
        # col_num_score: (B, num_labels); col_score: (B, max_col_len)
        # score = (col_num_score, col_score)
        return col_num_score, col_score

    def NLL_loss(self, score_tensor, truth_tensor):
        loss = 0
        # col_num_score: (B, num_labels); col_score: (B, max_col_len)
        col_num_score, col_score = score_tensor
        # truth_num_var: (B, ); truth_var: (B, max_col_len)
        truth_num_var, truth_var = truth_tensor
        # --------------------------------------------------------------------------------
        #                       loss for the column number
        # --------------------------------------------------------------------------------
        loss += self.CE(col_num_score, truth_num_var)
        # --------------------------------------------------------------------------------
        #                           loss for the key words
        # --------------------------------------------------------------------------------
        pred_prob = self.sigmoid(col_score)
        bce_loss = torch.mean(- 3 * (truth_var * torch.log(pred_prob + 1e-10))
                              - (1 - truth_var) * torch.log(1 - pred_prob + 1e-10))
        loss += bce_loss
        # loss += self.BCE(col_score, truth_var)
        return loss

    def norm_scores(self, score_tensor):
        # col_num_score: (B, num_labels); col_score: (B, max_col_len)
        col_num_score, col_score = score_tensor
        return torch.softmax(col_num_score, dim=-1).data.cpu().numpy(), torch.sigmoid(col_score).data.cpu().numpy()

    def check_acc(self, score_tensor, truth_labels):
        num_err, err, tot_err = 0, 0, 0
        B = len(truth_labels)
        pred = []
        col_num_score, col_score = self.norm_scores(score_tensor)
        for b in range(B):
            cur_pred = {}
            # when col_num_score[b] = 0 --> select 1 element
            # col_num = [1, num_labels]
            # col_num = np.argmax(col_num_score[b]) + 1  # idx starting from 0 --> a[idx+1] = max(a)
            col_num = np.argmax(col_num_score[b])
            cur_pred['col_num'] = col_num
            cur_pred['col'] = np.argsort(-col_score[b])[:col_num]
            pred.append(cur_pred)

        for b, (p, t) in enumerate(zip(pred, truth_labels)):
            col_num, col = p['col_num'], p['col']
            flag = True
            if col_num != len(t):  # double check truth format and for test cases
                num_err += 1
                flag = False
            # to eval col predicts, if the gold sql has JOIN and foreign key col, then both fks are acceptable
            fk_list = []
            regular = []
            for l in t:
                if isinstance(l, list):
                    fk_list.append(l)
                else:
                    regular.append(l)

            if flag:  # double check
                for c in col:
                    for fk in fk_list:
                        if c in fk:
                            fk_list.remove(fk)
                    for r in regular:
                        if c == r:
                            regular.remove(r)

                if len(fk_list) != 0 or len(regular) != 0:
                    err += 1
                    flag = False

            if not flag:
                tot_err += 1

        return np.array((num_err, err, tot_err))


class Word_Encoder(nn.Module):
    """
    The model builds character biLSTM, concatenated by word embeddings with attentional mechanism
    to pass through another biLSTM for extracting final features for affine layers
    """

    def __init__(self, word_HPs):
        super(Word_Encoder, self).__init__()
        self.word_nn_embs = RNN_layer(word_HPs)

    def forward(self, emb_inputs, word_lengths, init_hidden=None):
        return self.get_all_hiddens(emb_inputs, word_lengths, init_hidden)

    def get_all_hiddens(self, emb_inputs, word_lengths, init_hidden=None):
        rnn_out, hidden_out = self.word_nn_embs.get_all_hiddens(emb_inputs, word_lengths, init_hidden)
        return rnn_out, hidden_out

    def get_last_hiddens(self, emb_inputs, word_lengths, init_hidden=None):
        h_n = self.word_nn_embs.get_last_hiddens(emb_inputs, word_lengths, init_hidden)
        return h_n


class Word_alignment(nn.Module):
    def __init__(self, in_features, out_features):
        super(Word_alignment, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

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
        if input_mask is not None:
            out2[~input_mask] = -100
        # out2: [batch, seq_length1, seq_length2]
        # input1: [batch, seq_length1, in_features]
        satt = torch.bmm(F.softmax(out2, dim=1).transpose(1, -1), input1)
        # satt: [batch, seq_length2, in_features]
        return satt


class Col_awareness(nn.Module):
    def __init__(self, enc_features, col_features):
        super(Col_awareness, self).__init__()
        self.enc_features = enc_features
        self.col_features = col_features
        self.weight = Parameter(torch.Tensor(col_features, enc_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, input1, input2, input_mask=None):
        """

        :param input1: [batch, enc_features]
        :param input2: [batch, seq_length2, col_features]
        :param input_mask: mask of input1
        :return:
        """
        out1 = F.linear(input1, self.weight)
        # out1: [batch, out_features] -- > out1.unsqueeze(1): [batch, 1, out_features]
        # input2: [batch, seq_length2, out_features]
        out2 = torch.bmm(out1.unsqueeze(1), input2.transpose(1, -1))
        # TODO: use mask tensor to filter out padding in out2[:,seq_length1,:]
        if input_mask is not None:
            out2[~input_mask] = -100
        # out2: [batch, 1, seq_length2]
        # input2: [batch, seq_length2, col_features]
        satt = torch.bmm(F.softmax(out2, dim=2), input2)
        # satt: [batch, 1, col_features]
        return satt


class Word_Decoder(nn.Module):
    """
    The model builds character biLSTM, concatenated by word embeddings with attentional mechanism
    to pass through another biLSTM for extracting final features for affine layers
    """

    def __init__(self, word_HPs, drop_rate=0.5, num_labels=None, enc_att=False, sch_att=False):
        super(Word_Decoder, self).__init__()
        self.embedding = Emb_layer(word_HPs[0])
        self.decoder = Word_Encoder(word_HPs[1])
        self.num_labels = num_labels
        self.enc_att = enc_att
        self.sch_att = sch_att
        emb_dim = word_HPs[1][1]
        hidden_dim = word_HPs[1][2]
        fn_dim = hidden_dim
        self.finaldrop_layer = nn.Dropout(drop_rate)
        if enc_att:
            self.enc_attention = Word_alignment(hidden_dim, hidden_dim)
            fn_dim += hidden_dim
        if sch_att:
            self.tb_attention = Col_awareness(hidden_dim, emb_dim)
            fn_dim += emb_dim
        if num_labels > 2:
            self.hidden2tag_layer = nn.Linear(fn_dim, num_labels)
            self.lossF = nn.CrossEntropyLoss()
        else:
            self.hidden2tag_layer = nn.Linear(fn_dim, 1)
            self.lossF = nn.BCEWithLogitsLoss()

    def forward(self, word_inputs, word_lengths, init_hidden=None,
                enc_out=None, enc_mask=None, enc_hn=None, tbemb=None):
        rnn_out, _ = self.get_all_hiddens(word_inputs, word_lengths, init_hidden, enc_out, enc_mask, enc_hn, tbemb)
        label_score = self.scoring(rnn_out)
        return label_score

    def get_all_hiddens(self, word_inputs, word_lengths, init_hidden, enc_out, enc_mask, enc_hn, tbemb):
        emb_inputs = self.embedding(word_inputs)
        rnn_out, hidden_out = self.decoder.get_all_hiddens(emb_inputs, word_lengths, init_hidden)
        if self.enc_att:
            # enc_context: [batch, seq_length2, hidden_dim]
            enc_context = self.enc_attention(enc_out, rnn_out, enc_mask)
            rnn_out = torch.cat((rnn_out, enc_context), dim=-1)
        if self.sch_att:
            # tb_context: [batch, 1, col_features]
            tb_context = self.tb_attention(enc_hn, tbemb)
            # tb_context: [batch, seq_length2, col_features]
            tb_context = tb_context.expand(-1, rnn_out.size(1), -1)
            rnn_out = torch.cat((rnn_out, tb_context), dim=-1)
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
    import torch
    import torch.nn as nn
    from np5.utils.preprocessing import JSON
    from np5.utils.data_utils import Jsonfile, Vocab, seqPAD, Data2tensor, PAD, SOT
    device = torch.device("cpu")
    dtype = torch.long
    db_file = "../../data/nl2sql/tables.json"
    filename = "../../data/mysemQL/train.json"
    data = JSON.load(filename)
    streamdata = Jsonfile(filename, db_file)
    sdata = []
    for i in streamdata:
        sdata.append(i)

    # tb = streamdata.getts("department_management")
    s_paras = [None,  1]
    t_paras = [None, 1]
    vocab = Vocab(s_paras, t_paras)
    vocab.build([filename], db_file)

    nl2ids = vocab.tb2idx(vocab_words=vocab.sw2i, unk_words=True, eos=True)
    # tb2ids = vocab.tb2idx(vocab_words=vocab.sw2i, unk_words=True)
    tg2ids = vocab.wd2idx(vocab_words=vocab.tw2i, unk_words=False,  sos=True, eos=True)

    train_data = Jsonfile(filename, db_file, source2idx=nl2ids, target2idx=tg2ids)
    ts_dict = train_data.tsdict(vocab.dbids)
    data_idx = []
    batch = 16
    for d in vocab.minibatches(train_data, batch):
        data_idx.append(d)
        dbid, nl, tp, target = list(zip(*d))

        if len(vocab.dbids) == 1:
            tbs = [ts_dict[vocab.dbids[0]]]
        else:
            tbs = []
            for db in dbid:
                tbs.append(ts_dict[db])

        nl_pad_ids, nl_lens = seqPAD.pad_sequences(nl, pad_tok=vocab.sw2i[PAD], nlevels=2)
        tp_pad_ids, tp_lens = seqPAD.pad_sequences(tp, pad_tok=vocab.sw2i[PAD], nlevels=2)
        tb_pad_ids, tb_lens = seqPAD.pad_sequences(tbs, pad_tok=vocab.sw2i[PAD], nlevels=3)

        assert tp_lens == nl_lens

        lb_pad_ids, lb_lens = seqPAD.pad_sequences(target, pad_tok=vocab.tw2i[PAD], nlevels=1)

        nl_tensors = Data2tensor.sort_tensors(nl_pad_ids, nl_lens, dtype=torch.long, device=device)
        nl_tensor, nl_len_tensor, nl_ord_tensor, nl_recover_ord_tensor, _, _, _ = nl_tensors

        tp_tensor = Data2tensor.idx2tensor(tp_pad_ids, dtype=torch.long, device=device)
        tp_tensor = tp_tensor[nl_ord_tensor]

        # tp_tensors = Data2tensor.sort_tensors(tp_pad_ids, tp_lens, dtype=torch.long, device=device)
        # tp_tensor, tp_len_tensor, tp_ord_tensor, tp_recover_ord_tensor, _, _, _ = tp_tensors

        tb_tensors = Data2tensor.sort_tensors(tb_pad_ids, tb_lens, dtype=torch.long, device=device)
        tb_tensor, tb_len_tensor, tb_ord_tensor, tb_recover_ord_tensor, _, _, _ = tb_tensors

        lb_tensors = Data2tensor.sort_labelled_tensors(lb_pad_ids, lb_lens, label=True, dtype=torch.long, device=device)
        olb_tensor, ilb_tensor, lb_len_tensor, lb_ord_tensor, lb_recover_ord_tensor, _, _, _ = lb_tensors
        break

    emb_size = len(vocab.sw2i)
    emb_dim = 25
    emb_pretrained = None
    emb_drop_rate = 0.5
    emb_zero_padding = True
    requires_grad = True

    nn_mode = "lstm"
    nn_inp_dim = 25
    nn_out_dim = 100
    nn_layers = 2
    nn_bidirect = False
    nn_dropout = 0.5

    fn_dropout = 0.5

    # [size, dim, pre_embs, drop_rate, zero_padding, requires_grad] = HPs
    emb_hps = [emb_size, emb_dim, emb_pretrained, emb_drop_rate, emb_zero_padding, requires_grad]
    sembeddings = Emb_layer(emb_hps)
    # nlemb: (batch, q_len, emb_size)
    nlemb = sembeddings(nl_tensor).sum(dim=-2)
    # tpemb: (batch, q_len, emb_size)
    tpemb = sembeddings(tp_tensor).sum(dim=-2)
    # enc_emb: (batch, q_len, emb_size)
    enc_emb = nlemb + tpemb
    # tsemb: (batch, col_len, emb_size)
    tbemb = sembeddings(tb_tensor).sum(dim=-2).sum(dim=-2)

    # [nn_mode, nn_inp_dim, nn_out_dim, nn_layers, nn_bidirect, nn_dropout] = HPs
    ernn_hps = [nn_mode, nn_inp_dim, nn_out_dim, nn_layers, nn_bidirect, nn_dropout]
    encoder = Word_Encoder(ernn_hps)
    en_output, en_hidden = encoder(enc_emb, nl_len_tensor)

    if isinstance(en_hidden, tuple):
        de_hidden = tuple(hidden[:, nl_recover_ord_tensor, :] for hidden in en_hidden)
        de_hidden = tuple(hidden[:, lb_ord_tensor, :] for hidden in de_hidden)
        if nn_bidirect:
            en_hn = torch.cat((en_hidden[0][-2, :, :], en_hidden[0][-1, :, :]), -1)
        else:
            en_hn = en_hidden[0][-1, :, :]
    else:
        de_hidden = en_hidden[:, nl_recover_ord_tensor, :]
        de_hidden = de_hidden[:, lb_ord_tensor, :]
        if nn_bidirect:
            en_hn = torch.cat((en_hidden[-2, :, :], en_hidden[-1, :, :]), -1)
        else:
            en_hn = en_hidden[-1, :, :]

    # de_hidden = en_hidden_re_sort
    enc_out = en_output[nl_recover_ord_tensor, :, :]
    enc_out = enc_out[lb_ord_tensor, :, :]
    # enc_out = en_output_sort
    nl_len_tensor = nl_len_tensor[nl_recover_ord_tensor]
    nl_len_tensor = nl_len_tensor[lb_ord_tensor]
    enc_mask = torch.arange(max(nl_len_tensor))[None, :] < nl_len_tensor[:, None]
    # [size, dim, pre_embs, drop_rate, zero_padding, requires_grad] = HPs
    emb_hps = [emb_size, emb_dim, emb_pretrained, emb_drop_rate, emb_zero_padding, requires_grad]
    # tembeddings = Emb_layer(emb_hps)
    # lb_emb = tembeddings(ilb_tensor)
    # [nn_mode, nn_inp_dim, nn_out_dim, nn_layers, nn_bidirect, nn_dropout] = HPs
    drnn_hps = [nn_mode, nn_inp_dim, nn_out_dim, nn_layers, nn_bidirect, nn_dropout]

    num_labels = len(vocab.tw2i)
    decoder = Word_Decoder([emb_hps, drnn_hps], drop_rate=fn_dropout, num_labels=num_labels, enc_att=True, sch_att=True)
    score_tensor = decoder(ilb_tensor, lb_len_tensor, de_hidden, enc_out, enc_mask, en_hn, tbemb)
    label_mask = olb_tensor > 0
    batch_loss = decoder.NLL_loss(score_tensor[label_mask], olb_tensor[label_mask])

    batch_size, target_length = ilb_tensor.size()
    # Extract the first target word (tsos or SOT) to feed to decoder
    ilb_1tensor = Data2tensor.idx2tensor([[vocab.tw2i[SOT]]] * batch_size, dtype=torch.long,
                                         device=device)
    # t_seq_tensor = [batch_size, 1]
    lb_len_1tensor = Data2tensor.idx2tensor([1] * batch_size, dtype=torch.long, device=device)

    decoder.get_all_hiddens(ilb_1tensor, lb_len_1tensor, de_hidden, enc_out, enc_mask, en_hn, tbemb)
