"""
Created on 2018-11-27
@author: duytinvo
"""
import os
import sys
import time
import torch
import random
import argparse
import math
import csv
import copy
import numpy as np
import operator
from queue import PriorityQueue
import torch.optim as optim
from np5.utils.preprocessing import PREPROCESS, SCHEMA
from np5.utils.postprocessing import POSTPROCESS
from np5.utils.sem2sql import PARSER as sem2sqlParser
from np5.utils.data_utils import PAD, SOT, EOT
from np5.utils.bleu import compute_bleu
from np5.utils.data_utils import Progbar, Timer, SaveloadHP
from np5.utils.core_nns import Word_Encoder, Word_Decoder, Emb_layer
from np5.utils.data_utils import Vocab, Data2tensor, Jsonfile, seqPAD, Embeddings


Data2tensor.set_randseed(1234)


class BeamSearchNode(object):
    def __init__(self, t_word, t_seq_tensor, t_seq_len_tensor,
                 de_hidden, prevNode, logProb, length):
        self.t_word = t_word
        self.t_seq_tensor = t_seq_tensor
        self.t_seq_len_tensor = t_seq_len_tensor
        self.de_hidden = de_hidden
        self.prevNode = prevNode
        self.logProb = logProb
        self.length = length

    def eval(self, alpha=1.0, reward=0):
        # Add here a function for shaping a reward
        # return self.logProb / float(self.length - 1 + 1e-6) + alpha * reward
        return self.logProb


class Translator_model(object):
    def __init__(self, args=None):
        self.args = args
        self.device = torch.device("cuda:0" if self.args.use_cuda else "cpu")
        # Include SOt, EOt if set set_words, else Ignore SOt, EOt
        self.num_labels = len(self.args.vocab.tw2i)

        # Hyper-parameters at character-level source language
        self.source2idx = self.args.vocab.tb2idx(vocab_words=self.args.vocab.sw2i, unk_words=True,
                                                 sos=self.args.ssos, eos=self.args.seos)
        # Hyper-parameters at character-level target language
        self.target2idx = self.args.vocab.wd2idx(vocab_words=self.args.vocab.tw2i, unk_words=False,
                                                 sos=self.args.tsos, eos=self.args.teos,
                                                 reverse=self.args.t_reverse)

        # Hyper-parameters at word-level source language
        # [size, dim, pre_embs, drop_rate, zero_padding, requires_grad] = HPs
        semb_HPs = [len(self.args.vocab.sw2i), self.args.swd_dim, self.args.swd_pretrained,
                    self.args.wd_dropout, self.args.wd_padding, self.args.swd_reqgrad]
        self.sembedding = Emb_layer(semb_HPs)

        # Hyper-parameters at hidden-level source language
        # [nn_mode, nn_inp_dim, nn_out_dim, nn_layers, nn_bidirect, nn_dropout] = HPs
        enc_HPs = [self.args.ed_mode, self.args.swd_dim, self.args.ed_outdim,
                    self.args.ed_layers, self.args.ed_bidirect, self.args.ed_dropout]

        self.encoder = Word_Encoder(word_HPs=enc_HPs).to(self.device)

        # Hyper-parameters at word-level target language
        # [size, dim, pre_embs, drop_rate, zero_padding, requires_grad] = HPs
        temb_HPs = [len(self.args.vocab.tw2i), self.args.twd_dim, self.args.twd_pretrained,
                    self.args.wd_dropout, self.args.wd_padding, self.args.twd_reqgrad]

        # Hyper-parameters at word-level target language
        dec_HPs = [self.args.ed_mode, self.args.twd_dim, self.args.ed_outdim,
                   self.args.ed_layers, self.args.ed_bidirect, self.args.ed_dropout]

        self.decoder = Word_Decoder(word_HPs=[temb_HPs, dec_HPs], drop_rate=self.args.final_dropout,
                                    num_labels=self.num_labels, enc_att=self.args.enc_att,
                                    sch_att=self.args.sch_att).to(self.device)

        if args.optimizer.lower() == "adamax":
            self.encoder_optimizer = optim.Adamax(self.encoder.parameters(), lr=self.args.lr)
            self.decoder_optimizer = optim.Adamax(self.decoder.parameters(), lr=self.args.lr)
        elif args.optimizer.lower() == "adam":
            self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=self.args.lr)
            self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=self.args.lr)
        elif args.optimizer.lower() == "adadelta":
            self.encoder_optimizer = optim.Adadelta(self.encoder.parameters(), lr=self.args.lr)
            self.decoder_optimizer = optim.Adadelta(self.decoder.parameters(), lr=self.args.lr)
        elif args.optimizer.lower() == "adagrad":
            self.encoder_optimizer = optim.Adagrad(self.encoder.parameters(), lr=self.args.lr)
            self.decoder_optimizer = optim.Adagrad(self.decoder.parameters(), lr=self.args.lr)
        else:
            self.encoder_optimizer = optim.SGD(self.encoder.parameters(), lr=self.args.lr, momentum=0.9)
            self.decoder_optimizer = optim.SGD(self.decoder.parameters(), lr=self.args.lr, momentum=0.9)
        self.schema_dict = {}

    def greedy_predict(self, sentence, dbid):
        preporcess = PREPROCESS(self.args.table_path, self.args.kb_relatedto, self.args.kb_isa)
        entry = dict()
        entry["db_id"] = dbid
        entry["question"] = sentence
        entry['question_toks'] = sentence.split()
        preporcess.build_one(entry)

        self.encoder.eval()
        self.decoder.eval()
        twords = [SOT]
        prob = 1.0
        with torch.no_grad():
            nl = self.source2idx(entry["question_arg"])
            tp = self.source2idx(entry["question_arg_type"])
            tbs = self.source2idx(self.schema_dict[dbid])

            nl_pad_ids, nl_lens = seqPAD.pad_sequences([nl], pad_tok=self.args.vocab.sw2i[PAD], nlevels=2)

            tp_pad_ids, tp_lens = seqPAD.pad_sequences([tp], pad_tok=self.args.vocab.sw2i[PAD], nlevels=2)

            tb_pad_ids, tb_lens = seqPAD.pad_sequences([tbs], pad_tok=self.args.vocab.sw2i[PAD], nlevels=3)

            assert tp_lens == nl_lens

            nl_tensors = Data2tensor.sort_tensors(nl_pad_ids, nl_lens, dtype=torch.long, device=self.device)
            nl_tensor, nl_len_tensor, nl_ord_tensor, nl_recover_ord_tensor, _, _, _ = nl_tensors

            tp_tensor = Data2tensor.idx2tensor(tp_pad_ids, dtype=torch.long, device=self.device)
            tp_tensor = tp_tensor[nl_ord_tensor]
            # tp_tensors = Data2tensor.sort_tensors(tp_pad_ids, tp_lens, dtype=torch.long, device=self.device)
            # tp_tensor, tp_len_tensor, tp_ord_tensor, tp_recover_ord_tensor, _, _, _ = tp_tensors

            tb_tensors = Data2tensor.sort_tensors(tb_pad_ids, tb_lens, dtype=torch.long, device=self.device)
            tb_tensor, tb_len_tensor, tb_ord_tensor, tb_recover_ord_tensor, _, _, _ = tb_tensors

            # nlemb: (batch, q_len, emb_size)
            nlemb = self.sembedding(nl_tensor).sum(dim=-2)
            # hqemb: (batch, h_len, emb_size)
            tpemb = self.sembedding(tp_tensor).sum(dim=-2)
            enc_inp = nlemb + tpemb
            # tsemb: (batch, col_len, emb_size)
            tbemb = self.sembedding(tb_tensor).sum(dim=-2).sum(dim=-2)

            # source --> encoder --> source_hidden
            en_output, en_hidden = self.encoder(enc_inp, nl_len_tensor)
            if isinstance(en_hidden, tuple):
                if self.args.ed_bidirect:
                    en_hn = torch.cat((en_hidden[0][-2, :, :], en_hidden[0][-1, :, :]), -1)
                else:
                    en_hn = en_hidden[0][-1, :, :]
            else:
                if self.args.ed_bidirect:
                    en_hn = torch.cat((en_hidden[-2, :, :], en_hidden[-1, :, :]), -1)
                else:
                    en_hn = en_hidden[-1, :, :]

            de_hidden = en_hidden
            enc_mask = torch.arange(max(nl_len_tensor), dtype=torch.long, device=self.device)[None, :] < nl_len_tensor[:, None]
            batch_size = 1
            # Extract the first target word (tsos or SOT) to feed to decoder
            t_seq_tensor = Data2tensor.idx2tensor([[self.args.vocab.tw2i[SOT]]] * batch_size, dtype=torch.long,
                                                  device=self.device)
            t_seq_len_tensor = Data2tensor.idx2tensor([1] * batch_size, dtype=torch.long, device=self.device)

            while True:
                de_output, de_hidden = self.decoder.get_all_hiddens(t_seq_tensor, t_seq_len_tensor,
                                                                    de_hidden, en_output, enc_mask, en_hn, tbemb)
                de_score = self.decoder.scoring(de_output)
                label_prob, label_pred = self.decoder.inference(de_score, 1)
                # label_prob = label_pred = [batch_size, 1, 1]
                t_seq_tensor = label_pred.squeeze(-1).detach()  # detach from history as input
                # t_seq_tensor = [batch_size, 1]

                tword = Vocab.idx2text(t_seq_tensor.squeeze(-1).tolist(), self.args.vocab.i2tw, 1)
                twords += tword

                prob *= label_prob.squeeze().item()
                if tword[0] == EOT:
                    break

            entry['model_result'] = " ".join(twords)

            schemas = SCHEMA(self.args.schema_file).table_dict
            postprocess = POSTPROCESS()
            simqlparser = sem2sqlParser()

            schema = schemas[dbid]
            postprocess.build(entry, schema)
            result, transformed_sql = simqlparser.transform(entry, schema)
            entry['pred_query'] = result
            return entry, prob

    def beam_predict(self, sentence, dbid, bw=2, topk=2):
        """

        :param sentence: source sentence
        :param bw: beam width
        :param topk: number of target sentences (bw <= topk <= bw**2)
        :return: target sentences with confident scores
        """
        preporcess = PREPROCESS(self.args.table_path, self.args.kb_relatedto, self.args.kb_isa)
        entry = dict()
        entry["db_id"] = dbid
        entry["question"] = sentence
        entry['question_toks'] = sentence.split()
        preporcess.build_one(entry)

        self.encoder.eval()
        self.decoder.eval()
        decoded_batch = []
        with torch.no_grad():
            nl = self.source2idx(entry["question_arg"])
            tp = self.source2idx(entry["question_arg_type"])
            tbs = self.source2idx(self.schema_dict[dbid])

            nl_pad_ids, nl_lens = seqPAD.pad_sequences([nl], pad_tok=self.args.vocab.sw2i[PAD], nlevels=2)

            tp_pad_ids, tp_lens = seqPAD.pad_sequences([tp], pad_tok=self.args.vocab.sw2i[PAD], nlevels=2)

            tb_pad_ids, tb_lens = seqPAD.pad_sequences([tbs], pad_tok=self.args.vocab.sw2i[PAD], nlevels=3)

            assert tp_lens == nl_lens

            nl_tensors = Data2tensor.sort_tensors(nl_pad_ids, nl_lens, dtype=torch.long, device=self.device)
            nl_tensor, nl_len_tensor, nl_ord_tensor, nl_recover_ord_tensor, _, _, _ = nl_tensors

            tp_tensor = Data2tensor.idx2tensor(tp_pad_ids, dtype=torch.long, device=self.device)
            tp_tensor = tp_tensor[nl_ord_tensor]
            # tp_tensors = Data2tensor.sort_tensors(tp_pad_ids, tp_lens, dtype=torch.long, device=self.device)
            # tp_tensor, tp_len_tensor, tp_ord_tensor, tp_recover_ord_tensor, _, _, _ = tp_tensors

            tb_tensors = Data2tensor.sort_tensors(tb_pad_ids, tb_lens, dtype=torch.long, device=self.device)
            tb_tensor, tb_len_tensor, tb_ord_tensor, tb_recover_ord_tensor, _, _, _ = tb_tensors

            enc_mask = nl_tensor > 0

            # nlemb: (batch, q_len, emb_size)
            nlemb = self.sembedding(nl_tensor).sum(dim=-2)
            # hqemb: (batch, h_len, emb_size)
            tpemb = self.sembedding(tp_tensor).sum(dim=-2)
            enc_inp = nlemb + tpemb
            # tsemb: (batch, col_len, emb_size)
            tbemb = self.sembedding(tb_tensor).sum(dim=-2).sum(dim=-2)

            # source --> encoder --> source_hidden
            en_output, en_hidden = self.encoder(enc_inp, nl_len_tensor)
            if isinstance(en_hidden, tuple):
                if self.args.ed_bidirect:
                    en_hn = torch.cat((en_hidden[0][-2, :, :], en_hidden[0][-1, :, :]), -1)
                else:
                    en_hn = en_hidden[0][-1, :, :]
            else:
                if self.args.ed_bidirect:
                    en_hn = torch.cat((en_hidden[-2, :, :], en_hidden[-1, :, :]), -1)
                else:
                    en_hn = en_hidden[-1, :, :]

            de_hidden = en_hidden
            enc_mask = torch.arange(max(nl_len_tensor), dtype=torch.long, device=self.device)[None, :] < nl_len_tensor[:, None]
            batch_size = 1
            # Extract the first target word (tsos or SOT) to feed to decoder
            t_seq_tensor = Data2tensor.idx2tensor([[self.args.vocab.tw2i[SOT]]] * batch_size, dtype=torch.long,
                                                  device=self.device)
            # t_seq_tensor = [batch_size, 1]
            t_seq_len_tensor = Data2tensor.idx2tensor([1] * batch_size, dtype=torch.long, device=self.device)

            # Number of sentence to generate
            endnodes = []

            # starting node -  t_seq_tensor, t_seq_len_tensor,
            #                  de_hidden, previous node, logp, length
            node = BeamSearchNode(SOT, t_seq_tensor, t_seq_len_tensor, de_hidden, None, 0, 1)
            nodes = PriorityQueue()

            # start the queue
            nodes.put((-node.eval(), node))
            qsize = 1

            # start beam search
            while True:
                # give up when decoding takes too long
                if qsize > 2000:
                    break

                # fetch the best node
                score, n = nodes.get()
                t_seq_tensor = n.t_seq_tensor
                t_seq_len_tensor = n.t_seq_len_tensor
                de_hidden = n.de_hidden

                if n.t_word == EOT and n.prevNode is not None:
                    endnodes.append((score, n))
                    # if we reached maximum # of sentences required
                    if len(endnodes) >= topk:
                        break
                    else:
                        continue

                # decode for one step using decoder
                de_output, de_hidden = self.decoder.get_all_hiddens(t_seq_tensor, t_seq_len_tensor,
                                                                    de_hidden, en_output, enc_mask, en_hn, tbemb)
                de_score = self.decoder.scoring(de_output)
                # de_score = [batch_size, 1, num_labels]
                # PUT HERE REAL BEAM SEARCH OF TOP
                log_prob, label_pred = self.decoder.logsm_inference(de_score, bw)
                # label_prob = label_pred = [batch_size, 1, bw]
                nextnodes = []
                for new_k in range(bw):
                    t_seq_tensor = label_pred[:, :, new_k].detach()  # detach from history as input
                    # t_seq_tensor = [batch_size, 1]
                    logProb = log_prob[:, :, new_k].squeeze().item()
                    t_word = Vocab.idx2text(t_seq_tensor.squeeze(-1).tolist(), self.args.vocab.i2tw, 1)

                    node = BeamSearchNode(t_word[0], t_seq_tensor, t_seq_len_tensor,
                                          de_hidden, n, n.logProb + logProb, n.length + 1)
                    score = -node.eval()
                    nextnodes.append((score, node))

                # put them into queue
                for i in range(len(nextnodes)):
                    score, nn = nextnodes[i]
                    nodes.put((score, nn))
                    # increase qsize
                qsize += len(nextnodes) - 1

            # choose nbest paths, back trace them
            if len(endnodes) == 0:
                # even if the beam search doesn't reach EOT, we still pop the topk best paths for inference
                endnodes = [nodes.get() for _ in range(topk)]

            utterances = []
            for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                utterance = list()
                utterance.append(n.t_word)
                # if n.t_word == EOT and n.prevNode is not None:
                #     logProb = n.prevNode.logProb
                # else:
                #     logProb = n.logProb
                logProb = n.logProb
                # back trace
                while n.prevNode is not None:
                    n = n.prevNode
                    utterance.append(n.t_word)

                utterance = utterance[::-1]
                utterances.append((" ".join(utterance), math.exp(logProb)))

            decoded_batch.extend(utterances)

            entry['pred_query'] = []
            for sem, prob in decoded_batch:
                entry['model_result'] = sem

                schemas = SCHEMA(self.args.schema_file).table_dict
                postprocess = POSTPROCESS()
                simqlparser = sem2sqlParser()

                schema = schemas[dbid]
                postprocess.build(entry, schema)
                result, transformed_sql = simqlparser.transform(entry, schema)
                entry['pred_query'].append((result, prob))
            entry['model_result'] = decoded_batch

        return decoded_batch

    def evaluate_batch(self, eva_data):
        start = time.time()
        self.encoder.eval()
        self.decoder.eval()
        dev_loss = []
        total_tokens = 0
        reference = []
        candidate = []
        with torch.no_grad():
            for i, d in enumerate(self.args.vocab.minibatches(eva_data, batch_size=self.args.batch_size)):
                dbid, nl, tp, target = list(zip(*d))
                if len(self.args.vocab.dbids) == 1:
                    tbs = [self.schema_dict[self.args.vocab.dbids[0]]]
                else:
                    tbs = []
                    for db in dbid:
                        tbs.append(self.schema_dict[db])

                nl_pad_ids, nl_lens = seqPAD.pad_sequences(nl, pad_tok=self.args.vocab.sw2i[PAD], nlevels=2)

                tp_pad_ids, tp_lens = seqPAD.pad_sequences(tp, pad_tok=self.args.vocab.sw2i[PAD], nlevels=2)

                tb_pad_ids, tb_lens = seqPAD.pad_sequences(tbs, pad_tok=self.args.vocab.sw2i[PAD], nlevels=3)

                lb_pad_ids, lb_lens = seqPAD.pad_sequences(target, pad_tok=self.args.vocab.tw2i[PAD], nlevels=1)

                assert tp_lens == nl_lens

                nl_tensors = Data2tensor.sort_tensors(nl_pad_ids, nl_lens, dtype=torch.long, device=self.device)
                nl_tensor, nl_len_tensor, nl_ord_tensor, nl_recover_ord_tensor, _, _, _ = nl_tensors

                tp_tensor = Data2tensor.idx2tensor(tp_pad_ids, dtype=torch.long, device=self.device)
                tp_tensor = tp_tensor[nl_ord_tensor]
                # tp_tensors = Data2tensor.sort_tensors(tp_pad_ids, tp_lens, dtype=torch.long, device=self.device)
                # tp_tensor, tp_len_tensor, tp_ord_tensor, tp_recover_ord_tensor, _, _, _ = tp_tensors

                tb_tensors = Data2tensor.sort_tensors(tb_pad_ids, tb_lens, dtype=torch.long, device=self.device)
                tb_tensor, tb_len_tensor, tb_ord_tensor, tb_recover_ord_tensor, _, _, _ = tb_tensors

                lb_tensors = Data2tensor.sort_labelled_tensors(lb_pad_ids, lb_lens, label=True, dtype=torch.long,
                                                               device=self.device)
                olb_tensor, ilb_tensor, lb_len_tensor, lb_ord_tensor, lb_recover_ord_tensor, _, _, _ = lb_tensors

                # nlemb: (batch, q_len, emb_size)
                nlemb = self.sembedding(nl_tensor).sum(dim=-2)
                # hqemb: (batch, h_len, emb_size)
                tpemb = self.sembedding(tp_tensor).sum(dim=-2)
                enc_inp = nlemb + tpemb
                # tsemb: (batch, col_len, emb_size)
                tbemb = self.sembedding(tb_tensor).sum(dim=-2).sum(dim=-2)

                # source --> encoder --> source_hidden
                en_output, en_hidden = self.encoder(enc_inp, nl_len_tensor)
                # enc_mask = nl_tensor > 0
                # en_output = tensor(batch_size, seq_length, rnn_dim * num_directions)
                # en_hidden = (h_n,c_n); h_n = c_n = tensor(num_layers * num_directions, batch_size, rnn_dim)

                # Re-ordering and sort encoding outputs following target
                # en_output_re = en_output[sseqord_recover_tensor]
                # en_output_re_sort = en_output_re[tseqord_tensor]
                if isinstance(en_hidden, tuple):
                    de_hidden = tuple(hidden[:, nl_recover_ord_tensor, :] for hidden in en_hidden)
                    de_hidden = tuple(hidden[:, lb_ord_tensor, :] for hidden in de_hidden)
                    if self.args.ed_bidirect:
                        en_hn = torch.cat((en_hidden[0][-2, :, :], en_hidden[0][-1, :, :]), -1)
                    else:
                        en_hn = en_hidden[0][-1, :, :]
                else:
                    de_hidden = en_hidden[:, nl_recover_ord_tensor, :]
                    de_hidden = de_hidden[:, lb_ord_tensor, :]
                    if self.args.ed_bidirect:
                        en_hn = torch.cat((en_hidden[-2, :, :], en_hidden[-1, :, :]), -1)
                    else:
                        en_hn = en_hidden[-1, :, :]
                # de_hidden = en_hidden_re_sort

                enc_out = en_output[nl_recover_ord_tensor, :, :]
                enc_out = enc_out[lb_ord_tensor, :, :]
                # enc_out = en_output_sort
                nl_len_tensor = nl_len_tensor[nl_recover_ord_tensor]
                nl_len_tensor = nl_len_tensor[lb_ord_tensor]
                enc_mask = torch.arange(max(nl_len_tensor), dtype=torch.long, device=self.device)[None, :] < nl_len_tensor[:, None]

                batch_size, target_length = ilb_tensor.size()
                # Extract the first target word (tsos or SOT) to feed to decoder
                ilb_1tensor = Data2tensor.idx2tensor([[self.args.vocab.tw2i[SOT]]] * batch_size, dtype=torch.long,
                                                     device=self.device)
                # t_seq_tensor = [batch_size, 1]
                lb_len_1tensor = Data2tensor.idx2tensor([1] * batch_size, dtype=torch.long, device=self.device)
                # t_seq_len_tensor = [batch_size]

                count_tokens = 0
                batch_loss = 0
                # Ignore SOT
                label_words = []
                predict_words = []
                for j in range(target_length):
                    de_output, de_hidden = self.decoder.get_all_hiddens(ilb_1tensor, lb_len_1tensor, de_hidden,
                                                                        enc_out, enc_mask, en_hn, tbemb)
                    # de_output = [batch_size, 1, hidden_dim]
                    de_score = self.decoder.scoring(de_output)
                    # de_score = [batch_size, 1, num_labels]
                    label_prob, label_pred = self.decoder.inference(de_score, 1)
                    # label_prob = label_pred = [batch_size, 1, 1]
                    ilb_1tensor = label_pred.squeeze(-1).detach()  # detach from history as input
                    # t_seq_tensor = [batch_size, 1]

                    labels = olb_tensor[:, j]
                    label_mask = labels > 0
                    # labels = label_mask = [batch_size]
                    label_words += [Vocab.idx2text(labels.squeeze().tolist(), self.args.vocab.i2tw, 1)]
                    predict_words += [Vocab.idx2text(ilb_1tensor.squeeze().tolist(), self.args.vocab.i2tw, 1)]
                    count_tokens += label_mask.sum().item()
                    batch_loss += self.decoder.NLL_loss(de_score[label_mask], labels[label_mask])

                batch_loss = batch_loss / count_tokens
                dev_loss.append(batch_loss.item())
                total_tokens += count_tokens
                # label_words = [[w1, ..., w1], ..., [EOT, ..., EOT]]
                # label_words = list(zip(*label_words))
                # label_words = [[w1, ..., EOT], ..., [w1, ..., EOT]]
                label_words = Translator_model.filter_pad(label_words, (olb_tensor > 0).sum(dim=1).tolist())

                # predict_words = [[w1, ..., w1], ..., [EOT, ..., EOT]]
                # predict_words = list(zip(*predict_words)s)
                # predict_words = [[w1, ..., EOT], ..., [w1, ..., EOT]]
                predict_words = Translator_model.filter_pad(predict_words, (olb_tensor > 0).sum(dim=1).tolist())

                reference.extend(label_words)
                # reference = [[w1, ..., EOT], ..., [w1, ..., EOT]]
                if sum([len(k) for k in predict_words]) != 0:
                    candidate.extend(predict_words)
                # candidate = [[w1, ..., EOT], ..., [w1, ..., EOT]]
                # if batch_size != self.args.batch_size:
                #     print("\n")
                #     print(len(label_words))
                #     print("Sample gold labels: ", reference[: self.args.batch_size])
                #     print("sample predicted labels: ", candidate[: self.args.batch_size])
                #     print("\n")
        if len(candidate) != 0:
            bleu_score = Translator_model.class_metrics(list(zip(reference)), candidate)
        else:
            bleu_score = [0.0]
        # print(bleu_score)
        end = time.time() - start
        speed = total_tokens / end
        return np.mean(dev_loss), bleu_score, speed

    def train_batch(self, train_data):
        clip_rate = self.args.clip
        batch_size = self.args.batch_size
        num_train = train_data.length
        total_batch = num_train // batch_size + 1
        prog = Progbar(target=total_batch)
        # set model in train model
        self.encoder.train()
        self.decoder.train()
        train_loss = []
        self.schema_dict = train_data.tsdict(self.args.vocab.dbids)
        for i, d in enumerate(self.args.vocab.minibatches(train_data, batch_size=batch_size)):
            dbid, nl, tp, target = list(zip(*d))
            if len(self.args.vocab.dbids) == 1:
                tbs = [self.schema_dict[self.args.vocab.dbids[0]]]
            else:
                tbs = []
                for db in dbid:
                    tbs.append(self.schema_dict[db])

            nl_pad_ids, nl_lens = seqPAD.pad_sequences(nl, pad_tok=self.args.vocab.sw2i[PAD], nlevels=2)

            tp_pad_ids, tp_lens = seqPAD.pad_sequences(tp, pad_tok=self.args.vocab.sw2i[PAD], nlevels=2)

            tb_pad_ids, tb_lens = seqPAD.pad_sequences(tbs, pad_tok=self.args.vocab.sw2i[PAD], nlevels=3)

            lb_pad_ids, lb_lens = seqPAD.pad_sequences(target, pad_tok=self.args.vocab.tw2i[PAD], nlevels=1)

            assert tp_lens == nl_lens

            nl_tensors = Data2tensor.sort_tensors(nl_pad_ids, nl_lens, dtype=torch.long, device=self.device)
            nl_tensor, nl_len_tensor, nl_ord_tensor, nl_recover_ord_tensor, _, _, _ = nl_tensors

            tp_tensor = Data2tensor.idx2tensor(tp_pad_ids, dtype=torch.long, device=self.device)
            tp_tensor = tp_tensor[nl_ord_tensor]
            # tp_tensors = Data2tensor.sort_tensors(tp_pad_ids, tp_lens, dtype=torch.long, device=self.device)
            # tp_tensor, tp_len_tensor, tp_ord_tensor, tp_recover_ord_tensor, _, _, _ = tp_tensors

            tb_tensors = Data2tensor.sort_tensors(tb_pad_ids, tb_lens, dtype=torch.long, device=self.device)
            tb_tensor, tb_len_tensor, tb_ord_tensor, tb_recover_ord_tensor, _, _, _ = tb_tensors

            lb_tensors = Data2tensor.sort_labelled_tensors(lb_pad_ids, lb_lens, label=True, dtype=torch.long,
                                                           device=self.device)
            olb_tensor, ilb_tensor, lb_len_tensor, lb_ord_tensor, lb_recover_ord_tensor, _, _, _ = lb_tensors

            # nlemb: (batch, q_len, emb_size)
            nlemb = self.sembedding(nl_tensor).sum(dim=-2)
            # tpemb: (batch, h_len, emb_size)
            tpemb = self.sembedding(tp_tensor).sum(dim=-2)

            enc_inp = nlemb + tpemb
            # tbemb: (batch, col_len, emb_size)
            tbemb = self.sembedding(tb_tensor).sum(dim=-2).sum(dim=-2)

            self.encoder.zero_grad()
            self.decoder.zero_grad()
            # source --> encoder --> source_hidden
            en_output, en_hidden = self.encoder(enc_inp, nl_len_tensor)
            # en_output = tensor(batch_size, seq_length, rnn_dim * num_directions)
            # en_hidden = (h_n,c_n); h_n = c_n = tensor(num_layers * num_directions, batch_size, rnn_dim)

            # Re-ordering and sort encoding outputs following target order
            # --> en_output_re = en_output[sseqord_recover_tensor]: re-oder to the original order of the source language
            # --> en_output_re_sort = en_output_re[tseqord_tensor]: follow-up the other of the target language
            if isinstance(en_hidden, tuple):
                de_hidden = tuple(hidden[:, nl_recover_ord_tensor, :] for hidden in en_hidden)
                de_hidden = tuple(hidden[:, lb_ord_tensor, :] for hidden in de_hidden)
                if self.args.ed_bidirect:
                    en_hn = torch.cat((en_hidden[0][-2, :, :], en_hidden[0][-1, :, :]), -1)
                else:
                    en_hn = en_hidden[0][-1, :, :]
            else:
                de_hidden = en_hidden[:, nl_recover_ord_tensor, :]
                de_hidden = de_hidden[:, lb_ord_tensor, :]
                if self.args.ed_bidirect:
                    en_hn = torch.cat((en_hidden[-2, :, :], en_hidden[-1, :, :]), -1)
                else:
                    en_hn = en_hidden[-1, :, :]

            enc_out = en_output[nl_recover_ord_tensor, :, :]
            enc_out = enc_out[lb_ord_tensor, :, :]

            nl_len_tensor = nl_len_tensor[nl_recover_ord_tensor]
            nl_len_tensor = nl_len_tensor[lb_ord_tensor]
            enc_mask = torch.arange(max(nl_len_tensor), dtype=torch.long, device=self.device)[None, :] < nl_len_tensor[:, None]

            batch_loss = 0
            use_teacher_forcing = True if random.random() < self.args.teacher_forcing_ratio else False

            if use_teacher_forcing:  # directly feed the ground-truth target language to the decoder
                # tseq_tensor = [batch_size, seq_len]
                # tseq_len_tensor = [batch_size, ]
                # twd_tensor = [batch_size* seq_len, wd_len]
                # twdord_recover_tensor = twd_len_tensor: [batch_size* seq_len, ]
                score_tensor = self.decoder(ilb_tensor, lb_len_tensor, de_hidden, enc_out, enc_mask, en_hn, tbemb)
                label_mask = olb_tensor > 0
                batch_loss = self.decoder.NLL_loss(score_tensor[label_mask], olb_tensor[label_mask])
                train_loss.append(batch_loss.item())
            else:
                batch_size, target_length = ilb_tensor.size()
                # Extract the first target word (tsos or SOT) to feed to decoder
                ilb_1tensor = Data2tensor.idx2tensor([[self.args.vocab.tw2i[SOT]]] * batch_size, dtype=torch.long,
                                                      device=self.device)
                # t_seq_tensor = [batch_size, 1]
                lb_len_1tensor = Data2tensor.idx2tensor([1] * batch_size, dtype=torch.long, device=self.device)
                # t_seq_len_tensor = [batch_size]
                count_tokens = 0
                for j in range(target_length):
                    de_output, de_hidden = self.decoder.get_all_hiddens(ilb_1tensor, lb_len_1tensor,
                                                                        de_hidden, enc_out, enc_mask, en_hn, tbemb)
                    # de_output = [batch_size, 1, hidden_dim]
                    de_score = self.decoder.scoring(de_output)
                    # de_score = [batch_size, 1, num_labels]
                    label_prob, label_pred = self.decoder.inference(de_score, 1)
                    # label_prob = label_pred = [batch_size, 1, 1]
                    ilb_1tensor = label_pred.squeeze(-1).detach()  # detach from history as input
                    # t_seq_tensor = [batch_size, 1]
                    labels = olb_tensor[:, j]
                    label_mask = labels > 0
                    count_tokens += label_mask.sum().item()
                    batch_loss += self.decoder.NLL_loss(de_score[label_mask], labels[label_mask])

                batch_loss = batch_loss / count_tokens
                train_loss.append(batch_loss.item())

            batch_loss.backward()
            if clip_rate > 0:
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), clip_rate)
                torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), clip_rate)
            self.decoder_optimizer.step()
            self.encoder_optimizer.step()
            prog.update(i + 1, [("Train loss", batch_loss.item())])
        return np.mean(train_loss)

    def train(self):
        train_data = Jsonfile(self.args.train_file, self.args.schema_file,
                              source2idx=self.source2idx, target2idx=self.target2idx)
        dev_data = Jsonfile(self.args.dev_file,  self.args.schema_file,
                           source2idx=self.source2idx, target2idx=self.target2idx)
        test_data = Jsonfile(self.args.test_file,  self.args.schema_file,
                            source2idx=self.source2idx, target2idx=self.target2idx)

        encoder_filename = os.path.join(self.args.model_dir, self.args.encoder_file)
        decoder_filename = os.path.join(self.args.model_dir, self.args.decoder_file)
        max_epochs = self.args.max_epochs
        saved_epoch = 0
        best_dev = -np.inf if self.args.metric == "bleu" else np.inf
        nepoch_no_imprv = 0
        epoch_start = time.time()
        for epoch in range(1, max_epochs + 1):
            print("Epoch: %s/%s" % (epoch, max_epochs))
            train_loss = self.train_batch(train_data)
            # evaluate on developing data
            dev_loss, dev_bleu, dev_speed = self.evaluate_batch(dev_data)
            dev_metric = dev_bleu[0] if self.args.metric == "bleu" else dev_loss
            cond = dev_metric > best_dev if self.args.metric == "bleu" else dev_loss < best_dev
            if cond:
                nepoch_no_imprv = 0
                saved_epoch = epoch
                best_dev = dev_metric
                print("\nUPDATES: - New improvement")
                print("         - Train loss: %.4f" % train_loss)
                print("         - Dev loss: %.4f; Dev bleu: %.4f; Dev speed: %.2f(tokens/s)" %
                      (dev_loss, dev_bleu[0], dev_speed))
                print("         - Save the models to %s, %s at epoch %d" %
                      (encoder_filename, decoder_filename, saved_epoch))
                # Convert model to CPU to avoid out of GPU memory
                self.encoder.to("cpu")
                torch.save(self.encoder.state_dict(), encoder_filename)
                self.encoder.to(self.device)

                self.decoder.to("cpu")
                torch.save(self.decoder.state_dict(), decoder_filename)
                self.decoder.to(self.device)
            else:
                nepoch_no_imprv += 1
                if self.args.decay_rate > 0:
                    self.lr_decay(epoch)

                if nepoch_no_imprv >= self.args.patience:
                    self.encoder.load_state_dict(torch.load(encoder_filename))
                    self.encoder.to(self.device)

                    self.decoder.load_state_dict(torch.load(decoder_filename))
                    self.decoder.to(self.device)

                    test_loss, test_bleu, test_speed = self.evaluate_batch(test_data)
                    print("\nSUMMARY: - Early stopping after %d epochs without improvements" % nepoch_no_imprv)
                    print("         - Dev metric (%s): %.4f" % (self.args.metric, best_dev))
                    print("         - Load the best models from: %s, %s at epoch %d" %
                          (encoder_filename, decoder_filename, saved_epoch))
                    print("         - Test loss: %.4f; Test bleu: %.4f; Test speed: %.2f(tokens/s)" %
                          (test_loss, test_bleu[0], test_speed))
                    return

            epoch_finish = Timer.timeEst(epoch_start, epoch / max_epochs)
            print("\nINFO: - Trained time (Remained time for %d epochs): %s" % (max_epochs - epoch, epoch_finish))

        self.encoder.load_state_dict(torch.load(encoder_filename))
        self.encoder.to(self.device)

        self.decoder.load_state_dict(torch.load(decoder_filename))
        self.decoder.to(self.device)

        test_loss, test_bleu, test_speed = self.evaluate_batch(test_data)
        print("\nSUMMARY: - Completed %d epoches" % max_epochs)
        print("         - Dev metric (%s): %.4f" % (self.args.metric, best_dev))
        print("         - Load the best model from: %s, %s at epoch %d" %
              (encoder_filename, decoder_filename, saved_epoch))
        print("         - Test loss: %.4f; Test bleu: %.4f; Test speed: %.2f(tokens/s)" %
              (test_loss, test_bleu[0], test_speed))
        return

    def lr_decay(self, epoch):
        lr = self.args.lr / (1 + self.args.decay_rate * epoch)
        print("INFO: - No improvement; Learning rate is setted as: %f" % lr)
        for param_group in self.encoder_optimizer.param_groups:
            param_group['lr'] = lr
        for param_group in self.decoder_optimizer.param_groups:
            param_group['lr'] = lr

    @staticmethod
    def filter_pad(label_words, seq_len):
        # label_words = [[w1, ..., w1], ..., [EOT, ..., EOT]]
        label_words = list(zip(*label_words))
        # label_words = [[w1, ..., EOT], ..., [w1, ..., EOT]]
        # ignore EOT (i-1)
        filter_words = [words[:i-1] if EOT not in words else words[: words.index(EOT)]
                        for words, i in zip(label_words, seq_len)]
        # print("Sequence length: ", seq_len)
        # print("Before filter: ", label_words)
        # print("After filter: ", filter_words)
        return filter_words

    @staticmethod
    def class_metrics(reference, candidate):
        bleu_score = compute_bleu(reference, candidate)
        return bleu_score

    @staticmethod
    def build_data(args):
        print("Building dataset...")
        if not os.path.exists(args.model_dir):
            os.mkdir(args.model_dir)
        s_paras = [args.wl_th, args.wcutoff]
        t_paras = [args.wl_th, args.wcutoff]
        vocab = Vocab(s_paras, t_paras)
        vocab.build(files=[args.train_file, args.dev_file], dbfile=args.schema_file, limit=-1)
        args.vocab = vocab
        # Source language
        if len(args.swd_embfile) != 0:
            scale = np.sqrt(3.0 / args.swd_dim)
            args.swd_pretrained = Embeddings.get_W(args.swd_embfile, args.swd_dim, vocab.sw2i, scale)
        else:
            args.swd_pretrained = None

        # Target language
        if len(args.twd_embfile) != 0:
            scale = np.sqrt(3.0 / args.twd_dim)
            args.twd_pretrained = Embeddings.get_W(args.twd_embfile, args.twd_dim, vocab.tw2i, scale)
        else:
            args.twd_pretrained = None

        # directly integrate transfer learning if no updating new words
        SaveloadHP.save(args, os.path.join(args.model_dir, args.model_args))

        return args


if __name__ == '__main__':
    """
    python model.py --use_cuda --teacher_forcing_ratio 0.8
    """
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--schema_file', help='Schema file in Json format', type=str,
                           default="../data/nl2sql/tables.json")

    argparser.add_argument('--train_file', help='Trained file (semQL) in Json format', type=str,
                           default="../data/mysemQL/train.json")

    argparser.add_argument('--dev_file', help='Validated file (semQL) in Json format', type=str,
                           default="../data/mysemQL/dev.json")

    argparser.add_argument('--test_file', help='Tested file (semQL) in Json format', type=str,
                           default="../data/mysemQL/dev.json")

    # Language parameters
    argparser.add_argument("--wl_th", type=int, default=None, help="Word length threshold")

    argparser.add_argument("--wcutoff", type=int, default=1, help="Prune words occurring <= wcutoff")

    argparser.add_argument("--wd_dropout", type=float, default=0.5,
                           help="Dropout rate at word-level embedding")

    argparser.add_argument("--wd_padding", action='store_true', default=False,
                           help="Flag to set all padding tokens to zero during training at word level")

    argparser.add_argument("--ed_bidirect", action='store_false', default=True,
                           help="Word-level NN Bi-directional flag")

    argparser.add_argument("--ed_mode", type=str, default="lstm", help="Word-level neural network type")

    argparser.add_argument("--ed_outdim", type=int, default=600,
                           help="Source Word-level neural network dimension")

    argparser.add_argument("--ed_layers", type=int, default=2, help="Source Number of NN layers at word level")

    argparser.add_argument("--ed_dropout", type=float, default=0.5,
                           help="Dropout rate at the encoder-decoder layer")

    argparser.add_argument("--enc_att", action='store_false', default=True,
                           help="Encoder-level NN attentional mechanism flag")

    argparser.add_argument("--sch_att", action='store_false', default=True,
                           help="Schema-level NN attentional mechanism flag")

    # Source language parameters
    argparser.add_argument("--ssos", action='store_true', default=False,
                           help="Start padding flag at a source sentence level")

    argparser.add_argument("--seos", action='store_false', default=True,
                           help="End padding flag at a source sentence level (True)")

    argparser.add_argument("--swd_embfile", type=str, help="Source Word embedding file", default="")

    argparser.add_argument("--swd_dim", type=int, default=300, help="Source Word embedding size")

    argparser.add_argument("--swd_reqgrad", action='store_false', default=True,
                           help="Either freezing or unfreezing pretrained embedding")

    # Target language parameters
    argparser.add_argument("--tsos", action='store_false', default=True,
                           help="Start padding flag at a target sentence level")

    argparser.add_argument("--teos", action='store_false', default=True,
                           help="End padding flag at a target sentence level (True)")

    argparser.add_argument("--t_reverse", action='store_true', default=False,
                           help="Reversing flag (reverse the sequence order of target language)")

    argparser.add_argument("--twd_embfile", type=str, help="Target Word embedding file", default="")

    argparser.add_argument("--twd_dim", type=int, default=300, help="Target Word embedding size")

    argparser.add_argument("--twd_reqgrad", action='store_false', default=True,
                           help="Either freezing or unfreezing pretrained embedding")

    # Other parameters
    argparser.add_argument("--final_dropout", type=float, default=0.5, help="Dropout rate at the last layer")

    argparser.add_argument("--patience", type=int, default=32,
                           help="Early stopping if no improvement after patience epoches")

    argparser.add_argument("--optimizer", type=str, default="ADAM", help="Optimized method (adagrad, sgd, ...)")

    argparser.add_argument("--metric", type=str, default="bleu", help="Optimized criterion (loss or bleu)")

    argparser.add_argument("--lr", type=float, default=0.001, help="Learning rate (ADAM: 0.001)")

    argparser.add_argument("--decay_rate", type=float, default=-1.0, help="Decay rate (0.05)")

    argparser.add_argument("--max_epochs", type=int, default=256, help="Maximum trained epochs")

    argparser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size")

    argparser.add_argument('--clip', default=-1, type=int, help='Clipping value (5)')

    argparser.add_argument('--model_dir', help='Model directory', default="../data/trained_model/", type=str)

    argparser.add_argument('--encoder_file', help='Trained encoder filename', default="test_encoder.m", type=str)

    argparser.add_argument('--decoder_file', help='Trained decoder filename', default="test_decoder.m", type=str)

    argparser.add_argument('--model_args', help='Trained argument filename', default="test_translator.args", type=str)

    argparser.add_argument("--use_cuda", action='store_true', default=False, help="GPUs Flag (default False)")

    argparser.add_argument('--teacher_forcing_ratio', help='teacher forcing ratio', default=0.5, type=float)

    argparser.add_argument("--transfer_learning", action='store_false', default=True, help="GPUs Flag (default True)")

    args = argparser.parse_args()

    args = Translator_model.build_data(args)

    translator = Translator_model(args)

    translator.train()


