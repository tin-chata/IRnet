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
import numpy as np
import operator
from queue import PriorityQueue
import torch.optim as optim
from np1.utils.data_utils import PAD, SOT, EOT
from np1.utils.bleu import compute_bleu
from np1.utils.data_utils import Progbar, Timer, SaveloadHP
from np1.utils.core_nns import Char_Word_Encoder, Char_Word_Decoder
from np1.utils.data_utils import Vocab, Data2tensor, Csvfile, seqPAD, Embeddings


Data2tensor.set_randseed(1234)


class BeamSearchNode(object):
    def __init__(self, t_word, t_seq_tensor, t_seq_len_tensor,
                 t_wd_tensor, t_wd_len_tensor, t_wdord_recover_tensor,
                 de_hidden, prevNode, logProb, length):
        self.t_word = t_word
        self.t_seq_tensor = t_seq_tensor
        self.t_seq_len_tensor = t_seq_len_tensor
        self.t_wd_tensor = t_wd_tensor
        self.t_wd_len_tensor = t_wd_len_tensor
        self.t_wdord_recover_tensor = t_wdord_recover_tensor
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
        if self.args.use_char:
            # Hyper-parameters at character-level source language
            schar_HPs = [len(self.args.vocab.sc2i), self.args.schar_dim, self.args.schar_pretrained,
                         self.args.char_drop_rate, self.args.char_zero_padding, self.args.schar_nn_mode,
                         self.args.schar_dim, self.args.schar_nn_out_dim, self.args.schar_nn_layers,
                         self.args.char_nn_bidirect, self.args.char_nn_attention]
            self.source2idx = self.args.vocab.wd2idx(vocab_words=self.args.vocab.sw2i, unk_words=True,
                                                     sos=self.args.ssos, eos=self.args.seos,
                                                     vocab_chars=self.args.vocab.sc2i, unk_chars=True,
                                                     sow=self.args.ssow, eow=self.args.seow)
            # Hyper-parameters at character-level target language
            tchar_HPs = [len(self.args.vocab.tc2i), self.args.tchar_dim, self.args.tchar_pretrained,
                         self.args.char_drop_rate, self.args.char_zero_padding, self.args.tchar_nn_mode,
                         self.args.tchar_dim, self.args.tchar_nn_out_dim, self.args.tchar_nn_layers,
                         self.args.char_nn_bidirect, self.args.char_nn_attention]
            self.target2idx = self.args.vocab.wd2idx(vocab_words=self.args.vocab.tw2i, unk_words=True,
                                                     sos=self.args.tsos, eos=self.args.teos,
                                                     vocab_chars=self.args.vocab.tc2i, unk_chars=True,
                                                     sow=self.args.tsow, eow=self.args.teow,
                                                     reverse=self.args.treverse)
            # a function to convert character to index
            self.tchar2idx = self.args.vocab.wd2idx(vocab_words=None, unk_words=False,
                                                    sos=False, eos=False,
                                                    vocab_chars=self.args.vocab.tc2i, unk_chars=True,
                                                    sow=self.args.tsow, eow=self.args.teow)
        else:
            # Hyper-parameters at character-level source language
            schar_HPs = None
            self.source2idx = self.args.vocab.wd2idx(vocab_words=self.args.vocab.sw2i, unk_words=True,
                                                     sos=self.args.ssos, eos=self.args.seos,
                                                     vocab_chars=None, unk_chars=True,
                                                     sow=self.args.ssow, eow=self.args.seow)
            # Hyper-parameters at character-level target language
            tchar_HPs = None
            self.target2idx = self.args.vocab.wd2idx(vocab_words=self.args.vocab.tw2i, unk_words=True,
                                                     sos=self.args.tsos, eos=self.args.teos,
                                                     vocab_chars=None, unk_chars=True,
                                                     sow=self.args.tsow, eow=self.args.teow,
                                                     reverse=self.args.treverse)
        # Hyper-parameters at word-level source language
        sword_HPs = [len(self.args.vocab.sw2i), self.args.sword_dim, self.args.sword_pretrained,
                     self.args.word_drop_rate, self.args.word_zero_padding, self.args.sword_nn_mode,
                     self.args.sword_dim, self.args.sword_nn_out_dim, self.args.sword_nn_layers,
                     self.args.word_nn_bidirect, self.args.word_nn_attention]

        self.encoder = Char_Word_Encoder(word_HPs=sword_HPs, char_HPs=schar_HPs).to(self.device)

        # Hyper-parameters at word-level target language
        tword_HPs = [len(self.args.vocab.tw2i), self.args.tword_dim, self.args.tword_pretrained,
                     self.args.word_drop_rate, self.args.word_zero_padding, self.args.tword_nn_mode,
                     self.args.tword_dim, self.args.tword_nn_out_dim, self.args.tword_nn_layers,
                     self.args.word_nn_bidirect, self.args.word_nn_attention]

        self.decoder = Char_Word_Decoder(word_HPs=tword_HPs, char_HPs=tchar_HPs, drop_rate=self.args.final_drop_rate,
                                         num_labels=self.num_labels, enc_att=self.args.encoder_attention).to(self.device)

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

    def greedy_predict(self, sentence):
        self.encoder.eval()
        self.decoder.eval()
        twords = [SOT]
        prob = 1.0
        with torch.no_grad():
            sentence = Csvfile.process_seq(sentence)
            source = self.source2idx(sentence)
            if self.args.use_char:
                schars, source = list(zip(*source))
                schar_ids, swd_lens = seqPAD.pad_sequences(schars, pad_tok=self.args.vocab.sc2i[PAD], nlevels=2)
            else:
                schar_ids, swd_lens = None, None

            sword_ids, sseq_lens = seqPAD.pad_sequences([source], pad_tok=self.args.vocab.sw2i[PAD])
            source_tensors = Data2tensor.sorted_tensors(sword_ids, sseq_lens, schar_ids, swd_lens, False,
                                                        torch.long, self.device)
            sseq_tensor, sseq_len_tensor, sseqord_tensor, sseqord_recover_tensor, \
            swd_tensor, swd_len_tensor, swdord_recover_tensor = source_tensors

            # source --> encoder --> source_hidden
            en_output, en_hidden = self.encoder(sseq_tensor, sseq_len_tensor,
                                                swd_tensor, swd_len_tensor, swdord_recover_tensor)

            de_hidden = en_hidden
            batch_size = 1
            # Extract the first target word (tsos or SOT) to feed to decoder
            t_seq_tensor = Data2tensor.idx2tensor([[self.args.vocab.tw2i[SOT]]] * batch_size, dtype=torch.long,
                                                  device=self.device)
            t_seq_len_tensor = Data2tensor.idx2tensor([1] * batch_size, dtype=torch.long, device=self.device)
            if self.args.use_char:
                t_chars = self.tchar2idx([[] for _ in range(batch_size)])
                t_wd_ids, t_wd_lens = seqPAD.pad_sequences(t_chars, pad_tok=self.args.vocab.tc2i[PAD])
                t_tensors = Data2tensor.sorted_tensors(t_wd_ids, t_wd_lens, None, None, False)
                t_wd_tensor, t_wd_len_tensor, t_wdord_tensor, t_wdord_recover_tensor, _, _, _ = t_tensors
            else:
                t_wd_tensor, t_wd_len_tensor, t_wdord_recover_tensor = None, None, None

            while True:
                de_output, de_hidden = self.decoder.get_all_hiddens(t_seq_tensor, t_seq_len_tensor, t_wd_tensor,
                                                                    t_wd_len_tensor, t_wdord_recover_tensor,
                                                                    de_hidden, en_output)
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

                if self.args.use_char:
                    t_chars = self.tchar2idx([self.args.vocab.i2tw[wid] for wid in t_seq_tensor.squeeze().tolist()])
                    t_wd_ids, t_wd_lens = seqPAD.pad_sequences(t_chars, pad_tok=self.args.vocab.tc2i[PAD])
                    t_tensors = Data2tensor.sorted_tensors(t_wd_ids, t_wd_lens, None, None, False)
                    t_wd_tensor, t_wd_len_tensor, t_wdord_tensor, t_wdord_recover_tensor, _, _, _ = t_tensors
                else:
                    t_wd_tensor, t_wd_len_tensor, t_wdord_recover_tensor = None, None, None
            return " ".join(twords), prob

    def beam_predict(self, sentence, bw=2, topk=2):
        """

        :param sentence: source sentence
        :param bw: beam width
        :param topk: number of target sentences (bw <= topk <= bw**2)
        :return: target sentences with confident scores
        """
        self.encoder.eval()
        self.decoder.eval()
        decoded_batch = []
        with torch.no_grad():
            sentence = Csvfile.process_seq(sentence)
            source = self.source2idx(sentence)
            if self.args.use_char:
                schars, source = list(zip(*source))
                schar_ids, swd_lens = seqPAD.pad_sequences(schars, pad_tok=self.args.vocab.sc2i[PAD], nlevels=2)
            else:
                schar_ids, swd_lens = None, None

            sword_ids, sseq_lens = seqPAD.pad_sequences([source], pad_tok=self.args.vocab.sw2i[PAD])
            source_tensors = Data2tensor.sorted_tensors(sword_ids, sseq_lens, schar_ids, swd_lens, False,
                                                        torch.long, self.device)
            sseq_tensor, sseq_len_tensor, sseqord_tensor, sseqord_recover_tensor, \
            swd_tensor, swd_len_tensor, swdord_recover_tensor = source_tensors

            # source --> encoder --> source_hidden
            en_output, en_hidden = self.encoder(sseq_tensor, sseq_len_tensor,
                                                swd_tensor, swd_len_tensor, swdord_recover_tensor)

            de_hidden = en_hidden
            batch_size = 1
            # Extract the first target word (tsos or SOT) to feed to decoder
            t_seq_tensor = Data2tensor.idx2tensor([[self.args.vocab.tw2i[SOT]]] * batch_size, dtype=torch.long,
                                                  device=self.device)
            # t_seq_tensor = [batch_size, 1]
            t_seq_len_tensor = Data2tensor.idx2tensor([1] * batch_size, dtype=torch.long, device=self.device)
            if self.args.use_char:
                t_chars = self.tchar2idx([[] for _ in range(batch_size)])
                t_wd_ids, t_wd_lens = seqPAD.pad_sequences(t_chars, pad_tok=self.args.vocab.tc2i[PAD])
                t_tensors = Data2tensor.sorted_tensors(t_wd_ids, t_wd_lens, None, None, False)
                t_wd_tensor, t_wd_len_tensor, t_wdord_tensor, t_wdord_recover_tensor, _, _, _ = t_tensors
            else:
                t_wd_tensor, t_wd_len_tensor, t_wdord_recover_tensor = None, None, None

            # Number of sentence to generate
            endnodes = []

            # starting node -  t_seq_tensor, t_seq_len_tensor,
            #                  t_wd_tensor, t_wd_len_tensor, t_wdord_recover_tensor,
            #                  de_hidden, previous node, logp, length
            node = BeamSearchNode(SOT, t_seq_tensor, t_seq_len_tensor,
                                  t_wd_tensor, t_wd_len_tensor, t_wdord_recover_tensor,
                                  de_hidden, None, 0, 1)
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
                t_wd_tensor = n.t_wd_tensor
                t_wd_len_tensor = n.t_wd_len_tensor
                t_wdord_recover_tensor = n.t_wdord_recover_tensor
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
                                                                    t_wd_tensor, t_wd_len_tensor,
                                                                    t_wdord_recover_tensor,
                                                                    de_hidden, en_output)
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

                    if self.args.use_char:
                        t_chars = self.tchar2idx([self.args.vocab.i2tw[wid] for wid in t_seq_tensor.squeeze().tolist()])
                        t_wd_ids, t_wd_lens = seqPAD.pad_sequences(t_chars, pad_tok=self.args.vocab.tc2i[PAD])
                        t_tensors = Data2tensor.sorted_tensors(t_wd_ids, t_wd_lens, None, None, False)
                        t_wd_tensor, t_wd_len_tensor, t_wdord_tensor, t_wdord_recover_tensor, _, _, _ = t_tensors
                    else:
                        t_wd_tensor, t_wd_len_tensor, t_wdord_recover_tensor = None, None, None

                    node = BeamSearchNode(t_word[0], t_seq_tensor, t_seq_len_tensor,
                                          t_wd_tensor, t_wd_len_tensor, t_wdord_recover_tensor,
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
            for i, (source, target) in enumerate(self.args.vocab.minibatches(eva_data, batch_size=self.args.batch_size)):
                if self.args.use_char:
                    schars, source = list(zip(*source))
                    schar_ids, swd_lens = seqPAD.pad_sequences(schars, pad_tok=self.args.vocab.sc2i[PAD], nlevels=2)

                    tchars, target = list(zip(*target))
                    tchar_ids, twd_lens = seqPAD.pad_sequences(tchars, pad_tok=self.args.vocab.tc2i[PAD], nlevels=2)
                else:
                    schar_ids, swd_lens = None, None
                    tchar_ids, twd_lens = None, None

                sword_ids, sseq_lens = seqPAD.pad_sequences(source, pad_tok=self.args.vocab.sw2i[PAD])
                source_tensors = Data2tensor.sorted_tensors(sword_ids, sseq_lens, schar_ids, swd_lens, False,
                                                            torch.long, self.device)
                sseq_tensor, sseq_len_tensor, sseqord_tensor, sseqord_recover_tensor, \
                swd_tensor, swd_len_tensor, swdord_recover_tensor = source_tensors

                # source --> encoder --> source_hidden
                en_output, en_hidden = self.encoder(sseq_tensor, sseq_len_tensor,
                                                    swd_tensor, swd_len_tensor, swdord_recover_tensor)

                tword_ids, tseq_lens = seqPAD.pad_sequences(target, pad_tok=self.args.vocab.tw2i[PAD])
                target_tensors = Data2tensor.sorted_tensors(tword_ids, tseq_lens, tchar_ids, twd_lens, True,
                                                            torch.long, self.device)
                ttag_tensor, tseq_tensor, tseq_len_tensor, tseqord_tensor, tseqord_recover_tensor, \
                twd_tensor, twd_len_tensor, twdord_recover_tensor = target_tensors

                # Re-ordering and sort encoding outputs following target
                # en_output_re = en_output[sseqord_recover_tensor]
                # en_output_re_sort = en_output_re[tseqord_tensor]
                if isinstance(en_hidden, tuple):
                    en_hidden_re = tuple(hidden[:, sseqord_recover_tensor, :] for hidden in en_hidden)
                    en_hidden_re_sort = tuple(hidden[:, tseqord_tensor, :] for hidden in en_hidden_re)
                else:
                    en_hidden_re = en_hidden[:, sseqord_recover_tensor, :]
                    en_hidden_re_sort = en_hidden_re[:, tseqord_tensor, :]
                de_hidden = en_hidden_re_sort

                en_output_re = en_output[sseqord_recover_tensor, :, :]
                en_output_sort = en_output_re[tseqord_tensor, :, :]
                enc_out = en_output_sort

                batch_size, target_length = tseq_tensor.size()
                # Extract the first target word (tsos or SOT) to feed to decoder
                t_seq_tensor = Data2tensor.idx2tensor([[self.args.vocab.tw2i[SOT]]] * batch_size, dtype=torch.long,
                                                      device=self.device)
                # t_seq_tensor = [batch_size, 1]
                t_seq_len_tensor = Data2tensor.idx2tensor([1] * batch_size, dtype=torch.long, device=self.device)
                # t_seq_len_tensor = [batch_size]
                if self.args.use_char:
                    t_chars = self.tchar2idx([[] for _ in range(batch_size)])
                    t_wd_ids, t_wd_lens = seqPAD.pad_sequences(t_chars, pad_tok=self.args.vocab.tc2i[PAD])
                    t_tensors = Data2tensor.sorted_tensors(t_wd_ids, t_wd_lens, None, None, False)
                    t_wd_tensor, t_wd_len_tensor, t_wdord_tensor, t_wdord_recover_tensor, _, _, _ = t_tensors
                    # t_wd_tensor = [batch_size*1, word_length]
                    # t_wd_len_tensor = [batch_size*1]
                else:
                    t_wd_tensor, t_wd_len_tensor, t_wdord_recover_tensor = None, None, None

                count_tokens = 0
                batch_loss = 0
                # Ignore SOT
                label_words = []
                predict_words = []
                for j in range(target_length):
                    de_output, de_hidden = self.decoder.get_all_hiddens(t_seq_tensor, t_seq_len_tensor, t_wd_tensor,
                                                                        t_wd_len_tensor, t_wdord_recover_tensor,
                                                                        de_hidden, enc_out)
                    # de_output = [batch_size, 1, hidden_dim]
                    de_score = self.decoder.scoring(de_output)
                    # de_score = [batch_size, 1, num_labels]
                    label_prob, label_pred = self.decoder.inference(de_score, 1)
                    # label_prob = label_pred = [batch_size, 1, 1]
                    t_seq_tensor = label_pred.squeeze(-1).detach()  # detach from history as input
                    # t_seq_tensor = [batch_size, 1]
                    if self.args.use_char:
                        t_chars = self.tchar2idx([self.args.vocab.i2tw[wid] for wid in t_seq_tensor.squeeze().tolist()])
                        t_wd_ids, t_wd_lens = seqPAD.pad_sequences(t_chars, pad_tok=self.args.vocab.tc2i[PAD])
                        t_tensors = Data2tensor.sorted_tensors(t_wd_ids, t_wd_lens, None, None, False)
                        t_wd_tensor, t_wd_len_tensor, t_wdord_tensor, t_wdord_recover_tensor, _, _, _ = t_tensors
                        # t_wd_tensor = [batch_size*1, word_length]
                        # t_wd_len_tensor = [batch_size*1]
                    else:
                        t_wd_tensor, t_wd_len_tensor, t_wdord_recover_tensor = None, None, None
                    labels = ttag_tensor[:, j]
                    label_mask = labels > 0
                    # labels = label_mask = [batch_size]
                    label_words += [Vocab.idx2text(labels.squeeze().tolist(), self.args.vocab.i2tw, 1)]
                    predict_words += [Vocab.idx2text(t_seq_tensor.squeeze().tolist(), self.args.vocab.i2tw, 1)]
                    count_tokens += label_mask.sum().item()
                    batch_loss += self.decoder.NLL_loss(de_score[label_mask], labels[label_mask])

                batch_loss = batch_loss / count_tokens
                dev_loss.append(batch_loss.item())
                total_tokens += count_tokens
                # label_words = [[w1, ..., w1], ..., [EOT, ..., EOT]]
                # label_words = list(zip(*label_words))
                # label_words = [[w1, ..., EOT], ..., [w1, ..., EOT]]
                label_words = Translator_model.filter_pad(label_words, (ttag_tensor > 0).sum(dim=1).tolist())

                # predict_words = [[w1, ..., w1], ..., [EOT, ..., EOT]]
                # predict_words = list(zip(*predict_words)s)
                # predict_words = [[w1, ..., EOT], ..., [w1, ..., EOT]]
                predict_words = Translator_model.filter_pad(predict_words, (ttag_tensor > 0).sum(dim=1).tolist())

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
        num_train = len(train_data)
        total_batch = num_train // batch_size + 1
        prog = Progbar(target=total_batch)
        # set model in train model
        self.encoder.train()
        self.decoder.train()
        train_loss = []
        for i, (source, target) in enumerate(self.args.vocab.minibatches(train_data, batch_size=batch_size)):
            if self.args.use_char:
                schars, source = list(zip(*source))
                schar_ids, swd_lens = seqPAD.pad_sequences(schars, pad_tok=self.args.vocab.sc2i[PAD], nlevels=2)

                tchars, target = list(zip(*target))
                tchar_ids, twd_lens = seqPAD.pad_sequences(tchars, pad_tok=self.args.vocab.tc2i[PAD], nlevels=2)
            else:
                schar_ids, swd_lens = None, None
                tchar_ids, twd_lens = None, None

            sword_ids, sseq_lens = seqPAD.pad_sequences(source, pad_tok=self.args.vocab.sw2i[PAD])
            source_tensors = Data2tensor.sorted_tensors(sword_ids, sseq_lens, schar_ids, swd_lens, False,
                                                        torch.long, self.device)
            sseq_tensor, sseq_len_tensor, sseqord_tensor, sseqord_recover_tensor, \
            swd_tensor, swd_len_tensor, swdord_recover_tensor = source_tensors

            self.encoder.zero_grad()
            self.decoder.zero_grad()
            # source --> encoder --> source_hidden
            en_output, en_hidden = self.encoder(sseq_tensor, sseq_len_tensor,
                                                swd_tensor, swd_len_tensor, swdord_recover_tensor)
            enc_mask = sseq_tensor > 0
            # en_output = tensor(batch_size, seq_length, rnn_dim * num_directions)
            # en_hidden = (h_n,c_n); h_n = c_n = tensor(num_layers * num_directions, batch_size, rnn_dim)

            tword_ids, tseq_lens = seqPAD.pad_sequences(target, pad_tok=self.args.vocab.tw2i[PAD])
            target_tensors = Data2tensor.sorted_tensors(tword_ids, tseq_lens, tchar_ids, twd_lens, True,
                                                        torch.long, self.device)
            ttag_tensor, tseq_tensor, tseq_len_tensor, tseqord_tensor, tseqord_recover_tensor, \
            twd_tensor, twd_len_tensor, twdord_recover_tensor = target_tensors

            # Re-ordering and sort encoding outputs following target order
            # --> en_output_re = en_output[sseqord_recover_tensor]: re-oder to the original order of the source language
            # --> en_output_re_sort = en_output_re[tseqord_tensor]: follow-up the other of the target language
            if isinstance(en_hidden, tuple):
                en_hidden_re = tuple(hidden[:, sseqord_recover_tensor, :] for hidden in en_hidden)
                en_hidden_re_sort = tuple(hidden[:, tseqord_tensor, :] for hidden in en_hidden_re)
            else:
                en_hidden_re = en_hidden[:, sseqord_recover_tensor, :]
                en_hidden_re_sort = en_hidden_re[:, tseqord_tensor, :]
            de_hidden = en_hidden_re_sort

            en_output_re = en_output[sseqord_recover_tensor, :, :]
            en_output_sort = en_output_re[tseqord_tensor, :, :]
            enc_out = en_output_sort

            batch_loss = 0
            use_teacher_forcing = True if random.random() < self.args.teacher_forcing_ratio else False

            if use_teacher_forcing:  # directly feed the ground-truth target language to the decoder
                # tseq_tensor = [batch_size, seq_len]
                # tseq_len_tensor = [batch_size, ]
                # twd_tensor = [batch_size* seq_len, wd_len]
                # twdord_recover_tensor = twd_len_tensor: [batch_size* seq_len, ]
                score_tensor = self.decoder(tseq_tensor, tseq_len_tensor,
                                            twd_tensor, twd_len_tensor, twdord_recover_tensor,
                                            de_hidden, enc_out)
                label_mask = ttag_tensor > 0
                batch_loss = self.decoder.NLL_loss(score_tensor[label_mask], ttag_tensor[label_mask])
                train_loss.append(batch_loss.item())
            else:
                batch_size, target_length = tseq_tensor.size()
                # Extract the first target word (tsos or SOT) to feed to decoder
                t_seq_tensor = Data2tensor.idx2tensor([[self.args.vocab.tw2i[SOT]]] * batch_size, dtype=torch.long,
                                                      device=self.device)
                # t_seq_tensor = [batch_size, 1]
                t_seq_len_tensor = Data2tensor.idx2tensor([1] * batch_size, dtype=torch.long, device=self.device)
                # t_seq_len_tensor = [batch_size]
                if self.args.use_char:
                    t_chars = self.tchar2idx([[] for _ in range(batch_size)])
                    t_wd_ids, t_wd_lens = seqPAD.pad_sequences(t_chars, pad_tok=self.args.vocab.tc2i[PAD])
                    t_tensors = Data2tensor.sorted_tensors(t_wd_ids, t_wd_lens, None, None, False)
                    t_wd_tensor, t_wd_len_tensor, t_wdord_tensor, t_wdord_recover_tensor, _, _, _ = t_tensors
                    # t_wd_tensor = [batch_size*1, word_length]
                    # t_wd_len_tensor = [batch_size*1]
                else:
                    t_wd_tensor, t_wd_len_tensor, t_wdord_recover_tensor = None, None, None

                count_tokens = 0

                for j in range(target_length):
                    de_output, de_hidden = self.decoder.get_all_hiddens(t_seq_tensor, t_seq_len_tensor, t_wd_tensor,
                                                                        t_wd_len_tensor, t_wdord_recover_tensor,
                                                                        de_hidden, enc_out)
                    # de_output = [batch_size, 1, hidden_dim]
                    de_score = self.decoder.scoring(de_output)
                    # de_score = [batch_size, 1, num_labels]
                    label_prob, label_pred = self.decoder.inference(de_score, 1)
                    # label_prob = label_pred = [batch_size, 1, 1]
                    t_seq_tensor = label_pred.squeeze(-1).detach()  # detach from history as input
                    # t_seq_tensor = [batch_size, 1]
                    if self.args.use_char:
                        t_chars = self.tchar2idx([self.args.vocab.i2tw[wid] for wid in label_pred.squeeze().tolist()])
                        t_wd_ids, t_wd_lens = seqPAD.pad_sequences(t_chars, pad_tok=self.args.vocab.tc2i[PAD])
                        t_tensors = Data2tensor.sorted_tensors(t_wd_ids, t_wd_lens, None, None, False)
                        t_wd_tensor, t_wd_len_tensor, t_wdord_tensor, t_wdord_recover_tensor, _, _, _ = t_tensors
                        # t_wd_tensor = [batch_size*1, word_length]
                        # t_wd_len_tensor = [batch_size*1]
                    else:
                        t_wd_tensor, t_wd_len_tensor, t_wdord_recover_tensor = None, None, None
                    labels = ttag_tensor[:, j]
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
        train_data = Csvfile(self.args.train_file, firstline=self.args.firstline,
                             source2idx=self.source2idx, target2idx=self.target2idx)
        dev_data = Csvfile(self.args.dev_file, firstline=self.args.firstline,
                           source2idx=self.source2idx, target2idx=self.target2idx)
        test_data = Csvfile(self.args.test_file, firstline=self.args.firstline,
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

    def regression_test(self, readfile, writefile, threshold=0.9, firstline=True):
        c = 0
        t = 0
        fail = 0
        low = 0
        print("\nREGRESSION TEST: ...")
        start = time.time()
        data = Csvfile(readfile, firstline=firstline)
        if len(data) > 0:
            with open(writefile, "w", newline='') as f:
                csvwriter = csv.writer(f)
                csvwriter.writerow(["english_query", "confident_score", "formal_representation", "error_type"])
                for source, target in data:
                    t += 1
                    sentence = " ".join(source)
                    decoded_batch = self.beam_predict(sentence)
                    # Remove SOT and EOT
                    pred_target = decoded_batch[0][0][4:-5]
                    pred_prob = decoded_batch[0][1]
                    if pred_target != " ".join(target):
                        csvwriter.writerow([sentence, pred_prob, pred_target, "INCORRECT_FR"])
                        print("\nINCORRECT FR (p = %.4f)" % pred_prob)
                        print(sentence, '<-->', " ".join(target), '-->', pred_target)
                        c += 1
                        fail += 1
                    else:
                        if pred_prob < threshold:
                            csvwriter.writerow([sentence, pred_prob, pred_target, "LOW_SCORE"])
                            print("\nLOW SCORE (p = %.4f)" % pred_prob)
                            print(sentence, '<-->', " ".join(target), '-->', pred_target)
                            c += 1
                            low += 1
                    # if t % 1000 == 0:
                    #     print("\nINFO: - Processing %d queries in %.4f (mins)" % (t, (time.time() - start)/60))
            end = (time.time() - start)/60
            print("\nSUMMARY: - Consumed time: %.2f (mins)" % end)
            print("         - Input file: %s" % readfile)
            print("         - Output file: %s" % writefile)
            print("         - Accuracy: %.4f (%%); Mislabelling: %.4f (%%)" % (1-c/t, c/t))
            print("            - Total failed queries: %d" % fail)
            print("            - Total low_conf queries: %d" % low)
        else:
            print("\nWARNING: The regression file is EMPTY")
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
        if args.update_data:
            Translator_model.update_data(args)

        print("Building dataset...")
        if not os.path.exists(args.model_dir):
            os.mkdir(args.model_dir)
        s_paras = [args.wl_th, args.wcutoff, args.cl_th, args.ccutoff]
        t_paras = [args.wl_th, args.wcutoff, args.cl_th, args.ccutoff]
        vocab = Vocab(s_paras, t_paras)
        vocab.build(files=[args.train_file, args.dev_file, args.test_file], firstline=args.firstline, limit=-1)
        args.vocab = vocab
        # Source language
        if len(args.sword_emb_file) != 0:
            scale = np.sqrt(3.0 / args.sword_dim)
            args.sword_pretrained = Embeddings.get_W(args.sword_emb_file, args.sword_dim, vocab.sw2i, scale)
        else:
            args.sword_pretrained = None

        if len(args.schar_emb_file) != 0:
            scale = np.sqrt(3.0 / args.schar_dim)
            args.schar_pretrained = Embeddings.get_W(args.schar_emb_file, args.schar_dim, vocab.sc2i, scale)
        else:
            args.schar_pretrained = None
        # Target language
        if len(args.tword_emb_file) != 0:
            scale = np.sqrt(3.0 / args.tword_dim)
            args.tword_pretrained = Embeddings.get_W(args.tword_emb_file, args.tword_dim, vocab.tw2i, scale)
        else:
            args.tword_pretrained = None

        if len(args.tchar_emb_file) != 0:
            scale = np.sqrt(3.0 / args.tchar_dim)
            args.tchar_pretrained = Embeddings.get_W(args.tchar_emb_file, args.tchar_dim, vocab.tc2i, scale)
        else:
            args.tchar_pretrained = None

        # directly integrate transfer learning if no updating new words
        if not os.path.exists(os.path.join(args.model_dir, args.model_args)):
            SaveloadHP.save(args, os.path.join(args.model_dir, args.model_args))
            translator = Translator_model(args)
        else:
            if args.transfer_learning:
                print("\nTRANSFER LEARNING: - IS ON")
                print("                   - Load model arguments")
                pargs = SaveloadHP.load(os.path.join(args.model_dir, args.model_args))
                # directly integrate transfer learning if no updating new words
                if (set(args.vocab.tw2i.keys()) - set(pargs.vocab.tw2i.keys()) == set()) and \
                        (set(args.vocab.tc2i.keys()) - set(pargs.vocab.tc2i.keys()) == set()):
                    print("                   - Is ABLE to use (unchanged dataset)")
                    translator = transfer_learning(pargs)
                else:
                    print("                   - Is UNABLE to use (changed dataset)")
                    SaveloadHP.save(args, os.path.join(args.model_dir, args.model_args))
                    translator = Translator_model(args)
            else:
                SaveloadHP.save(args, os.path.join(args.model_dir, args.model_args))
                translator = Translator_model(args)
        return translator

    @staticmethod
    def update_data(args):
        print("Update dataset from database...")

        Csvfile.get_csv_from_db(args.connection_string, args.training_set_query, args.train_file)
        Csvfile.get_csv_from_db(args.connection_string, args.dev_set_query, args.dev_file)
        Csvfile.get_csv_from_db(args.connection_string, args.test_set_query, args.test_file)
        Csvfile.get_csv_from_db(args.connection_string, args.gold_training_set_query, args.gold_train_file)
        return


def transfer_learning(margs):
    translator = Translator_model(margs)
    # print("\nTRANSFER LEARNING: - Load model arguments")
    encoder_filename = os.path.join(margs.model_dir, margs.encoder_file)
    print("                   - Load pre-trained encoder from file: %s" % encoder_filename)
    translator.encoder.load_state_dict(torch.load(encoder_filename))
    translator.encoder.to(translator.device)

    decoder_filename = os.path.join(margs.model_dir, margs.decoder_file)
    print("                   - Load pre-trained decoder from file: %s" % decoder_filename)
    translator.decoder.load_state_dict(torch.load(decoder_filename))
    translator.decoder.to(translator.device)
    return translator


def main(argv):
    """
    python model.py --use_cuda --teacher_forcing_ratio 0.8
    """
    argparser = argparse.ArgumentParser(argv)

    # this file is used for regression test
    argparser.add_argument('--gold_train_file', help='Trained file', type=str,
                           default="./data/nl2fr/gold_train_set.csv")

    argparser.add_argument('--train_file', help='Trained file', type=str,
                           default="./data/nl2fr/train_set.csv")

    argparser.add_argument('--dev_file', help='Validated file', type=str,
                           default="./data/nl2fr/dev_set.csv")

    argparser.add_argument('--test_file', help='Tested file', type=str,
                           default="./data/nl2fr/test_set.csv")

    argparser.add_argument("--use_stripe", action='store_true', default=False, help="stripe Flag (default False)")

    argparser.add_argument("--firstline", action='store_false', default=True, help="A header flag")

    # Language parameters
    argparser.add_argument("--wl_th", type=int, default=None, help="Word threshold")

    argparser.add_argument("--wcutoff", type=int, default=1, help="Prune words occurring <= wcutoff")

    argparser.add_argument("--cl_th", type=int, default=None, help="Char threshold")

    argparser.add_argument("--ccutoff", type=int, default=1, help="Prune characters occurring <= ccutoff")

    argparser.add_argument("--use_char", action='store_true', default=False,
                           help="Character-level NN Flag (default False)")

    argparser.add_argument("--char_drop_rate", type=float, default=0.2,
                           help="Dropout rate at character-level embedding")

    argparser.add_argument("--char_zero_padding", action='store_true', default=False,
                           help="Flag to set all padding tokens to zero during training at character level")

    argparser.add_argument("--char_nn_bidirect", action='store_true', default=False,
                           help="Character-level NN Bi-directional flag")

    argparser.add_argument("--char_nn_attention", action='store_true', default=False,
                           help="Character-level NN attentional mechanism flag")

    argparser.add_argument("--word_drop_rate", type=float, default=0.2,
                           help="Dropout rate at word-level embedding")

    argparser.add_argument("--word_zero_padding", action='store_true', default=False,
                           help="Flag to set all padding tokens to zero during training at word level")

    argparser.add_argument("--word_nn_bidirect", action='store_true', default=False,
                           help="Word-level NN Bi-directional flag")

    argparser.add_argument("--word_nn_attention", action='store_true', default=False,
                           help="Word-level NN attentional mechanism flag")

    # Source language parameters
    argparser.add_argument("--ssos", action='store_true', default=False,
                           help="Start padding flag at a source sentence level")

    argparser.add_argument("--seos", action='store_false', default=True,
                           help="End padding flag at a source sentence level (True)")

    argparser.add_argument("--ssow", action='store_true', default=False,
                           help="Start padding flag at a source word level")

    argparser.add_argument("--seow", action='store_true', default=False,
                           help="End padding flag at a source word level")

    argparser.add_argument("--schar_emb_file", type=str, help="Source Character embedding file", default="")

    argparser.add_argument("--schar_dim", type=int, default=64, help="Source Char embedding size")

    argparser.add_argument("--schar_nn_mode", type=str, default="lstm",
                           help="Source Character-level neural network type")

    argparser.add_argument("--schar_nn_out_dim", type=int, default=64,
                           help="Source Character-level neural network dimension")

    argparser.add_argument("--schar_nn_layers", type=int, default=1,
                           help="Source Number of NN layers at character level")

    argparser.add_argument("--sword_emb_file", type=str, help="Source Word embedding file", default="")

    argparser.add_argument("--sword_dim", type=int, default=128, help="Source Word embedding size")

    argparser.add_argument("--sword_nn_mode", type=str, default="lstm", help="Source Word-level neural network type")

    argparser.add_argument("--sword_nn_out_dim", type=int, default=128,
                           help="Source Word-level neural network dimension")

    argparser.add_argument("--sword_nn_layers", type=int, default=2, help="Source Number of NN layers at word level")

    argparser.add_argument("--encoder_attention", action='store_true', default=False,
                           help="Encoder-level NN attentional mechanism flag")

    # Target language parameters
    argparser.add_argument("--tsos", action='store_false', default=True,
                           help="Start padding flag at a target sentence level")

    argparser.add_argument("--teos", action='store_false', default=True,
                           help="End padding flag at a target sentence level (True)")

    argparser.add_argument("--tsow", action='store_true', default=False,
                           help="Start padding flag at a target word level")

    argparser.add_argument("--teow", action='store_true', default=False,
                           help="End padding flag at a target word level")

    argparser.add_argument("--treverse", action='store_true', default=False,
                           help="Reversing flag (reverse the sequence order of target language)")

    argparser.add_argument("--tchar_emb_file", type=str, help="Target Character embedding file", default="")

    argparser.add_argument("--tchar_dim", type=int, default=64, help="Target Char embedding size")

    argparser.add_argument("--tchar_nn_mode", type=str, default="lstm",
                           help="Target Character-level neural network type")

    argparser.add_argument("--tchar_nn_out_dim", type=int, default=64,
                           help="Target Character-level neural network dimension")

    argparser.add_argument("--tchar_nn_layers", type=int, default=1,
                           help="Target Number of NN layers at character level")

    argparser.add_argument("--tword_emb_file", type=str, help="Target Word embedding file", default="")

    argparser.add_argument("--tword_dim", type=int, default=128, help="Target Word embedding size")

    argparser.add_argument("--tword_nn_mode", type=str, default="lstm", help="Target Word-level neural network type")

    argparser.add_argument("--tword_nn_out_dim", type=int, default=128,
                           help="Target Word-level neural network dimension")

    argparser.add_argument("--tword_nn_layers", type=int, default=2, help="Target Number of NN layers at word level")

    # Other parameters
    argparser.add_argument("--final_drop_rate", type=float, default=0.2, help="Dropout rate at the last layer")

    argparser.add_argument("--patience", type=int, default=32,
                           help="Early stopping if no improvement after patience epoches")

    argparser.add_argument("--optimizer", type=str, default="ADAM", help="Optimized method (adagrad, sgd, ...)")

    argparser.add_argument("--metric", type=str, default="loss", help="Optimized criterion (loss or bleu)")

    argparser.add_argument("--lr", type=float, default=0.001, help="Learning rate (ADAM: 0.001)")

    argparser.add_argument("--decay_rate", type=float, default=-1.0, help="Decay rate (0.05)")

    argparser.add_argument("--max_epochs", type=int, default=512, help="Maximum trained epochs")

    argparser.add_argument("--batch_size", type=int, default=128, help="Mini-batch size")

    argparser.add_argument('--clip', default=-1, type=int, help='Clipping value (5)')

    argparser.add_argument('--model_dir', help='Model directory', default="./data/trained_model/", type=str)

    argparser.add_argument('--encoder_file', help='Trained encoder filename', default="test_encoder.m", type=str)

    argparser.add_argument('--decoder_file', help='Trained decoder filename', default="test_decoder.m", type=str)

    argparser.add_argument('--model_args', help='Trained argument filename', default="test_translator.args", type=str)

    argparser.add_argument("--use_cuda", action='store_true', default=False, help="GPUs Flag (default False)")

    argparser.add_argument('--teacher_forcing_ratio', help='teacher forcing ratio', default=1.0, type=float)

    argparser.add_argument("--update_data", action='store_false', default=True,
                           help="Update dataset from database (default True)")

    argparser.add_argument("--regression", action='store_true', default=False,
                           help="Regression test (default False)")

    argparser.add_argument("--transfer_learning", action='store_false', default=True, help="GPUs Flag (default True)")

    argparser.add_argument("--connection_string", help='Postgres connection string', required=True, type=str)

    argparser.add_argument("--training_set_query", help='training set query', required=True, type=str)

    argparser.add_argument("--gold_training_set_query", help='human training set query', required=True, type=str)

    argparser.add_argument("--dev_set_query", help='dev set query', required=True, type=str)

    argparser.add_argument("--test_set_query", help='test set query', required=True, type=str)

    argparser.add_argument("--baseline", action='store_true', default=False,
                           help="flag to split data to 60:20:20 (default False)")

    args = argparser.parse_args()

    translator = Translator_model.build_data(args)

    translator.train()

    if args.regression:
        translator.regression_test(args.gold_train_file, "./data/red_queries/rq_gold_train_set.csv", threshold=0.9,
                                   firstline=True)

        translator.regression_test(args.test_file, "./data/red_queries/rq_test_set.csv", threshold=0.9,
                                   firstline=True)


if __name__ == '__main__':
    """
    Train the model with default settings
    """
    ConnectionString = 'postgresql://training_data_user:FNZozDW4fRCHsfaW2Yta@35.226.48.166:5432/training_data'
    TrainingSetQuery = """
        SELECT english_query, formal_representation FROM training_set 
        UNION 
        SELECT english_query, formal_representation FROM permutor_training_set 
        UNION 
        SELECT english_query, formal_representation FROM reverse_translation_training_set
    """
    GoldTrainingSetQuery = "SELECT english_query, formal_representation FROM training_set"
    DevSetQuery = """
        SELECT english_query, formal_representation FROM training_set
        UNION
        SELECT english_query, formal_representation FROM test_set
    """
    TestSetQuery = "SELECT english_query, formal_representation FROM test_set"

    if '--connection_string' not in sys.argv:
        sys.argv.extend(['--connection_string', str(ConnectionString)])

    if '--training_set_query' not in sys.argv:
        sys.argv.extend(['--training_set_query', str(TrainingSetQuery)])

    if '--gold_training_set_query' not in sys.argv:
        sys.argv.extend(['--gold_training_set_query', str(GoldTrainingSetQuery)])

    if '--dev_set_query' not in sys.argv:
        sys.argv.extend(['--dev_set_query', str(DevSetQuery)])

    if '--test_set_query' not in sys.argv:
        sys.argv.extend(['--test_set_query', str(TestSetQuery)])

    main(sys.argv)

