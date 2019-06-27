"""
Created on 2018-11-27
@author: duytinvo
"""
import time
import gzip
import sys
import pickle
import math
import random
import torch
import itertools
import numpy as np
from collections import Counter
import json
from np5.utils.preprocessing import SCHEMA

# ----------------------
#    PAD symbols
# ----------------------
PAD = u"<PAD>"
SOT = u"<s>"
EOT = u"</s>"
UNK = u"<UNK>"


class Jsonfile(object):
    def __init__(self, filename, dbfile, source2idx=None, target2idx=None, limit=-1):
        self.limit = limit if limit > 0 else None
        with open(filename, "r") as f:
            self.data = json.load(f)
        self.dbreader = SCHEMA(dbfile)
        self.source2idx = source2idx
        self.target2idx = target2idx
        self.length = len(self.data)

    def __iter__(self):
        for line in itertools.islice(self.data, self.limit):
            dbid = line["db_id"]
            nl = line["question_arg"]
            tp = line["question_arg_type"]
            target = line["rule_label"].split()
            if self.source2idx is not None:
                nl = self.source2idx(nl)
                tp = self.source2idx(tp)
            if self.target2idx is not None:
                target = self.target2idx(target)
            datum = (dbid, nl, tp, target)
            yield datum

    def getts(self, dbid):
        tbinfo = self.dbreader.tbinfo(dbid)
        ts = Jsonfile.process_schema(tbinfo)
        if self.source2idx is not None:
            ts = self.source2idx(ts)
        return ts

    def tsdict(self, dbids):
        ts_dict = {}
        for dbid in dbids:
            ts = self.getts(dbid)
            ts_dict[dbid] = ts
        return ts_dict

    @staticmethod
    def process_nl(nl):
        return [wd.lower() for wd in nl]

    @staticmethod
    def process_schema(ts):
            tb_names = [x.replace("_", " ").split() for x in ts[0]]
            col_names = [x[1].replace("_", " ").split() for x in ts[1]]
            tab_id = [x[0] for x in ts[1]]
            col_types = ts[2]
            cols_add = []
            for tid, col, dtype in zip(tab_id, col_names, col_types):
                tabn = ["all"] if tid == -1 else tb_names[tid]
                col_row = [tabn] + [col] + [[dtype]]
                cols_add.append(col_row)
            return cols_add

    @staticmethod
    def process_history(hs):
        nhs = []
        for word in hs:
            if isinstance(word, list) or isinstance(word, tuple):
                nhs += [[str(wd).replace("_", " ").split() for wd in word[:-1]]]
            else:
                nhs += [str(word).replace("_", " ").split()]
        return nhs


# ----------------------------------------------------------------------------------------------------------------------
# ======================================== DATA-RELATED FUNCTIONS ======================================================
# ----------------------------------------------------------------------------------------------------------------------
class Vocab(object):
    def __init__(self, s_paras, t_paras):
        """
        s_paras = [swl_th=None, swcutoff=1]
        t_paras = [twl_th=None, twcutoff=1]
        """
        # NL query
        self.swl, self.swcutoff = s_paras
        self.sw2i, self.i2sw  = {}, {}
        # Table and history
        self.twl, self.twcutoff = t_paras
        self.tw2i, self.i2tw = {}, {}

        # label
        self.l2i, self.i2l = {}, {}
        self.label_count = 0

        self.dbids = []

    @staticmethod
    def flatten(lst):
        return [item for sublist in lst for item in sublist]

    @staticmethod
    def idx2text(pad_ids, i2t, level=2):
        if level == 3:
            return [[[i2t[char] for char in chars] for chars in wds] for wds in pad_ids]
        elif level == 2:
            return [[i2t[wd] for wd in wds] for wds in pad_ids]
        else:
            return [i2t[token] for token in pad_ids]

    @staticmethod
    def update_sent(sent, wcnt, wl):
        newsent = []
        for item in sent:
            if isinstance(item, list) or isinstance(item, tuple):
                newsent.extend([tk for tk in item])
            else:
                newsent.append(str(item))
        # newsent = " ".join(newsent).split()
        wcnt.update(newsent)
        wl = max(wl, len(newsent))
        return wcnt, wl

    @staticmethod
    def update_label(sent, wcnt):
        newsent = []
        if isinstance(sent, int):
            wcnt.update([sent])
        else:
            for item in sent:
                if isinstance(item, list) or isinstance(item, tuple):
                    newsent.extend([tk for tk in item])
                else:
                    newsent.append(item)
            wcnt.update(newsent)
        return wcnt

    @staticmethod
    def update_vocab(cnt, cutoff, pads):
        lst = pads + [x for x, y in cnt.most_common() if y >= cutoff]
        vocabs = dict([(y, x) for x, y in enumerate(lst)])
        return vocabs

    @staticmethod
    def reversed_dict(cur_dict):
        inv_dict = {v: k for k, v in cur_dict.items()}
        return inv_dict

    def build(self, files, dbfile, limit=-1):
        """
        Read a list of file names, return vocabulary
        :param files: list of file names
        :param limit: read number of lines
        """
        swcnt, twcnt = Counter(), Counter()
        swl, twl = 0, 0
        count = 0
        for fname in files:
            raw = Jsonfile(fname, dbfile, limit=limit)
            for line in raw:
                count += 1
                (dbid, nl, tp, target) = line
                swcnt, swl = Vocab.update_sent(Vocab.flatten(nl), swcnt, swl)
                swcnt, swl = Vocab.update_sent(Vocab.flatten(tp), swcnt, swl)

                tb = None
                if dbid not in self.dbids:
                    self.dbids.append(dbid)
                    tb = raw.getts(dbid)
                if tb is not None:
                    swcnt, swl = Vocab.update_sent(Vocab.flatten(tb), swcnt, swl)

                twcnt, twl = Vocab.update_sent(target, twcnt, twl)

        swvocab = Vocab.update_vocab(swcnt, self.swcutoff, [PAD, SOT, EOT, UNK])

        twvocab = Vocab.update_vocab(twcnt, self.twcutoff, [PAD, SOT, EOT, UNK])

        self.sw2i = swvocab
        self.i2sw = Vocab.reversed_dict(swvocab)
        self.swl = swl if self.swl is None else min(swl, self.swl)

        self.tw2i = twvocab
        self.i2tw = Vocab.reversed_dict(twvocab)
        self.twl = twl if self.twl is None else min(twl, self.twl)

        print("Extracting vocabulary: %d total samples" % count)
        print("\tNatural Language Query: ")
        print("\t\t%d total words" % (sum(swcnt.values())))
        print("\t\t%d unique words" % (len(swcnt)))
        print("\t\t%d unique words appearing at least %d times" % (len(swvocab) - 4, self.swcutoff))
        print("\tSQL and Schema: ")
        print("\t\t%d total words" % (sum(twcnt.values())))
        print("\t\t%d unique words" % (len(twcnt)))
        print("\t\t%d unique words appearing at least %d times" % (len(twvocab) - 4, self.twcutoff))

    @staticmethod
    def tb2idx(vocab_words=None, unk_words=True, sos=False, eos=False, reverse=False):
        """
        Return a function to convert tag2idx or word/word2idx
        """
        def f(sent):
            level_flag = False
            word_ids = []
            if vocab_words is not None:
                for word in sent:
                    token_id = []
                    for token in word:
                        # if table
                        if isinstance(token, list) or isinstance(token, tuple):
                            level_flag = True
                            tk_ids = []
                            for tk in token:
                                # ignore words out of vocabulary
                                if tk in vocab_words:
                                    tk_ids += [vocab_words[tk]]
                                else:
                                    if unk_words:
                                        tk_ids += [vocab_words[UNK]]
                                    else:
                                        raise Exception(
                                            "Unknown key is not allowed. Check that your vocab (tags?) is correct")
                            token_id += [tk_ids]
                        # if either type or nl
                        else:
                            # ignore words out of vocabulary
                            if token in vocab_words:
                                tk_ids = vocab_words[token]
                            else:
                                if unk_words:
                                    tk_ids = vocab_words[UNK]
                                else:
                                    raise Exception("Unknown key is not allowed. Check that your vocab (tags?) is correct")
                            token_id += [tk_ids]

                    if level_flag:
                        if reverse:
                            token_id = token_id[::-1]
                        if sos:
                            # Add start-of-sentence
                            token_id = [[vocab_words[SOT]]] + token_id

                        if eos:
                            # add end-of-sentence
                            token_id = token_id + [[vocab_words[EOT]]]

                        if len(token_id) == 0:
                            token_id = [[[vocab_words[PAD], ]]]

                    word_ids += [token_id]
            if not level_flag:
                if reverse:
                    word_ids = word_ids[::-1]
                if sos:
                    # Add start-of-sentence

                    word_ids = [[vocab_words[SOT]]] + word_ids

                if eos:
                    # add end-of-sentence
                    word_ids = word_ids + [[vocab_words[EOT]]]

                if len(word_ids) == 0:
                    word_ids = [[[vocab_words[PAD], ]]]
            return word_ids
        return f

    @staticmethod
    def wd2idx(vocab_words=None, unk_words=True, sos=False, eos=False,
               vocab_chars=None, unk_chars=True, sow=False, eow=False,
               reverse=False):
        """
        Return a function to convert tag2idx or word/word2idx
        """

        def f(sent):
            if vocab_words is not None:
                # SOw,EOw words for  SOW
                word_ids = []
                for word in sent:
                    # ignore words out of vocabulary
                    if word in vocab_words:
                        word_ids += [vocab_words[word]]
                    else:
                        if unk_words:
                            word_ids += [vocab_words[UNK]]
                        else:
                            raise Exception("Unknown key is not allowed. Check that your vocab (tags?) is correct")
                if reverse:
                    word_ids = word_ids[::-1]
                if sos:
                    # Add start-of-sentence
                    word_ids = [vocab_words[SOT]] + word_ids

                if eos:
                    # add end-of-sentence
                    word_ids = word_ids + [vocab_words[EOT]]

                if len(word_ids) == 0:
                    word_ids = [[vocab_words[PAD], ]]

            if vocab_chars is not None:
                char_ids = []
                padding = []
                if sow:
                    # add start-of-word
                    padding = [vocab_chars[SOT]] + padding
                if eow:
                    # add end-of-word
                    padding = padding + [vocab_chars[EOT]]

                for word in sent:
                    if word not in [SOT, EOT, PAD, UNK]:
                        char_id = []
                        for char in word:
                            # ignore chars out of vocabulary
                            if char in vocab_chars:
                                char_id += [vocab_chars[char]]
                            else:
                                if unk_chars:
                                    char_id += [vocab_chars[UNK]]
                                else:
                                    raise Exception("Unknow key is not allowed. Check that your vocab (tags?) is correct")
                        if sow:
                            # add start-of-word
                            char_id = [vocab_chars[SOT]] + char_id
                        if eow:
                            # add end-of-word
                            char_id = char_id + [vocab_chars[EOT]]
                        char_ids += [char_id]
                    else:
                        char_ids += [[vocab_chars[word], ]]

                if reverse:
                    char_ids = char_ids[::-1]
                if sos:
                    # add padding start-of-sentence
                    char_ids = [padding] + char_ids
                if eos:
                    # add padding end-of-sentence
                    char_ids = char_ids + [padding]
                if len(char_ids) == 0:
                    char_ids = [padding]

            if vocab_words is not None:
                if vocab_chars is not None:
                    return list(zip(char_ids, word_ids))
                else:
                    return word_ids
            else:
                return char_ids

        return f

    @staticmethod
    def minibatches(data, batch_size):
        """
        Args:
            data: generator of (sentence, tags) tuples
            minibatch_size: (int)

        Yields:
            list of tuples
        """
        d_batch = []
        for datum in data:
            if len(d_batch) == batch_size:
                yield d_batch
                d_batch = []
            d_batch += [datum]

        if len(d_batch) != 0:
            yield d_batch


class seqPAD:
    @staticmethod
    def flatten(labels):
        """
        Binary flatten labels
        """
        flabels = []
        for i, sublist in enumerate(labels):
            nitems = []
            for item in sublist:
                if isinstance(item, int):
                    nitems.append(item)
                else:
                    nitems.extend(item)
            flabels.append(nitems)
        return flabels

    @staticmethod
    def _pad_sequences(sequences, pad_tok, max_length):
        """
        Args:
            sequences: a generator of list or tuple
            pad_tok: the word to pad with

        Returns:
            a list of list where each sublist has same length
        """
        sequence_padded, sequence_length = [], []

        for seq in sequences:
            seq = list(seq)
            seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
            sequence_padded += [seq_]
            sequence_length += [min(len(seq), max_length)]

        return sequence_padded, sequence_length

    @staticmethod
    def pad_sequences(sequences, pad_tok, nlevels=1, wthres=-1, cthres=-1):
        """
        Args:
            sequences: a generator of list or tuple
            pad_tok: the word to pad with
            nlevels: "depth" of padding, for the case where we have word ids

        Returns:
            a list of list where each sublist has same length

        """
        if nlevels == 1:
            max_length = max(map(lambda x: len(x), sequences))
            max_length = wthres if wthres > 0 else max_length
            sequence_padded, sequence_length = seqPAD._pad_sequences(sequences, pad_tok, max_length)

        elif nlevels == 2:
            max_length_word = max([max(map(lambda x: len(x), seq)) for seq in sequences])
            max_length_word = cthres if cthres > 0 else max_length_word
            word_padded, word_length = [], []
            for seq in sequences:
                # pad the word-level first to make the word length being the same
                sp, sl = seqPAD._pad_sequences(seq, pad_tok, max_length_word)
                word_padded += [sp]
                word_length += [sl]
            # pad the word-level to make the sequence length being the same
            max_length_sentence = max(map(lambda x: len(x), sequences))
            max_length_sentence = wthres if wthres > 0 else max_length_sentence
            sequence_padded, sequence_length = seqPAD._pad_sequences(word_padded, [pad_tok] * max_length_word,
                                                       max_length_sentence)
            # set sequence length to 1 by inserting padding
            word_length, _ = seqPAD._pad_sequences(word_length, 1, max_length_sentence)
        elif nlevels == 3:
            max_length_token = max([max([max(map(lambda x: len(x), wd)) for wd in seq]) for seq in sequences])
            max_length_word = max([max(map(lambda x: len(x), seq)) for seq in sequences])
            ts_pad = []
            for tb in sequences:
                ts_pad_ids, _ = seqPAD.pad_sequences(tb, pad_tok=pad_tok, wthres=max_length_word,
                                                           cthres=max_length_token, nlevels=2)
                ts_pad.append(ts_pad_ids)
            sequence_padded, sequence_length = seqPAD.pad_sequences(ts_pad,
                                                                    pad_tok=[[pad_tok]*max_length_token]*max_length_word)
        return sequence_padded, sequence_length


class Data2tensor:
    @staticmethod
    def idx2tensor(indexes, dtype=torch.long, device=torch.device("cpu")):
        vec = torch.tensor(indexes, dtype=dtype, device=device)
        return vec

    @staticmethod
    def sort_tensors(word_ids, seq_lens, char_ids=None, wd_lens=None, dtype=torch.long,
                     device=torch.device("cpu")):
        word_tensor = Data2tensor.idx2tensor(word_ids, dtype, device)
        seq_len_tensor = Data2tensor.idx2tensor(seq_lens, dtype, device)
        seq_len_tensor, seqord_tensor = seq_len_tensor.sort(0, descending=True)
        word_tensor = word_tensor[seqord_tensor]
        _, seqord_recover_tensor = seqord_tensor.sort(0, descending=False)

        if char_ids is not None:
            char_tensor = Data2tensor.idx2tensor(char_ids, dtype, device)
            wd_len_tensor = Data2tensor.idx2tensor(wd_lens, dtype, device)
            batch_size = len(word_ids)
            max_seq_len = seq_len_tensor.max()
            char_tensor = char_tensor[seqord_tensor].view(batch_size * max_seq_len.item(), -1)
            wd_len_tensor = wd_len_tensor[seqord_tensor].view(batch_size * max_seq_len.item(), )
            wd_len_tensor, wdord_tensor = wd_len_tensor.sort(0, descending=True)
            char_tensor = char_tensor[wdord_tensor]
            _, wdord_recover_tensor = wdord_tensor.sort(0, descending=False)
        else:
            char_tensor = None
            wd_len_tensor = None
            wdord_recover_tensor = None
        return word_tensor, seq_len_tensor, seqord_tensor, seqord_recover_tensor, \
               char_tensor, wd_len_tensor, wdord_recover_tensor

    @staticmethod
    def sort_labelled_tensors(word_ids, seq_lens, char_ids=None, wd_lens=None, label=False, dtype=torch.long,
                              device=torch.device("cpu")):
        word_tensor = Data2tensor.idx2tensor(word_ids, dtype, device)
        seq_len_tensor = Data2tensor.idx2tensor(seq_lens, dtype, device)
        seq_len_tensor, seqord_tensor = seq_len_tensor.sort(0, descending=True)
        word_tensor = word_tensor[seqord_tensor]
        _, seqord_recover_tensor = seqord_tensor.sort(0, descending=False)
        if label:
            input_tensor = word_tensor[:, : -1]
            seq_len_tensor = (input_tensor > 0).sum(dim=1)
            output_tensor = word_tensor[:, 1:]

        if char_ids is not None:
            char_tensor = Data2tensor.idx2tensor(char_ids, dtype, device)
            wd_len_tensor = Data2tensor.idx2tensor(wd_lens, dtype, device)
            if label:
                char_tensor = char_tensor[:, : -1, :]
                wd_len_tensor = wd_len_tensor[:, : -1]
            batch_size = len(word_ids)
            max_seq_len = seq_len_tensor.max()
            char_tensor = char_tensor[seqord_tensor].view(batch_size * max_seq_len.item(), -1)
            wd_len_tensor = wd_len_tensor[seqord_tensor].view(batch_size * max_seq_len.item(), )
            wd_len_tensor, wdord_tensor = wd_len_tensor.sort(0, descending=True)
            char_tensor = char_tensor[wdord_tensor]
            _, wdord_recover_tensor = wdord_tensor.sort(0, descending=False)
        else:
            char_tensor = None
            wd_len_tensor = None
            wdord_recover_tensor = None
        if label:
            return output_tensor, input_tensor, seq_len_tensor, seqord_tensor, seqord_recover_tensor, \
                   char_tensor, wd_len_tensor, wdord_recover_tensor
        else:
            return word_tensor, seq_len_tensor, seqord_tensor, seqord_recover_tensor, \
                   char_tensor, wd_len_tensor, wdord_recover_tensor

    @staticmethod
    def set_randseed(seed_num=12345):
        random.seed(seed_num)
        torch.manual_seed(seed_num)
        np.random.seed(seed_num)


class Embeddings:
    @staticmethod
    def load_embs(fname, use_small=False):
        embs = dict()
        s = 0
        V = 0
        c = 0
        with open(fname, 'r') as f:
            for line in f:
                p = line.strip().split()
                if len(p) == 2:
                    V = int(p[0])  # Vocabulary
                    s = int(p[1])  # embeddings size
                else:
                    try:
                        w = "".join(p[0])
                        e = [float(i) for i in p[1:]]
                        embs[w] = np.array(e, dtype="float32")
                        c += 1
                    except:
                        continue
                if c >= use_small > 0:
                    break
        #        assert len(embs) == V
        return embs

    @staticmethod
    def get_embmtx(emb_file, wsize=300, use_small=False, vocab=set(), scale=0.25):
        """
        Get word matrix. W[i] is the vector for word indexed by i
        """
        print("Extracting pretrained embeddings:")
        word_vecs = Embeddings.load_embs(emb_file, use_small)
        word_len = len(vocab.union(set(word_vecs.keys()))) + 4
        print('\t%d pre-trained word embeddings' % (len(word_vecs)))
        W = np.zeros(shape=(word_len, wsize), dtype="float32")
        w2i = {}
        for k in [PAD, SOT, EOT, UNK]:
            w2i[k] = len(w2i)
        for word, emb in word_vecs.items():
            w2i[word] = len(w2i)
            W[w2i[word]] = emb
        if len(vocab) != 0:
            for word in vocab.difference(set(word_vecs.keys())):
                w2i[word] = len(w2i)
                W[w2i[word]] = np.asarray(np.random.uniform(-scale, scale, (1, wsize)), dtype="float32")
        return W, w2i

    @staticmethod
    def get_W(emb_file, wsize, vocabx, scale=0.25, use_small=False):
        """
        Get word matrix. W[i] is the vector for word indexed by i
        """
        print("Extracting pretrained embeddings:")
        word_vecs = Embeddings.load_embs(emb_file, use_small)
        print('\t%d pre-trained word embeddings' % (len(word_vecs)))
        print('Mapping to vocabulary:')
        unk = 0
        part = 0
        W = np.zeros(shape=(len(vocabx), wsize), dtype="float32")
        for word, idx in vocabx.items():
            if idx == 0:
                continue
            if word_vecs.get(word) is not None:
                W[idx] = word_vecs.get(word)
            else:
                if word_vecs.get(word.lower()) is not None:
                    W[idx] = word_vecs.get(word.lower())
                    part += 1
                else:
                    unk += 1
                    rvector = np.asarray(np.random.uniform(-scale, scale, (1, wsize)), dtype="float32")
                    W[idx] = rvector
        print('\t%d randomly word vectors;' % unk)
        print('\t%d partially word vectors;' % part)
        print('\t%d pre-trained embeddings.' % (len(vocabx) - unk - part))
        return W

    @staticmethod
    def init_W(wsize, vocabx, scale=0.25):
        """
        Randomly initial word vectors between [-scale, scale]
        """
        W = np.zeros(shape=(len(vocabx), wsize), dtype="float32")
        for word, idx in vocabx.iteritems():
            if idx == 0:
                continue
            rvector = np.asarray(np.random.uniform(-scale, scale, (1, wsize)), dtype="float32")
            W[idx] = rvector
        return W


# --------------------------------------------------------------------------------------------------------------------
# ======================================== UTILITY FUNCTIONS =========================================================
# --------------------------------------------------------------------------------------------------------------------
class Timer:
    @staticmethod
    def asMinutes(s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    @staticmethod
    def asHours(s):
        h = math.floor(s / 3600)
        m = math.floor((s - h * 3600) / 60)
        s -= (h * 3600 + m * 60)
        return '%dh %dm %ds' % (h, m, s)

    @staticmethod
    def timeSince(since):
        now = time.time()
        s = now - since
        return '%s' % (Timer.asMinutes(s))

    @staticmethod
    def timeEst(since, percent):
        s = time.time() - since
        es = s / (percent)
        rs = es - s
        return '%s (- %s)' % (Timer.asMinutes(s), Timer.asHours(rs))


# Save and load hyper-parameters
class SaveloadHP:
    @staticmethod
    def save(args, argfile='./results/model_args.pklz'):
        """
        argfile='model_args.pklz'
        """
        print("Writing hyper-parameters into %s" % argfile)
        with gzip.open(argfile, "wb") as fout:
            pickle.dump(args, fout, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(argfile='./results/model_args.pklz'):
        print("Reading hyper-parameters from %s" % argfile)
        with gzip.open(argfile, "rb") as fin:
            args = pickle.load(fin)
        return args


class Progbar(object):
    """Progbar class copied from keras (https://github.com/fchollet/keras/)

    Displays a progress bar.
    Small edit : added strict arg to update
    # Arguments
        target: Total number of steps expected.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=30, verbose=1):
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose

    def update(self, current, values=[], exact=[], strict=[]):
        """
        Updates the progress bar.
        # Arguments
            current: Index of current step.
            values: List of tuples (name, value_for_last_step).
                The progress bar will display averages for these values.
            exact: List of tuples (name, value_for_last_step).
                The progress bar will display these values directly.
        """

        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far),
                                      current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        for k, v in exact:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = [v, 1]

        for k, v in strict:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = v

        self.seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            prev_total_width = self.total_width
            sys.stdout.write("\b" * prev_total_width)
            sys.stdout.write("\r")

            numdigits = int(np.floor(np.log10(self.target))) + 1
            barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
            bar = barstr % (current, self.target)
            prog = float(current) / self.target
            prog_width = int(self.width * prog)
            if prog_width > 0:
                bar += ('=' * (prog_width - 1))
                if current < self.target:
                    bar += '>'
                else:
                    bar += '='
            bar += ('.' * (self.width - prog_width))
            bar += ']'
            sys.stdout.write(bar)
            self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit * (self.target - current)
            info = ''
            if current < self.target:
                info += ' - ETA: %ds' % eta
            else:
                info += ' - %ds' % (now - self.start)
            for k in self.unique_values:
                if type(self.sum_values[k]) is list:
                    info += ' - %s: %.4f' % (k,
                                             self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                else:
                    info += ' - %s: %s' % (k, self.sum_values[k])

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width - self.total_width) * " ")

            sys.stdout.write(info)
            sys.stdout.flush()

            if current >= self.target:
                sys.stdout.write("\n")

        if self.verbose == 2:
            if current >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.unique_values:
                    info += ' - %s: %.4f' % (k,
                                             self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                sys.stdout.write(info + "\n")

    def add(self, n, values=[]):
        self.update(self.seen_so_far + n, values)


if __name__ == '__main__':
    import torch
    import torch.nn as nn
    from np5.utils.preprocessing import JSON
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
    tg2ids = vocab.wd2idx(vocab_words=vocab.tw2i, unk_words=False, sos=True, eos=True)

    train_data = Jsonfile(filename, db_file, source2idx=nl2ids, target2idx=tg2ids)
    ts_dict = train_data.tsdict(vocab.dbids)
    data_idx = []
    batch = 64
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

        lb_pad_ids, lb_lens = seqPAD.pad_sequences(target, pad_tok=vocab.tw2i[PAD], nlevels=1)

        nl_tensors = Data2tensor.sort_tensors(nl_pad_ids, nl_lens, dtype=torch.long, device=device)
        nl_tensor, nl_len_tensor, nl_ord_tensor, nl_recover_ord_tensor, _, _, _ = nl_tensors

        tp_tensors = Data2tensor.sort_tensors(tp_pad_ids, tp_lens, dtype=torch.long, device=device)
        tp_tensor, tp_len_tensor, tp_ord_tensor, tp_recover_ord_tensor, _, _, _ = tp_tensors

        tb_tensors = Data2tensor.sort_tensors(tb_pad_ids, tb_lens, dtype=torch.long, device=device)
        tb_tensor, tb_len_tensor, tb_ord_tensor, tb_recover_ord_tensor, _, _, _ = tb_tensors

        lb_tensors = Data2tensor.sort_labelled_tensors(lb_pad_ids, lb_lens, label=True, dtype=torch.long, device=device)
        olb_tensor, ilb_tensor, lb_len_tensor, lb_ord_tensor, lb_recover_ord_tensor, _, _, _ = lb_tensors
        break

    sembeddings = nn.Embedding(len(vocab.sw2i), 50, padding_idx=0)
    tembeddings = nn.Embedding(len(vocab.tw2i), 50, padding_idx=0)
    # nlemb: (batch, q_len, emb_size)
    nlemb = sembeddings(nl_tensor).sum(dim=-2)
    # hqemb: (batch, h_len, emb_size)
    tpemb = sembeddings(tp_tensor).sum(dim=-2)
    # tsemb: (batch, col_len, emb_size)
    tbemb = sembeddings(tb_tensor).sum(dim=-2).sum(dim=-2)

    lb_emb = tembeddings(ilb_tensor)

    # wv, w2i = Embeddings.get_embmtx("/media/data/embeddings/pretrained/glove.840B.300d.txt", 300, 1000)
    #
    # if len(vocab.dbids) == 1:
    #     # repeat for copying
    #     tsemb = tsemb.repeat(nlemb.size(0), 1, 1)
    #     ts_len_tensor = ts_len_tensor.repeat(nlemb.size(0))
    #     ts_ord_tensor = ts_ord_tensor.repeat(nlemb.size(0))
    #     ts_recover_ord_tensor = ts_recover_ord_tensor.repeat(nlemb.size(0))
    #     # expand for single view memory
    #     # tsemb = tsemb.expand(nlemb.size(0), -1, -1)
    #     # ts_len_tensor = ts_len_tensor.expand(nlemb.size(0))
    #     # ts_ord_tensor = ts_ord_tensor.expand(nlemb.size(0))
    #     # ts_recover_ord_tensor = ts_recover_ord_tensor.expand(nlemb.size(0))
