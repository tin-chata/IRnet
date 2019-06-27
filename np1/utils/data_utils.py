"""
Created on 2018-11-27
@author: duytinvo
"""
import time
import gzip
import sys
import pickle
import math
import csv
import random
import torch
import itertools
import numpy as np
from collections import Counter
from sqlalchemy import create_engine
from sqlalchemy.sql import text

# ----------------------
#    PAD symbols
# ----------------------
PAD = u"<PAD>"
SOT = u"<s>"
EOT = u"</s>"
UNK = u"<UNK>"


# ----------------------------------------------------------------------------------------------------------------------
# ======================================== DATA-RELATED FUNCTIONS ======================================================
# ----------------------------------------------------------------------------------------------------------------------
class Vocab(object):
    def __init__(self, s_paras, t_paras):
        """
        s_paras = [swl_th=None, swcutoff=1, scl_th=None, sccutoff=1]
        t_paras = [twl_th=None, twcutoff=1, tcl_th=None, tccutoff=1]
        """
        self.swl, self.swcutoff, self.scl, self.sccutoff = s_paras
        self.sw2i, self.sc2i = {}, {}

        self.twl, self.twcutoff, self.tcl, self.tccutoff = t_paras
        self.tw2i, self.tc2i = {}, {}
        self.i2tw, self.i2tc = {}, {}
        self.i2sw, self.i2sc = {}, {}

    @staticmethod
    def idx2text(pad_ids, i2t, level=2):
        if level == 3:
            return [[[i2t[char] for char in chars] for chars in wds] for wds in pad_ids]
        elif level == 2:
            return [[i2t[wd] for wd in wds] for wds in pad_ids]
        else:
            return [i2t[token] for token in pad_ids]

    @staticmethod
    def update_sent(sent, wcnt, wl, ccnt, cl):
        wcnt.update(sent)
        wl = max(wl, len(sent))
        ccnt.update("".join(sent))
        cl = max(cl, max([len(wd) for wd in sent]))
        return wcnt, wl, ccnt, cl

    @staticmethod
    def update_vocab(cnt, cutoff, pads):
        lst = pads + [x for x, y in cnt.most_common() if y >= cutoff]
        vocabs = dict([(y, x) for x, y in enumerate(lst)])
        return vocabs

    @staticmethod
    def reversed_dict(cur_dict):
        inv_dict = {v: k for k, v in cur_dict.items()}
        return inv_dict

    def build(self, files, firstline=False, limit=-1):
        """
        Read a list of file names, return vocabulary
        :param files: list of file names
        :param firstline: ignore first line flag
        :param limit: read number of lines
        """
        swcnt, sccnt = Counter(), Counter()
        twcnt, tccnt = Counter(), Counter()
        swl, scl = 0, 0
        twl, tcl = 0, 0
        count = 0
        for fname in files:
            raw = Csvfile(fname, firstline=firstline, limit=limit)
            for source, target in raw:
                count += 1
                swcnt, swl, sccnt, scl = Vocab.update_sent(source, swcnt, swl, sccnt, scl)
                twcnt, twl, tccnt, tcl = Vocab.update_sent(target, twcnt, twl, tccnt, tcl)

        swvocab = Vocab.update_vocab(swcnt, self.swcutoff, [PAD, SOT, EOT, UNK])
        scvocab = Vocab.update_vocab(sccnt, self.sccutoff, [PAD, SOT, EOT, UNK])

        twvocab = Vocab.update_vocab(twcnt, self.twcutoff, [PAD, SOT, EOT, UNK])
        tcvocab = Vocab.update_vocab(tccnt, self.tccutoff, [PAD, SOT, EOT, UNK])

        self.sw2i = swvocab
        self.i2sw = Vocab.reversed_dict(swvocab)
        self.swl = swl if self.swl is None else min(swl, self.swl)

        self.sc2i = scvocab
        self.i2sc = Vocab.reversed_dict(scvocab)
        self.scl = scl if self.scl is None else min(scl, self.scl)

        self.tw2i = twvocab
        self.i2tw = Vocab.reversed_dict(twvocab)
        self.twl = twl if self.twl is None else min(twl, self.twl)

        self.tc2i = tcvocab
        self.i2tc = Vocab.reversed_dict(tcvocab)
        self.tcl = tcl if self.tcl is None else min(tcl, self.tcl)
        print("Extracting vocabulary: %d total samples" % count)
        print("\tSOURCE: ")
        print("\t\t%d total words, %d total characters" % (sum(swcnt.values()), sum(sccnt.values())))
        print("\t\t%d unique words, %d unique characters" % (len(swcnt), len(sccnt)))
        print("\t\t%d unique words appearing at least %d times" % (len(swvocab) - 4, self.swcutoff))
        print("\t\t%d unique characters appearing at least %d times" % (len(scvocab) - 4, self.sccutoff))
        print("\tTARGET: ")
        print("\t\t%d total words, %d total characters" % (sum(twcnt.values()), sum(tccnt.values())))
        print("\t\t%d unique words, %d unique characters" % (len(twcnt), len(tccnt)))
        print("\t\t%d unique words appearing at least %d times" % (len(twvocab) - 4, self.twcutoff))
        print("\t\t%d unique characters appearing at least %d times" % (len(tcvocab) - 4, self.tccutoff))

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
        x_batch, y_batch = [], []
        for (x, y) in data:
            if len(x_batch) == batch_size:
                # yield a tuple of list ([wd_ch_i], [label_i])
                yield x_batch, y_batch
                x_batch, y_batch = [], []
            # if use char, decompose x into wd_ch_i=[([char_ids],...[char_ids]),(word_ids)]
            if type(x[0]) == tuple:
                x = list(zip(*x))
            if type(y[0]) == tuple:
                y = list(zip(*y))
            x_batch += [x]
            y_batch += [y]

        if len(x_batch) != 0:
            yield x_batch, y_batch


class Csvfile(object):
    """
    Read txt file
    """

    def __init__(self, fname, source2idx=None, target2idx=None, firstline=False, limit=-1):
        self.fname = fname
        self.firstline = firstline
        self.limit = limit if limit > 0 else None
        self.source2idx = source2idx
        self.target2idx = target2idx
        self.length = None

    def __iter__(self):
        with open(self.fname, 'r') as f:
            f.seek(0)
            csvreader = csv.reader(f)
            if self.firstline:
                # Skip the header
                next(csvreader)
            for line in itertools.islice(csvreader, self.limit):
                source, target = line
                source = Csvfile.process_seq(source)
                target = Csvfile.process_seq(target)
                if self.source2idx is not None:
                    source = self.source2idx(source)
                if self.target2idx is not None:
                    target = self.target2idx(target)
                yield source, target

    def __len__(self):
        """Iterates once over the corpus to set and store length"""
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1
        return self.length

    @staticmethod
    def process_seq(seq):
        # seq = seq.lower()
        return seq.split()

    @staticmethod
    def get_csv_from_db(connection_string, query, fname):
        engine = create_engine(connection_string)
        conn = engine.connect()
        result = conn.execute(query)
        fh = open(fname, 'w')
        outcsv = csv.writer(fh, lineterminator='\n')
        outcsv.writerow(result.keys())
        outcsv.writerows(result)
        fh.close()
        engine.dispose()


class seqPAD:
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
            max_length = min(wthres, max_length) if wthres > 0 else max_length
            sequence_padded, sequence_length = seqPAD._pad_sequences(sequences, pad_tok, max_length)

        elif nlevels == 2:
            max_length_word = max([max(map(lambda x: len(x), seq)) for seq in sequences])
            max_length_word = min(cthres, max_length_word) if cthres > 0 else max_length_word
            sequence_padded, sequence_length = [], []
            for seq in sequences:
                # pad the word-level first to make the word length being the same
                sp, sl = seqPAD._pad_sequences(seq, pad_tok, max_length_word)
                sequence_padded += [sp]
                sequence_length += [sl]
            # pad the word-level to make the sequence length being the same
            max_length_sentence = max(map(lambda x: len(x), sequences))
            max_length_sentence = min(wthres, max_length_sentence) if wthres > 0 else max_length_sentence
            sequence_padded, _ = seqPAD._pad_sequences(sequence_padded, [pad_tok] * max_length_word,
                                                       max_length_sentence)
            # set sequence length to 1 by inserting padding
            sequence_length, _ = seqPAD._pad_sequences(sequence_length, 1, max_length_sentence)

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
    def sorted_tensors(word_ids, seq_lens, char_ids=None, wd_lens=None, label=False, dtype=torch.long,
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
    def load_embs(fname):
        embs = dict()
        s = 0
        V = 0
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
                    except:
                        continue
        #        assert len(embs) == V
        return embs

    @staticmethod
    def get_W(emb_file, wsize, vocabx, scale=0.25):
        """
        Get word matrix. W[i] is the vector for word indexed by i
        """
        print("Extracting pretrained embeddings:")
        word_vecs = Embeddings.load_embs(emb_file)
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
class Util:
    @staticmethod
    def substring_in_list(substring, list_of_strings):
        return any(substring in s for s in list_of_strings)


class DbUtil:
    @staticmethod
    def truncate_table(connection_string, table_name, api_id=None):
        if not api_id: # old trainig data db: truncate
            engine = create_engine(connection_string)
            #engine.execute("TRUNCATE " + table_name)

            from sqlalchemy.orm import sessionmaker
            Session = sessionmaker(bind=engine)
            session = Session()
            session.execute("TRUNCATE TABLE " + table_name)
            session.commit()
            session.close()
            engine.dispose()
        else: # training_tool: delete
            qry = "DELETE FROM {} WHERE api_id = {}".format(table_name, str(api_id))
            #qry = "DELETE FROM ? WHERE api_id = ?"
            engine = create_engine(connection_string)
            conn = engine.connect()
            #result = conn.execute(qry, (table_name, api_id))
            result = conn.execute(qry)
            engine.dispose()

    @staticmethod
    def truncate_populate_specific_table(ConnectionStringPostgres, GeneratedCorpusCSV, table_name, api_id=None):
        DbUtil.truncate_table(ConnectionStringPostgres, table_name, api_id)
        if not api_id:
            with open(GeneratedCorpusCSV, 'r') as f:
                engine = create_engine(ConnectionStringPostgres)
                conn = engine.raw_connection()
                cursor = conn.cursor()
                cmd = 'COPY ' + table_name + '(english_query, formal_representation) FROM STDIN WITH (FORMAT CSV, HEADER FALSE)'
                cursor.copy_expert(cmd, f)
                conn.commit()
                engine.dispose()
            print('Table truncated and data inserted into:', table_name)
        else:  # training_tool: INSERT rather than COPY
            with open(GeneratedCorpusCSV, 'r') as f:
                engine = create_engine(ConnectionStringPostgres)
                conn = engine.raw_connection()
                cursor = conn.cursor()
                cmd = 'COPY ' + table_name + '(api_id, model_id, english_query, representation) FROM STDIN WITH (FORMAT CSV, HEADER FALSE)'
                cursor.copy_expert(cmd, f)
                conn.commit()
                engine.dispose()
            print('Table truncated and data inserted into:', table_name)
            """ # this is too slow
            # c.executemany('INSERT INTO stocks VALUES (?,?,?,?,?)', purchases)
            with open(GeneratedCorpusCSV, 'r') as f:
                data = [line for line in csv.DictReader(f, fieldnames=['q', 'r'])]
                engine = create_engine(ConnectionStringPostgres)
                conn = engine.connect()
                qry = text('INSERT INTO {} (api_id, model_id, english_query, representation) VALUES ({},1,:q,:r)'.format(table_name, api_id))
                conn.execute(qry, data)
                engine.dispose()
            print('Table truncated and data inserted into:', table_name)
            """

    @staticmethod
    # not used, can deprecate
    def populate_permutor_training_set(ConnectionStringPostgres, GeneratedCorpusCSV, permutor_training_set):
        DbUtil.truncate_table(ConnectionStringPostgres, permutor_training_set)
        with open(GeneratedCorpusCSV, 'r') as f:
            engine = create_engine(ConnectionStringPostgres)
            conn = engine.raw_connection()
            # conn.execute("TRUNCATE " + permutor_training_set_payment)
            cursor = conn.cursor()
            cmd = 'COPY ' + permutor_training_set + '(english_query, formal_representation) FROM STDIN WITH (FORMAT CSV, HEADER FALSE)'
            cursor.copy_expert(cmd, f)
            conn.commit()
            engine.dispose()


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

    ssos = False
    seos = True
    ssow = False
    seow = True

    tsos = True
    teos = True
    tsow = False
    teow = True

    source2idx = vocab.wd2idx(vocab_words=vocab.sw2i, unk_words=True, sos=ssos, eos=seos,
                              vocab_chars=vocab.sc2i, unk_chars=True, sow=ssow, eow=seow)
    target2idx = vocab.wd2idx(vocab_words=vocab.tw2i, unk_words=True, sos=tsos, eos=teos,
                              vocab_chars=vocab.tc2i, unk_chars=True, sow=tsow, eow=teow, reverse=False)

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
                             vocab_chars=vocab.tc2i, unk_chars=True, sow=tsow, eow=teow)
    target_char_idx = tchar2idx([vocab.i2tw[60], vocab.i2tw[40]])
    chars = Vocab.idx2text(target_char_idx, vocab.i2tc, 2)

