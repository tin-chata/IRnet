# -*- coding: utf-8 -*-
"""
Created on 2019-05-30
@author: duytinvo
"""
import copy
import json
import argparse
import nltk
import pickle
from pattern.en import lemma
from nltk.stem import WordNetLemmatizer


VALUE_FILTER = ['what', 'how', 'list', 'give', 'show', 'find', 'id', 'order', 'when']
AGG = ['average', 'sum', 'max', 'min', 'minimum', 'maximum', 'between']


class JSON:
    @staticmethod
    def load(json_file):
        with open(json_file, 'r', encoding='utf8') as f:
            data = json.load(f)
        return data

    @staticmethod
    def dump(data, json_file):
        with open(json_file, 'w', encoding='utf8') as f:
            json.dump(data, f, indent=2)


class SCHEMA:
    def __init__(self, table_file):
        self.table_file = table_file
        self.table_dict = self.get_table_dict()

    def get_schema(self, dbid):
        return self.table_dict[dbid]

    def get_colnames(self, dbid):
        return self.table_dict[dbid]['schema_content']

    def get_tbnames(self, dbid):
        return self.table_dict[dbid]['table_names']

    def get_colset(self, dbid):
        return self.table_dict[dbid]['col_set']

    def get_tbidcol(self, dbid):
        return self.table_dict[dbid]['col_table']

    def get_tbcoldict(self, dbid):
        table_names_original = self.table_dict[dbid]['table_names_original']
        column_names_original = self.table_dict[dbid]['column_names_original']
        tbcolnames = {'column_names_original': column_names_original, 'table_names_original': table_names_original}
        tbcoldict = {}
        for i, tabn in enumerate(table_names_original):
            table = str(tabn.lower())
            cols = [str(col[1].lower()) for col in column_names_original if col[0] == i]
            tbcoldict[table] = cols
        return tbcoldict, tbcolnames

    def get_keys(self, dbid):
        keys = {}
        for kv in self.table_dict[dbid]['foreign_keys']:
            keys[kv[0]] = kv[1]
            keys[kv[1]] = kv[0]
        for id_k in self.table_dict[dbid]['primary_keys']:
            keys[id_k] = id_k
        return keys

    def tbinfo(self, dbid):
        return [self.table_dict[dbid]["table_names"],
                self.table_dict[dbid]["column_names"],
                self.table_dict[dbid]["column_types"]]

    def colinfo(self, index, dbid):
        column_name = self.table_dict[dbid]["column_names"][index][1]
        table_index = self.table_dict[dbid]["column_names"][index][0]
        table_name = self.table_dict[dbid]["table_names"][table_index]
        return table_name, column_name, index

    def get_table_dict(self):
        data = JSON.load(self.table_file)
        table = dict()
        for item in data:
            tmp_col = []
            for x in item['column_names']:
                if x[1] not in tmp_col:
                    tmp_col.append(x[1])
            item['col_set'] = tmp_col
            item['schema_content'] = [col[1] for col in item['column_names']]
            item['col_table'] = [col[0] for col in item['column_names']]
            table[item["db_id"]] = item
        return table

    def read_table(self):
        schemas = JSON.load(self.table_file)
        tables_data = {}
        cdb = 0
        for db in schemas:
            db_id = db['db_id']
            table_names = db['table_names_original']
            # table_names = [name.replace(' ', '_') for name in table_names]
            col_types = db['column_types']
            column_names = db['column_names_original']
            # column_names = [name.replace(' ', '_') for name in column_names]
            foreign_keys = db['foreign_keys']
            primary_keys = db['primary_keys']

            column_info = list(zip(*column_names))
            # tb_nums, col_names, col_types, key_types
            column_schema = list(zip(column_info[0], column_info[1], col_types, ["NULL"]*len(col_types)))
            for i in primary_keys:
                column_schema[i] = column_schema[i][:-1] + tuple(['primary'])
            for i, j in foreign_keys:
                column_schema[i] = column_schema[i][:-1] + \
                                   tuple(['foreign_{}.{}'.format(table_names[column_schema[j][0]], column_schema[j][1])])
            db_schema = {}
            for i, tb_name in enumerate(table_names):
                cols = list(filter(lambda k: k[0] == i, column_schema))
                db_schema[tb_name] = ['table{}'.format(i), {col[1]: ['col{}'.format(j), col[2], col[3]] for j, col in enumerate(cols)}]
            tables_data['db_{}'.format(db_id)] = ['db{}'.format(cdb), db_schema]
            cdb += 1
        return tables_data


class PREPROCESS:
    def __init__(self, table_path, kb_relatedto, kb_isa):
        self.wordnet_lemmatizer = WordNetLemmatizer()
        self.schemas = SCHEMA(table_path)

        with open(kb_relatedto, 'rb') as f:
            self.english_RelatedTo = pickle.load(f)

        with open(kb_isa, 'rb') as f:
            self.english_IsA = pickle.load(f)

    @staticmethod
    def group_digital(toks, idx):
        test = toks[idx].replace(':', '')
        test = test.replace('.', '')
        return True if test.isdigit() else False

    @staticmethod
    def group_values(toks, idx, num_toks):
        def check_isupper(tok_lists):
            for tok_one in tok_lists:
                if tok_one[0].isupper() is False:
                    return False
            return True

        for endIdx in reversed(range(idx + 1, num_toks + 1)):
            sub_toks = toks[idx: endIdx]

            if len(sub_toks) > 1 and check_isupper(sub_toks) is True:
                return endIdx, sub_toks
            if len(sub_toks) == 1:
                if sub_toks[0][0].isupper() and sub_toks[0].lower() not in VALUE_FILTER and \
                        sub_toks[0].lower().isalnum() is True:
                    return endIdx, sub_toks
        return idx, None

    @staticmethod
    def get_concept_result(toks, graph, col_set):
        for begin_id in range(0, len(toks)):
            for r_ind in reversed(range(1, len(toks) + 1 - begin_id)):
                tmp_query = "_".join(toks[begin_id:r_ind])
                if tmp_query in graph:
                    mi = graph[tmp_query]
                    for col in col_set:
                        if col in mi:
                            return col

    @staticmethod
    def group_symbol(toks, idx, num_toks):
        if toks[idx - 1] == "'":
            for i in range(0, min(3, num_toks - idx)):
                if toks[i + idx] == "'":
                    return i + idx, toks[idx:i + idx]
        return idx, None

    @staticmethod
    def num2year(tok):
        """
        catch 4-digit strings
        """
        if len(str(tok)) == 4 and str(tok).isdigit() and 15 < int(str(tok)[:2]) < 22:
            return True
        return False

    @staticmethod
    def partial_header(toks, idx, header_toks):
        def check_in(list_one, list_two):
            if len(set(list_one) & set(list_two)) == len(list_one) and (len(list_two) <= 3):
                return True

        for endIdx in reversed(range(idx + 1, len(toks))):
            sub_toks = toks[idx: min(endIdx, len(toks))]
            if len(sub_toks) > 1:
                flag_count = 0
                tmp_heads = None
                for heads in header_toks:
                    if check_in(sub_toks, heads):
                        flag_count += 1
                        tmp_heads = heads
                if flag_count == 1:
                    return endIdx, tmp_heads
        return idx, None

    @staticmethod
    def group_header(toks, idx, num_toks, header_toks):
        for endIdx in reversed(range(idx + 1, num_toks + 1)):
            sub_toks = toks[idx: endIdx]
            sub_toks = " ".join(sub_toks)
            if sub_toks in header_toks:
                return endIdx, sub_toks
        return idx, None

    @staticmethod
    def fully_part_header(toks, idx, num_toks, header_toks):
        """
        Check if n_grams (>2) in toks is matched with header_toks or not
        """
        for endIdx in reversed(range(idx + 1, num_toks + 1)):
            sub_toks = toks[idx: endIdx]
            if len(sub_toks) > 1:
                sub_toks = " ".join(sub_toks)
                if sub_toks in header_toks:
                    return endIdx, sub_toks
        return idx, None

    @staticmethod
    def symbol_filter(questions):
        """
        new question with replaced quote tokens
        """
        question_tmp_q = []
        for q_id, q_val in enumerate(questions):
            if len(q_val) > 2 and q_val[0] in ["'", '"', '`', '鈥�', '鈥�'] and q_val[-1] in ["'", '"', '`', '鈥�']:
                question_tmp_q.append("'")
                question_tmp_q += ["".join(q_val[1:-1])]
                question_tmp_q.append("'")
            elif len(q_val) > 2 and q_val[0] in ["'", '"', '`', '鈥�']:
                question_tmp_q.append("'")
                question_tmp_q += ["".join(q_val[1:])]
            elif len(q_val) > 2 and q_val[-1] in ["'", '"', '`', '鈥�']:
                question_tmp_q += ["".join(q_val[0:-1])]
                question_tmp_q.append("'")
            elif q_val in ["'", '"', '`', '鈥�', '鈥�', '``', "''"]:
                question_tmp_q += ["'"]
            else:
                question_tmp_q += [q_val]
        return question_tmp_q

    @staticmethod
    def re_lemma(string):
        """
        stemmed version of string
        """
        lema = lemma(string.lower())
        return lema if len(lema) > 0 else string.lower()

    def build_one(self, entry):
        """
        entry["db_id"]
        entry['origin_question_toks']
        entry['question_toks']
        """
        dbid = entry["db_id"]
        # copy of the origin question_toks
        if 'origin_question_toks' not in entry:
            entry['origin_question_toks'] = entry['question_toks']
        # process nl query:
        # --> replace double quotes to standard quotes
        # --> remove article `the`
        # --> lemmatize token (catss --> cat)
        entry['question_toks'] = PREPROCESS.symbol_filter(entry['question_toks'])
        origin_question_toks = PREPROCESS.symbol_filter([x for x in entry['origin_question_toks']
                                                         if x.lower() != 'the'])
        question_toks = [self.wordnet_lemmatizer.lemmatize(x.lower()) for x in entry['question_toks'] if
                         x.lower() != 'the']
        entry['question_toks'] = question_toks

        # process table names
        # --> lemmatize table names (departments --> department) using nltk
        # --> stem table names (am, is ,are --> be) using pattern.en.lemma
        table_names = []
        table_names_pattern = []
        for y in self.schemas.get_tbnames(dbid):
            x = [self.wordnet_lemmatizer.lemmatize(x.lower()) for x in y.split(' ')]
            table_names.append(" ".join(x))
            x = [PREPROCESS.re_lemma(x.lower()) for x in y.split(' ')]
            table_names_pattern.append(" ".join(x))

        # process column names
        # --> lemmatize column names (departments --> department) using nltk
        # --> stem column names (am, is ,are --> be) using pattern.en.lemma
        header_toks = []
        header_toks_list = []
        header_toks_pattern = []
        header_toks_list_pattern = []
        for y in self.schemas.get_colset(dbid):
            x = [self.wordnet_lemmatizer.lemmatize(x.lower()) for x in y.split(' ')]
            header_toks.append(" ".join(x))
            header_toks_list.append(x)

            x = [PREPROCESS.re_lemma(x.lower()) for x in y.split(' ')]
            header_toks_pattern.append(" ".join(x))
            header_toks_list_pattern.append(x)

        idx = 0
        tok_concol = []
        type_concol = []
        num_toks = len(question_toks)
        nltk_result = nltk.pos_tag(question_toks)
        while idx < num_toks:
            # fully header: if NL n_gram (>=2) spans are matched with column names
            end_idx, header = PREPROCESS.fully_part_header(question_toks, idx, num_toks, header_toks)
            if header:
                tok_concol.append(question_toks[idx: end_idx])
                type_concol.append(["col"])
                idx = end_idx
                continue

            # check for table: if NL n_gram (>=1) spans are matched with table names
            end_idx, tname = PREPROCESS.group_header(question_toks, idx, num_toks, table_names)
            if tname:
                tok_concol.append(question_toks[idx: end_idx])
                type_concol.append(["table"])
                idx = end_idx
                continue

            # check for column: if NL n_gram (>=1) spans are matched with column names
            # TODO: redundancy?
            end_idx, header = PREPROCESS.group_header(question_toks, idx, num_toks, header_toks)
            if header:
                tok_concol.append(question_toks[idx: end_idx])
                type_concol.append(["col"])
                idx = end_idx
                continue

            # check for partial column
            end_idx, tname = PREPROCESS.partial_header(question_toks, idx, header_toks_list)
            if tname:
                tok_concol.append(tname)
                type_concol.append(["col"])
                idx = end_idx
                continue

            # check for aggregation
            end_idx, agg = PREPROCESS.group_header(question_toks, idx, num_toks, AGG)
            if agg:
                tok_concol.append(question_toks[idx: end_idx])
                type_concol.append(["agg"])
                idx = end_idx
                continue

            if nltk_result[idx][1] in ['RBR', 'JJR']:
                tok_concol.append([question_toks[idx]])
                type_concol.append(['MORE'])
                idx += 1
                continue

            if nltk_result[idx][1] in ['RBS', 'JJS']:
                tok_concol.append([question_toks[idx]])
                type_concol.append(['MOST'])
                idx += 1
                continue

            # string match for Time Format: manually process 4-digit strings to years
            if PREPROCESS.num2year(question_toks[idx]):
                question_toks[idx] = 'year'
                # TODO: redundancy?
                end_idx, header = PREPROCESS.group_header(question_toks, idx, num_toks, header_toks)
                if header:
                    tok_concol.append(question_toks[idx: end_idx])
                    type_concol.append(["col"])
                    idx = end_idx
                    continue

            end_idx, symbol = PREPROCESS.group_symbol(question_toks, idx, num_toks)
            if symbol:
                tmp_toks = [x for x in question_toks[idx: end_idx]]
                assert len(tmp_toks) > 0, print(symbol, question_toks)
                pro_result = PREPROCESS.get_concept_result(tmp_toks, self.english_IsA, self.schemas.get_colset(dbid))
                if pro_result is None:
                    pro_result = PREPROCESS.get_concept_result(tmp_toks, self.english_RelatedTo,
                                                               self.schemas.get_colset(dbid))
                if pro_result is None:
                    pro_result = "NONE"
                for tmp in tmp_toks:
                    tok_concol.append([tmp])
                    type_concol.append([pro_result])
                    pro_result = "NONE"
                idx = end_idx
                continue

            end_idx, values = PREPROCESS.group_values(origin_question_toks, idx, num_toks)
            if values and (len(values) > 1 or question_toks[idx - 1] not in ['?', '.']):
                tmp_toks = [self.wordnet_lemmatizer.lemmatize(x) for x in question_toks[idx: end_idx] if
                            x.isalnum() is True]
                assert len(tmp_toks) > 0, print(question_toks[idx: end_idx], values, question_toks, idx, end_idx)
                pro_result = PREPROCESS.get_concept_result(tmp_toks, self.english_IsA, self.schemas.get_colset(dbid))
                if pro_result is None:
                    pro_result = PREPROCESS.get_concept_result(tmp_toks, self.english_RelatedTo,
                                                               self.schemas.get_colset(dbid))
                if pro_result is None:
                    pro_result = "NONE"
                for tmp in tmp_toks:
                    tok_concol.append([tmp])
                    type_concol.append([pro_result])
                    pro_result = "NONE"
                idx = end_idx
                continue

            result = PREPROCESS.group_digital(question_toks, idx)
            if result is True:
                tok_concol.append(question_toks[idx: idx + 1])
                type_concol.append(["value"])
                idx += 1
                continue
            if question_toks[idx] == ['ha']:
                question_toks[idx] = ['have']

            tok_concol.append([question_toks[idx]])
            type_concol.append(['NONE'])
            idx += 1
            continue
        entry['question_arg'] = tok_concol
        entry['question_arg_type'] = type_concol
        entry['nltk_pos'] = nltk_result
        return entry

    def build(self, data_path):
        """
        entry["db_id"]
        entry['origin_question_toks']
        entry['question_toks']
        """
        ndata = []
        data = JSON.load(data_path)
        for d in data:
            entry = dict()
            entry["db_id"] = copy.deepcopy(d["db_id"])
            # entry["query"] = copy.deepcopy(d["query"])
            # entry["query_toks"] = copy.deepcopy(d["query_toks"])
            # entry['query_toks_no_value'] = copy.deepcopy(d['query_toks_no_value'])
            entry["question"] = copy.deepcopy(d["question"])
            entry['question_toks'] = copy.deepcopy(d['question_toks'])
            entry = self.build_one(entry)
            ndata.append(entry)
        return ndata


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data_path', type=str, help='data', default="../../data/nl2sql/dev.json")
    arg_parser.add_argument('--table_path', type=str, help='table data', default="../../data/nl2sql/tables.json")
    arg_parser.add_argument('--output', type=str, help='output data', default="../../data/processed/dev.json")
    arg_parser.add_argument('--kb_relatedto', type=str, help='conceptNet data',
                            default="../../data/conceptNet/english_RelatedTo.pkl")
    arg_parser.add_argument('--kb_isa', type=str, help='conceptNet data',
                            default="../../data/conceptNet/english_IsA.pkl")
    args = arg_parser.parse_args()

    preporcess = PREPROCESS(args.table_path, args.kb_relatedto, args.kb_isa)
    data = preporcess.build(args.data_path)
    JSON.dump(data, args.output)


