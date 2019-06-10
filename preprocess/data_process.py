# -*- coding: utf-8 -*-
"""
# @Time    : 2019/5/24
# @Author  : Jiaqi&Zecheng
# @File    : data_process.py
# @Software: PyCharm
"""
import json
import argparse
import nltk
import os
import pickle
from pattern.en import lemma
from nltk.stem import WordNetLemmatizer

VALUE_FILTER = ['what', 'how', 'list', 'give', 'show', 'find', 'id', 'order', 'when']
AGG = ['average', 'sum', 'max', 'min', 'minimum', 'maximum', 'between']

wordnet_lemmatizer = WordNetLemmatizer()


def group_digital(toks, idx):
    test = toks[idx].replace(':', '')
    test = test.replace('.', '')
    if test.isdigit():
        return True
    else:
        return False


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


def get_concept_result(toks, graph, col_set):
    for begin_id in range(0, len(toks)):
        for r_ind in reversed(range(1, len(toks) + 1 - begin_id)):
            tmp_query = "_".join(toks[begin_id:r_ind])
            if tmp_query in graph:
                mi = graph[tmp_query]
                for col in col_set:
                    if col in mi:
                        return col


def group_symbol(toks, idx, num_toks):
    if toks[idx-1] == "'":
        for i in range(0, min(3, num_toks-idx)):
            if toks[i + idx] == "'":
                return i + idx, toks[idx:i+idx]
    return idx, None


def num2year(tok):
    """
    catch 4-digit strings
    """
    if len(str(tok)) == 4 and str(tok).isdigit() and 15 < int(str(tok)[:2]) < 22:
        return True
    return False


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


def group_header(toks, idx, num_toks, header_toks):
    for endIdx in reversed(range(idx + 1, num_toks+1)):
        sub_toks = toks[idx: endIdx]
        sub_toks = " ".join(sub_toks)
        if sub_toks in header_toks:
            return endIdx, sub_toks
    return idx, None


def fully_part_header(toks, idx, num_toks, header_toks):
    """
    Check if n_grams (>2) in toks is matched with header_toks or not
    """
    for endIdx in reversed(range(idx + 1, num_toks+1)):
        sub_toks = toks[idx: endIdx]
        if len(sub_toks) > 1:
            sub_toks = " ".join(sub_toks)
            if sub_toks in header_toks:
                return endIdx, sub_toks
    return idx, None


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


def re_lemma(string):
    """
    stemmed version of string
    """
    lema = lemma(string.lower())
    if len(lema) > 0:
        return lema
    else:
        return string.lower()


def load_dataset(data_path, table_path):
    with open(table_path, 'r', encoding='utf8') as f:
        schema = json.load(f)
    with open(data_path, 'r', encoding='utf8') as f:
        data = json.load(f)

    output_tab = {}
    tables = {}
    tabel_name = set()
    for i in range(len(schema)):
        table = schema[i]
        temp = dict()
        temp['col_map'] = table['column_names']
        temp['table_names'] = table['table_names']
        tmp_col = []
        for cc in [x[1] for x in table['column_names']]:
            if cc not in tmp_col:
                tmp_col.append(cc)
        table['col_set'] = tmp_col
        db_name = table['db_id']
        tabel_name.add(db_name)
        table['schema_content'] = [col[1] for col in table['column_names']]
        table['col_table'] = [col[0] for col in table['column_names']]
        output_tab[db_name] = temp
        tables[db_name] = table

    for d in data:
        d['names'] = tables[d['db_id']]['schema_content']
        d['table_names'] = tables[d['db_id']]['table_names']
        d['col_set'] = tables[d['db_id']]['col_set']
        d['col_table'] = tables[d['db_id']]['col_table']
        keys = {}
        for kv in tables[d['db_id']]['foreign_keys']:
            keys[kv[0]] = kv[1]
            keys[kv[1]] = kv[0]
        for id_k in tables[d['db_id']]['primary_keys']:
            keys[id_k] = id_k
        d['keys'] = keys
    return data, tables


def process_datas(datas, args):
    """
    pass
    """
    with open(os.path.join(args.conceptNet, 'english_RelatedTo.pkl'), 'rb') as f:
        english_RelatedTo = pickle.load(f)

    with open(os.path.join(args.conceptNet, 'english_IsA.pkl'), 'rb') as f:
        english_IsA = pickle.load(f)

    # copy of the origin question_toks
    for d in datas:
        if 'origin_question_toks' not in d:
            d['origin_question_toks'] = d['question_toks']

    for entry in datas:
        # process nl query:
        # --> replace double quotes to standard quotes
        # --> remove article `the`
        # --> lemmatize token (catss --> cat)
        entry['question_toks'] = symbol_filter(entry['question_toks'])
        origin_question_toks = symbol_filter([x for x in entry['origin_question_toks'] if x.lower() != 'the'])
        question_toks = [wordnet_lemmatizer.lemmatize(x.lower()) for x in entry['question_toks'] if x.lower() != 'the']
        entry['question_toks'] = question_toks

        # process table names
        # --> lemmatize table names (departments --> department) using nltk
        # --> stem table names (am, is ,are --> be) using pattern.en.lemma
        table_names = []
        table_names_pattern = []
        for y in entry['table_names']:
            x = [wordnet_lemmatizer.lemmatize(x.lower()) for x in y.split(' ')]
            table_names.append(" ".join(x))
            x = [re_lemma(x.lower()) for x in y.split(' ')]
            table_names_pattern.append(" ".join(x))

        # process column names
        # --> lemmatize column names (departments --> department) using nltk
        # --> stem column names (am, is ,are --> be) using pattern.en.lemma
        header_toks = []
        header_toks_list = []
        header_toks_pattern = []
        header_toks_list_pattern = []
        for y in entry['col_set']:
            x = [wordnet_lemmatizer.lemmatize(x.lower()) for x in y.split(' ')]
            header_toks.append(" ".join(x))
            header_toks_list.append(x)

            x = [re_lemma(x.lower()) for x in y.split(' ')]
            header_toks_pattern.append(" ".join(x))
            header_toks_list_pattern.append(x)

        idx = 0
        tok_concol = []
        type_concol = []
        num_toks = len(question_toks)
        nltk_result = nltk.pos_tag(question_toks)

        while idx < num_toks:

            # fully header: if NL n_gram (>=2) spans are matched with column names
            end_idx, header = fully_part_header(question_toks, idx, num_toks, header_toks)
            if header:
                tok_concol.append(question_toks[idx: end_idx])
                type_concol.append(["col"])
                idx = end_idx
                continue

            # check for table: if NL n_gram (>=1) spans are matched with table names
            end_idx, tname = group_header(question_toks, idx, num_toks, table_names)
            if tname:
                tok_concol.append(question_toks[idx: end_idx])
                type_concol.append(["table"])
                idx = end_idx
                continue

            # check for column: if NL n_gram (>=1) spans are matched with column names
            # TODO: redundancy?
            end_idx, header = group_header(question_toks, idx, num_toks, header_toks)
            if header:
                tok_concol.append(question_toks[idx: end_idx])
                type_concol.append(["col"])
                idx = end_idx
                continue

            # check for partial column
            end_idx, tname = partial_header(question_toks, idx, header_toks_list)
            if tname:
                tok_concol.append(tname)
                type_concol.append(["col"])
                idx = end_idx
                continue

            # check for aggregation
            end_idx, agg = group_header(question_toks, idx, num_toks, AGG)
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
            if num2year(question_toks[idx]):
                question_toks[idx] = 'year'
                # TODO: redundancy?
                end_idx, header = group_header(question_toks, idx, num_toks, header_toks)
                if header:
                    tok_concol.append(question_toks[idx: end_idx])
                    type_concol.append(["col"])
                    idx = end_idx
                    continue

            end_idx, symbol = group_symbol(question_toks, idx, num_toks)
            if symbol:
                tmp_toks = [x for x in question_toks[idx: end_idx]]
                assert len(tmp_toks) > 0, print(symbol, question_toks)
                pro_result = get_concept_result(tmp_toks, english_IsA, entry['col_set'])
                if pro_result is None:
                    pro_result = get_concept_result(tmp_toks, english_RelatedTo, entry['col_set'])
                if pro_result is None:
                    pro_result = "NONE"
                for tmp in tmp_toks:
                    tok_concol.append([tmp])
                    type_concol.append([pro_result])
                    pro_result = "NONE"
                idx = end_idx
                continue

            end_idx, values = group_values(origin_question_toks, idx, num_toks)
            if values and (len(values) > 1 or question_toks[idx - 1] not in ['?', '.']):
                tmp_toks = [wordnet_lemmatizer.lemmatize(x) for x in question_toks[idx: end_idx] if x.isalnum() is True]
                assert len(tmp_toks) > 0, print(question_toks[idx: end_idx], values, question_toks, idx, end_idx)
                pro_result = get_concept_result(tmp_toks, english_IsA, entry['col_set'])
                if pro_result is None:
                    pro_result = get_concept_result(tmp_toks, english_RelatedTo, entry['col_set'])
                if pro_result is None:
                    pro_result = "NONE"
                for tmp in tmp_toks:
                    tok_concol.append([tmp])
                    type_concol.append([pro_result])
                    pro_result = "NONE"
                idx = end_idx
                continue

            result = group_digital(question_toks, idx)
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

    return datas


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data_path', type=str, help='data', default="../data/nl2sql/train.json")
    arg_parser.add_argument('--table_path', type=str, help='table data', default="../data/nl2sql/tables.json")
    arg_parser.add_argument('--output', type=str, help='output data', default="../data/processed_data.json")
    arg_parser.add_argument('--conceptNet', type=str, help='conceptNet data', default="../data/conceptNet")
    args = arg_parser.parse_args()

    # loading dataSets
    data, table = load_dataset(args.data_path, args.table_path)

    # # process datasets
    # process_result = process_datas(data, args)
    #
    # with open(args.output, 'w') as f:
    #     json.dump(data, f, indent=2)


