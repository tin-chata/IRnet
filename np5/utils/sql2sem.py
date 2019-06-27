# -*- coding: utf-8 -*-
"""
# @Time    : 2019/5/24
# @Author  : Jiaqi&Zecheng
# @File    : sql2sem.py
# @Software: PyCharm
"""
import copy
import argparse
from np5.utils.preprocessing import JSON, SCHEMA, PREPROCESS
from np5.utils.sqltree_parser import sqlParser
from np5.semQL.semQL import Root1, Root, N, A, C, T, Sel, Sup, Filter, Order


class Parser:
    def __init__(self, table_file):
        self.copy_selec = None
        self.sel_result = []
        self.colSet = set()
        self.schemas = SCHEMA(table_file)
        self.dbid = None

    def _init_rule(self):
        self.copy_selec = None
        self.colSet = set()

    def _parse_root(self, sql):
        """
        parsing the sql by the grammar
        R ::= Select | Select Filter | Select Order | ... |
        :return: [R(), states]
        """
        use_sup, use_ord, use_fil = True, True, False

        if sql['sql']['limit'] is None:
            use_sup = False
        if not sql['sql']['orderBy']:
            use_ord = False
        elif sql['sql']['limit'] is not None:
            use_ord = False

        # check the where and having
        if sql['sql']['where'] or sql['sql']['having']:
            use_fil = True

        if use_fil and use_sup:
            return [Root(0)], ['FILTER', 'SUP', 'SEL']
        elif use_fil and use_ord:
            return [Root(1)], ['ORDER', 'FILTER', 'SEL']
        elif use_sup:
            return [Root(2)], ['SUP', 'SEL']
        elif use_fil:
            return [Root(3)], ['FILTER', 'SEL']
        elif use_ord:
            return [Root(4)], ['ORDER', 'SEL']
        else:
            return [Root(5)], ['SEL']

    def _parser_column0(self, sql, select):
        """
        Find table of column '*'
        :return: T(table_id)
        """
        if len(sql['sql']['from']['table_units']) == 1:
            # Add sql to handle sub-query in FROM clause
            if 'sql' in sql['sql']['from']['table_units'][0]:
                nest_query = dict()
                nest_query['query_toks_no_value'] = ""
                nest_query['sql'] = sql['sql']['from']['table_units'][0][1]
                nest_query['question'] = sql['question']
                nest_query['query'] = sql['query']
                nest_query['db_id'] = sql['db_id']
                print(sql['sql']['from']['table_units'][0][1], " --> ", self.parser(nest_query))
                return self.parser(nest_query)
            else:
                return [T(sql['sql']['from']['table_units'][0][1])]
        else:
            # table_set in the FROM clause
            table_list = []
            for tmp_t in sql['sql']['from']['table_units']:
                if type(tmp_t[1]) == int:
                    table_list.append(tmp_t[1])
            table_set, other_set = set(table_list), set()

            # other_set in SELECT, WHERE clauses
            for sel_p in select:
                if sel_p[1][1][1] != 0:
                    other_set.add(self.schemas.get_tbidcol(self.dbid)[sel_p[1][1][1]])

            if len(sql['sql']['where']) == 1:
                other_set.add(self.schemas.get_tbidcol(self.dbid)[sql['sql']['where'][0][2][1][1]])
            elif len(sql['sql']['where']) == 3:
                other_set.add(self.schemas.get_tbidcol(self.dbid)[sql['sql']['where'][0][2][1][1]])
                other_set.add(self.schemas.get_tbidcol(self.dbid)[sql['sql']['where'][2][2][1][1]])
            elif len(sql['sql']['where']) == 5:
                other_set.add(self.schemas.get_tbidcol(self.dbid)[sql['sql']['where'][0][2][1][1]])
                other_set.add(self.schemas.get_tbidcol(self.dbid)[sql['sql']['where'][2][2][1][1]])
                other_set.add(self.schemas.get_tbidcol(self.dbid)[sql['sql']['where'][4][2][1][1]])

            table_set = table_set - other_set
            if len(table_set) == 1:
                return [T(list(table_set)[0])]
            elif len(table_set) == 0 and sql['sql']['groupBy']:
                return [T(self.schemas.get_tbidcol(self.dbid)[sql['sql']['groupBy'][0][1]])]
            else:
                question = sql['question']
                self.sel_result.append(question)
                print(sql['db_id'], ': ', 'column * table error', " --> ", table_set)
                print("\t--> ", sql['query'], ' --> ', sql['question'])
                return [T(sql['sql']['from']['table_units'][0][1])]

    def _parse_select(self, sql):
        """
        parsing the sql by the grammar
        Select ::= A | AA | AAA | ... |
        A ::= agg column table
        :return: [Sel(), states]
        """
        result = []
        select = sql['sql']['select'][1]
        result.append(Sel(0))
        result.append(N(len(select) - 1))

        for sel in select:
            result.append(A(sel[0]))
            col_id = self.schemas.get_colset(self.dbid).index(self.schemas.get_colnames(self.dbid)[sel[1][1][1]])
            self.colSet.add(col_id)
            result.append(C(col_id))
            # now check for the situation with *
            if sel[1][1][1] == 0:
                result.extend(self._parser_column0(sql, select))
            else:
                result.append(T(self.schemas.get_tbidcol(self.dbid)[sel[1][1][1]]))
            if not self.copy_selec:
                self.copy_selec = [copy.deepcopy(result[-2]), copy.deepcopy(result[-1])]
        return result, None

    def _parse_sup(self, sql):
        """
        parsing the LIMIT sql by the grammar
        Sup ::= Most A | Least A
        A ::= agg column table
        :return: [Sup(), states]
        """
        result = []
        select = sql['sql']['select'][1]
        if sql['sql']['limit'] is None:
            return result, None
        if sql['sql']['orderBy'][0] == 'desc':
            result.append(Sup(0))
        else:
            result.append(Sup(1))

        result.append(A(sql['sql']['orderBy'][1][0][1][0]))
        col_id = self.schemas.get_colset(self.dbid).index(self.schemas.get_colnames(self.dbid)[sql['sql']['orderBy'][1][0][1][1]])
        self.colSet.add(col_id)
        result.append(C(col_id))
        if sql['sql']['orderBy'][1][0][1][1] == 0:
            result.extend(self._parser_column0(sql, select))
        else:
            result.append(T(self.schemas.get_tbidcol(self.dbid)[sql['sql']['orderBy'][1][0][1][1]]))
        return result, None

    def _parse_order(self, sql):
        """
        parsing the order sql without LIMIT by the grammar
        Order ::= asc A | desc A
        A ::= agg column table
        :return: [Order(), states]
        """
        result = []

        if 'order' not in sql['query_toks_no_value'] or 'by' not in sql['query_toks_no_value']:
            return result, None
        elif 'limit' in sql['query_toks_no_value']:
            return result, None
        else:
            if not sql['sql']['orderBy']:
                return result, None
            else:
                select = sql['sql']['select'][1]
                if sql['sql']['orderBy'][0] == 'desc':
                    result.append(Order(0))
                else:
                    result.append(Order(1))
                result.append(A(sql['sql']['orderBy'][1][0][1][0]))
                col_id = self.schemas.get_colset(self.dbid).index(self.schemas.get_colnames(self.dbid)[sql['sql']['orderBy'][1][0][1][1]])
                self.colSet.add(col_id)
                result.append(C(col_id))
                if sql['sql']['orderBy'][1][0][1][1] == 0:
                    result.extend(self._parser_column0(sql, select))
                else:
                    result.append(T(self.schemas.get_tbidcol(self.dbid)[sql['sql']['orderBy'][1][0][1][1]]))
        return result, None

    def parse_one_condition(self, sql_condit, sql):
        result = []
        # check if V(root)
        nest_query = True
        if type(sql_condit[3]) != dict:
            nest_query = False

        if sql_condit[0]:
            # not ==  True
            if sql_condit[1] == 9:
                # not like only with values
                fil = Filter(10)  # not_like
            elif sql_condit[1] == 8:
                # not in with Root
                fil = Filter(19)  # not_in
            else:
                print(sql_condit[1])
                raise NotImplementedError("not implement for the others FIL")
        else:
            # check for Filter (<,=,>,!=,between, >=,  <=, ...):
            # WHERE_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists')
            single_map = {1: 8, 2: 2, 3: 5, 4: 4, 5: 7, 6: 6, 7: 3}
            nested_map = {1: 15, 2: 11, 3: 13, 4: 12, 5: 16, 6: 17, 7: 14}
            if sql_condit[1] in [1, 2, 3, 4, 5, 6, 7]:
                if not nest_query:
                    fil = Filter(single_map[sql_condit[1]])
                else:
                    fil = Filter(nested_map[sql_condit[1]])
            elif sql_condit[1] == 9:
                fil = Filter(9)  # like
            elif sql_condit[1] == 8:
                fil = Filter(18)  # in
            else:
                print(sql_condit[1])
                raise NotImplementedError("not implement for the others FIL")

        result.append(fil)
        result.append(A(sql_condit[2][1][0]))
        col_id = self.schemas.get_colset(self.dbid).index(self.schemas.get_colnames(self.dbid)[sql_condit[2][1][1]])
        self.colSet.add(col_id)
        result.append(C(col_id))
        if sql_condit[2][1][1] == 0:
            select = sql['sql']['select'][1]
            result.extend(self._parser_column0(sql, select))
        else:
            result.append(T(self.schemas.get_tbidcol(self.dbid)[sql_condit[2][1][1]]))

        # check for the nested value
        if type(sql_condit[3]) == dict:
            nest_query = dict()
            # nest_query['names'] = names
            nest_query['query_toks_no_value'] = ""
            nest_query['sql'] = sql_condit[3]
            # nest_query['col_table'] = self.schemas.get_tbidcol(self.dbid)
            # nest_query['col_set'] = self.schemas.get_colset(self.dbid)
            # nest_query['table_names'] = sql['table_names']
            nest_query['question'] = sql['question']
            nest_query['query'] = sql['query']
            nest_query['db_id'] = sql['db_id']
            # nest_query['keys'] = sql['keys']
            result.extend(self.parser(nest_query))
            # print(self.parser(nest_query))

        return result

    def _parse_filter(self, sql):
        """
        parsing the sql by the grammar
        Filter ::= and Filter Filter | ... |
        A ::= agg column table
        :return: [Filter(), states]
        """
        result = []
        # check WHERE and HAVING --> 'Filter and Filter Filter'
        if sql['sql']['where'] and sql['sql']['having']:
            result.append(Filter(0))

        if sql['sql']['where']:
            # check and/or
            if len(sql['sql']['where']) == 1:
                result.extend(self.parse_one_condition(sql['sql']['where'][0], sql))
            elif len(sql['sql']['where']) == 3:
                if sql['sql']['where'][1] == 'or':
                    result.append(Filter(1))
                else:
                    result.append(Filter(0))
                result.extend(self.parse_one_condition(sql['sql']['where'][0], sql))
                result.extend(self.parse_one_condition(sql['sql']['where'][2], sql))
            else:
                if sql['sql']['where'][1] == 'and' and sql['sql']['where'][3] == 'and':
                    result.append(Filter(0))
                    result.extend(self.parse_one_condition(sql['sql']['where'][0], sql))
                    result.append(Filter(0))
                    result.extend(self.parse_one_condition(sql['sql']['where'][2], sql))
                    result.extend(self.parse_one_condition(sql['sql']['where'][4], sql))
                elif sql['sql']['where'][1] == 'and' and sql['sql']['where'][3] == 'or':
                    result.append(Filter(1))
                    result.append(Filter(0))
                    result.extend(self.parse_one_condition(sql['sql']['where'][0], sql))
                    result.extend(self.parse_one_condition(sql['sql']['where'][2], sql))
                    result.extend(self.parse_one_condition(sql['sql']['where'][4], sql))
                elif sql['sql']['where'][1] == 'or' and sql['sql']['where'][3] == 'and':
                    result.append(Filter(1))
                    result.append(Filter(0))
                    result.extend(self.parse_one_condition(sql['sql']['where'][2], sql))
                    result.extend(self.parse_one_condition(sql['sql']['where'][4], sql))
                    result.extend(self.parse_one_condition(sql['sql']['where'][0], sql))
                else:
                    result.append(Filter(1))
                    result.append(Filter(1))
                    result.extend(self.parse_one_condition(sql['sql']['where'][0], sql))
                    result.extend(self.parse_one_condition(sql['sql']['where'][2], sql))
                    result.extend(self.parse_one_condition(sql['sql']['where'][4], sql))

        # check having
        if sql['sql']['having']:
            result.extend(self.parse_one_condition(sql['sql']['having'][0], sql))
        return result, None

    def _parse_step(self, state, sql):

        if state == 'ROOT':
            return self._parse_root(sql)

        if state == 'SEL':
            return self._parse_select(sql)

        elif state == 'SUP':
            return self._parse_sup(sql)

        elif state == 'FILTER':
            return self._parse_filter(sql)

        elif state == 'ORDER':
            return self._parse_order(sql)
        else:
            raise NotImplementedError("Not the right state")

    def parser(self, query):
        stack = ["ROOT"]
        result = []
        while len(stack) > 0:
            state = stack.pop()
            step_result, step_state = self._parse_step(state, query)
            result.extend(step_result)
            if step_state:
                stack.extend(step_state)
        return result

    def full_parse(self, query):
        self.dbid = query["db_id"]
        sql = query['sql']
        nest_query = dict()
        # nest_query['names'] = query['names']
        nest_query['query_toks_no_value'] = ""
        # nest_query['col_table'] = query['col_table']
        # nest_query['col_set'] = query['col_set']
        # nest_query['table_names'] = query['table_names']
        nest_query['question'] = query['question']
        nest_query['query'] = query['query']
        nest_query['db_id'] = query['db_id']
        # nest_query['keys'] = query['keys']

        # split multi-query into two sub-queries
        if sql['intersect']:
            results = [Root1(0)]
            nest_query['sql'] = sql['intersect']
            results.extend(self.parser(query))
            results.extend(self.parser(nest_query))
            return results

        if sql['union']:
            results = [Root1(1)]
            nest_query['sql'] = sql['union']
            results.extend(self.parser(query))
            results.extend(self.parser(nest_query))
            return results

        if sql['except']:
            results = [Root1(2)]
            nest_query['sql'] = sql['except']
            results.extend(self.parser(query))
            results.extend(self.parser(nest_query))
            return results

        results = [Root1(3)]
        results.extend(self.parser(query))

        return results


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data_path', type=str, help='data', default="../../data/nl2sql/dev.json")
    arg_parser.add_argument('--table_path', type=str, help='table data', default="../../data/nl2sql/tables.json")
    arg_parser.add_argument('--output', type=str, help='output data', default="../../data/mysemQL/dev.json")
    arg_parser.add_argument('--kb_relatedto', type=str, help='conceptNet data',
                            default="../../data/conceptNet/english_RelatedTo.pkl")
    arg_parser.add_argument('--kb_isa', type=str, help='conceptNet data',
                            default="../../data/conceptNet/english_IsA.pkl")
    args = arg_parser.parse_args()

    preporcess = PREPROCESS(args.table_path, args.kb_relatedto, args.kb_isa)
    semQL_parser = Parser(args.table_path)

    # # loading dataSets
    # data, table = load_dataset(args.data_path, args.table_path)
    processed_data = []

    data = JSON.load(args.data_path)
    database = SCHEMA(args.table_path)
    for i, d in enumerate(data):
        entry = dict()
        entry["db_id"] = copy.deepcopy(d["db_id"])
        entry["query"] = copy.deepcopy(d["query"])
        entry['query_toks_no_value'] = copy.deepcopy(d['query_toks_no_value'])
        entry["question"] = copy.deepcopy(d["question"])
        entry['question_toks'] = copy.deepcopy(d['question_toks'])

        tbcoldict, tbcolnames = database.get_tbcoldict(entry["db_id"])
        sql_parser = sqlParser(tbcoldict, tbcolnames)
        try:
            sql_tree = sql_parser.get_sql(entry["query"])
            entry['sql'] = sql_tree

            if len(entry['sql']['select'][1]) > 5:
                # print(data[i])
                continue

            entry = preporcess.build_one(entry)
            r = semQL_parser.full_parse(entry)

            entry['rule_label'] = " ".join([str(x) for x in r])
            processed_data.append(entry)
        except:
            print(d["question"])
            print(d["query"])
            print(d["db_id"])

    print('Finished %s data and failed %s data' % (len(processed_data), len(data) - len(processed_data)))
    JSON.dump(processed_data, args.output)


