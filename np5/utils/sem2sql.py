# -*- coding: utf-8 -*-
"""
# @Time    : 2019/5/27
# @Author  : Jiaqi&Zecheng
# @File    : sem2sql.py
# @Software: PyCharm
"""
import copy
import traceback
import argparse
from np5.utils.preprocessing import JSON, SCHEMA, PREPROCESS
from np5.utils.postprocessing import POSTPROCESS
from np5.semQL.graph import Graph
from np5.semQL.semQL import Sup, Sel, Order, Root, Filter, A, N, C, T, Root1


class PARSER:

    @staticmethod
    def split_logical_form(lf):
        """
        Split semQL token by token
        """
        indexs = [i+1 for i, letter in enumerate(lf) if letter == ')']
        indexs.insert(0, 0)
        components = list()
        for i in range(1, len(indexs)):
            components.append(lf[indexs[i-1]:indexs[i]].strip())
        return components

    @staticmethod
    def pop_front(array):
        if len(array) == 0:
            return 'None'
        return array.pop(0)

    @staticmethod
    def is_end(components, transformed_sql, is_root_processed):
        end = False
        c = PARSER.pop_front(components)
        c_instance = eval(c)

        if isinstance(c_instance, Root) and is_root_processed:
            # intersect, union, except
            end = True
        elif isinstance(c_instance, Filter):
            if 'where' not in transformed_sql:
                end = True
            else:
                num_conjunction = 0
                for f in transformed_sql['where']:
                    if isinstance(f, str) and (f == 'and' or f == 'or'):
                        num_conjunction += 1
                current_filters = len(transformed_sql['where'])
                valid_filters = current_filters - num_conjunction
                if valid_filters >= num_conjunction + 1:
                    end = True
        elif isinstance(c_instance, Order):
            if 'order' not in transformed_sql:
                end = True
            elif len(transformed_sql['order']) == 0:
                end = False
            else:
                end = True
        elif isinstance(c_instance, Sup):
            if 'sup' not in transformed_sql:
                end = True
            elif len(transformed_sql['sup']) == 0:
                end = False
            else:
                end = True
        components.insert(0, c)
        return end

    @staticmethod
    def replace_col_with_original_col(query, col, current_table):
        # print(query, col)
        if query == '*':
            return query

        cur_table = col
        cur_col = query
        single_final_col = None
        for col_ind, col_name in enumerate(current_table['schema_content_clean']):
            if col_name == cur_col:
                assert cur_table in current_table['table_names']
                if current_table['table_names'][current_table['col_table'][col_ind]] == cur_table:
                    single_final_col = current_table['column_names_original'][col_ind][1]
                    break

        assert single_final_col
        # if query != single_final_col:
        #     print(query, single_final_col)
        return single_final_col

    @staticmethod
    def _transform(components, transformed_sql, schema):
        processed_root = False
        col_set = schema['col_set']
        table_names = schema['table_names']
        current_table = schema

        while len(components) > 0:
            # check if we already processed all components
            if PARSER.is_end(components, transformed_sql, processed_root):
                break
            c = PARSER.pop_front(components)
            c_instance = eval(c)
            if isinstance(c_instance, Root):
                processed_root = True
                transformed_sql['select'] = list()
                if c_instance.id_c == 0:
                    # 'Root --> Sel Sup Filter',
                    transformed_sql['where'] = list()
                    transformed_sql['sup'] = list()
                elif c_instance.id_c == 1:
                    # 'Root --> Sel Filter Order',
                    transformed_sql['where'] = list()
                    transformed_sql['order'] = list()
                elif c_instance.id_c == 2:
                    # 'Root --> Sel Sup',
                    transformed_sql['sup'] = list()
                elif c_instance.id_c == 3:
                    # 'Root --> Sel Filter',
                    transformed_sql['where'] = list()
                elif c_instance.id_c == 4:
                    # 'Root --> Sel Order',
                    transformed_sql['order'] = list()
            elif isinstance(c_instance, Sel):
                continue
            elif isinstance(c_instance, N):
                for i in range(c_instance.id_c + 1):
                    # aggregation
                    agg = eval(PARSER.pop_front(components))
                    column = eval(PARSER.pop_front(components))
                    _table = PARSER.pop_front(components)
                    table = eval(_table)
                    if not isinstance(table, T):
                        table = None
                        components.insert(0, _table)
                    assert isinstance(agg, A) and isinstance(column, C)
                    # print(column, table)
                    if table is not None:
                        tmp = PARSER.replace_col_with_original_col(col_set[column.id_c],
                                                                   table_names[table.id_c],
                                                                   current_table)
                    else:
                        tmp = col_set[column.id_c]
                    transformed_sql['select'].append((agg.production.split()[1], tmp,
                                                      table_names[table.id_c] if table is not None else table))

            elif isinstance(c_instance, Sup):
                # des or asc
                transformed_sql['sup'].append(c_instance.production.split()[1])
                agg = eval(PARSER.pop_front(components))
                column = eval(PARSER.pop_front(components))
                _table = PARSER.pop_front(components)
                table = eval(_table)
                if not isinstance(table, T):
                    table = None
                    components.insert(0, _table)
                assert isinstance(agg, A) and isinstance(column, C)

                transformed_sql['sup'].append(agg.production.split()[1])
                if table:
                    fix_col_id = PARSER.replace_col_with_original_col(col_set[column.id_c],
                                                                      table_names[table.id_c], current_table)
                else:
                    fix_col_id = col_set[column.id_c]
                    raise RuntimeError('not found table !!!!')
                transformed_sql['sup'].append(fix_col_id)
                transformed_sql['sup'].append(table_names[table.id_c] if table is not None else table)

            elif isinstance(c_instance, Order):
                # des or asc
                transformed_sql['order'].append(c_instance.production.split()[1])
                agg = eval(PARSER.pop_front(components))
                column = eval(PARSER.pop_front(components))
                _table = PARSER.pop_front(components)
                table = eval(_table)
                if not isinstance(table, T):
                    table = None
                    components.insert(0, _table)
                assert isinstance(agg, A) and isinstance(column, C)
                # none, max, min, count, sum, avg
                transformed_sql['order'].append(agg.production.split()[1])
                transformed_sql['order'].append(PARSER.replace_col_with_original_col(col_set[column.id_c],
                                                                                     table_names[table.id_c],
                                                                                     current_table))
                transformed_sql['order'].append(table_names[table.id_c] if table is not None else table)

            elif isinstance(c_instance, Filter):
                op = c_instance.production.split()[1]
                if op == 'and' or op == 'or':
                    transformed_sql['where'].append(op)
                else:
                    # No Supquery
                    agg = eval(PARSER.pop_front(components))
                    column = eval(PARSER.pop_front(components))
                    _table = PARSER.pop_front(components)
                    table = eval(_table)
                    if not isinstance(table, T):
                        table = None
                        components.insert(0, _table)
                    assert isinstance(agg, A) and isinstance(column, C)
                    if len(c_instance.production.split()) == 3:
                        if table:
                            fix_col_id = PARSER.replace_col_with_original_col(col_set[column.id_c],
                                                                              table_names[table.id_c], current_table)
                        else:
                            fix_col_id = col_set[column.id_c]
                            raise RuntimeError('not found table !!!!')
                        transformed_sql['where'].append((op, agg.production.split()[1], fix_col_id,
                                                         table_names[table.id_c] if table is not None else table,
                                                         None))
                    else:
                        # Subquery
                        new_dict = dict()
                        new_dict['sql'] = transformed_sql['sql']
                        fix_col_id = PARSER.replace_col_with_original_col(col_set[column.id_c],
                                                                          table_names[table.id_c], current_table)
                        transformed_sql['where'].append((op, agg.production.split()[1], fix_col_id,
                                                         table_names[table.id_c] if table is not None else table,
                                                         PARSER._transform(components, new_dict, schema)))
        return transformed_sql

    @staticmethod
    def build_graph(schema):
        relations = list()
        foreign_keys = schema['foreign_keys']
        for (fkey, pkey) in foreign_keys:
            fkey_table = schema['table_names_original'][schema['column_names'][fkey][0]]
            pkey_table = schema['table_names_original'][schema['column_names'][pkey][0]]
            relations.append((fkey_table, pkey_table))
            relations.append((pkey_table, fkey_table))
        return Graph(relations)

    @staticmethod
    def preprocess_schema(schema):
        # tmp_col = []
        # for cc in [x[1] for x in schema['column_names']]:
        #     if cc not in tmp_col:
        #         tmp_col.append(cc)
        # schema['col_set'] = tmp_col
        # # print table
        # schema['schema_content'] = [col[1] for col in schema['column_names']]
        # schema['col_table'] = [col[0] for col in schema['column_names']]
        schema['schema_content_clean'] = [x[1] for x in schema['column_names']]
        schema['schema_content'] = [x[1] for x in schema['column_names_original']]
        graph = PARSER.build_graph(schema)
        schema['graph'] = graph

    @staticmethod
    def transform(query, schema, origin=None):
        PARSER.preprocess_schema(schema)
        if origin is None:
            lf = query['model_result_replace']
        else:
            lf = origin
        # lf = query['rule_label']
        # col_set = schema['col_set']
        # table_names = schema['table_names']
        # current_table = schema
        # current_table['schema_content_clean'] = [x[1] for x in current_table['column_names']]
        # current_table['schema_content'] = [x[1] for x in current_table['column_names_original']]
        components = PARSER.split_logical_form(lf)
        transformed_sql = dict()
        transformed_sql['sql'] = query
        c = PARSER.pop_front(components)
        c_instance = eval(c)
        assert isinstance(c_instance, Root1)
        # parse multi-query
        if c_instance.id_c == 0:
            # intersect
            transformed_sql['intersect'] = dict()
            transformed_sql['intersect']['sql'] = query
            PARSER._transform(components, transformed_sql, schema)
            PARSER._transform(components, transformed_sql['intersect'], schema)
        elif c_instance.id_c == 1:
            # union
            transformed_sql['union'] = dict()
            transformed_sql['union']['sql'] = query
            PARSER._transform(components, transformed_sql, schema)
            PARSER._transform(components, transformed_sql['union'], schema)
        elif c_instance.id_c == 2:
            # except
            transformed_sql['except'] = dict()
            transformed_sql['except']['sql'] = query
            PARSER._transform(components, transformed_sql, schema)
            PARSER._transform(components, transformed_sql['except'], schema)
        else:
            # single query
            PARSER._transform(components, transformed_sql, schema)
        parse_result = PARSER.to_str(transformed_sql, 1, schema)
        parse_result = parse_result.replace('\t', '')
        return [parse_result, transformed_sql]

    @staticmethod
    def col_to_str(agg, col, tab, table_names, N=1):
        _col = col.replace(' ', '_')
        if agg == 'none':
            if tab not in table_names:
                table_names[tab] = 'T' + str(len(table_names) + N)
            table_alias = table_names[tab]
            if col == '*':
                return '*'
            return '%s.%s' % (table_alias, _col)
        else:
            if col == '*':
                if tab is not None and tab not in table_names:
                    table_names[tab] = 'T' + str(len(table_names) + N)
                return '%s(%s)' % (agg, _col)
            else:
                if tab not in table_names:
                    table_names[tab] = 'T' + str(len(table_names) + N)
                table_alias = table_names[tab]
                return '%s(%s.%s)' % (agg, table_alias, _col)

    @staticmethod
    def infer_from_clause(table_names, schema, columns):
        tables = list(table_names.keys())
        # print(table_names)
        start_table = None
        end_table = None
        join_clause = list()
        if len(tables) == 1:
            join_clause.append((tables[0], table_names[tables[0]]))
        elif len(tables) == 2:
            use_graph = True
            # print(schema['graph'].vertices)
            for t in tables:
                if t not in schema['graph'].vertices:
                    use_graph = False
                    break
            if use_graph:
                start_table = tables[0]
                end_table = tables[1]
                _tables = list(schema['graph'].dijkstra(tables[0], tables[1]))
                # print('Two tables: ', _tables)
                max_key = 1
                for t, k in table_names.items():
                    _k = int(k[1:])
                    if _k > max_key:
                        max_key = _k
                for t in _tables:
                    if t not in table_names:
                        table_names[t] = 'T' + str(max_key + 1)
                        max_key += 1
                    join_clause.append((t, table_names[t],))
            else:
                join_clause = list()
                for t in tables:
                    join_clause.append((t, table_names[t],))
        else:
            # > 2
            # print('More than 2 table')
            for t in tables:
                join_clause.append((t, table_names[t],))

        if len(join_clause) >= 3:
            star_table = None
            for agg, col, tab in columns:
                if col == '*':
                    star_table = tab
                    break
            if star_table is not None:
                star_table_count = 0
                for agg, col, tab in columns:
                    if tab == star_table and col != '*':
                        star_table_count += 1
                if star_table_count == 0 and \
                        ((end_table is None or end_table == star_table) or
                         (start_table is None or start_table == star_table)):
                    # Remove the table the rest tables still can join without star_table
                    new_join_clause = list()
                    for t in join_clause:
                        if t[0] != star_table:
                            new_join_clause.append(t)
                    join_clause = new_join_clause

        join_clause = ' JOIN '.join(['%s AS %s' % (jc[0], jc[1]) for jc in join_clause])
        return 'FROM ' + join_clause

    @staticmethod
    def to_str(sql_json, N_T, schema, pre_table_names=None):
        all_columns = list()
        select_clause = list()
        table_names = dict()
        current_table = schema
        for (agg, col, tab) in sql_json['select']:
            all_columns.append((agg, col, tab))
            select_clause.append(PARSER.col_to_str(agg, col, tab, table_names, N_T))
        select_clause_str = 'SELECT ' + ', '.join(select_clause).strip()

        sup_clause = ''
        order_clause = ''
        direction_map = {"des": 'DESC', 'asc': 'ASC'}

        if 'sup' in sql_json:
            (direction, agg, col, tab,) = sql_json['sup']
            all_columns.append((agg, col, tab))
            subject = PARSER.col_to_str(agg, col, tab, table_names, N_T)
            sup_clause = ('ORDER BY %s %s LIMIT 1' % (subject, direction_map[direction])).strip()
        elif 'order' in sql_json:
            (direction, agg, col, tab,) = sql_json['order']
            all_columns.append((agg, col, tab))
            subject = PARSER.col_to_str(agg, col, tab, table_names, N_T)
            order_clause = ('ORDER BY %s %s' % (subject, direction_map[direction])).strip()

        has_group_by = False
        where_clause = ''
        have_clause = ''
        if 'where' in sql_json:
            conjunctions = list()
            filters = list()
            # print(sql_json['where'])
            for f in sql_json['where']:
                if isinstance(f, str):
                    # and/or
                    conjunctions.append(f)
                else:
                    # op = "=, !=, <, >,..."
                    op, agg, col, tab, value = f
                    if value:
                        value['sql'] = sql_json['sql']
                    all_columns.append((agg, col, tab))
                    subject = PARSER.col_to_str(agg, col, tab, table_names, N_T)
                    if value is None:
                        where_value = '1'
                        if op == 'between':
                            where_value = '1 AND 2'
                        filters.append('%s %s %s' % (subject, op, where_value))
                    else:
                        if op == 'in' and \
                                len(value['select']) == 1 and \
                                value['select'][0][0] == 'none' and \
                                'where' not in value and \
                                'order' not in value and \
                                'sup' not in value:
                                # and value['select'][0][2] not in table_names:
                            if value['select'][0][2] not in table_names:
                                table_names[value['select'][0][2]] = 'T' + str(len(table_names) + N_T)
                            filters.append(None)

                        else:
                            subquery = '(' + PARSER.to_str(value, len(table_names) + 1, schema) + ')'
                            filters.append('%s %s %s' % (subject, op, subquery))
                    if len(conjunctions):
                        filters.append(conjunctions.pop())

            aggs = ['count(', 'avg(', 'min(', 'max(', 'sum(']
            having_filters = list()
            idx = 0
            while idx < len(filters):
                _filter = filters[idx]
                if _filter is None:
                    idx += 1
                    continue
                for agg in aggs:
                    if _filter.startswith(agg):
                        having_filters.append(_filter)
                        filters.pop(idx)
                        # print(filters)
                        if 0 < idx and (filters[idx - 1] in ['and', 'or']):
                            filters.pop(idx - 1)
                            # print(filters)
                        break
                else:
                    idx += 1

            if len(having_filters) > 0:
                have_clause = 'HAVING ' + ' '.join(having_filters).strip()

            if len(filters) > 0:
                # print(filters)
                filters = [_f for _f in filters if _f is not None]
                conjun_num = 0
                filter_num = 0
                for _f in filters:
                    if _f in ['or', 'and']:
                        conjun_num += 1
                    else:
                        filter_num += 1
                if conjun_num > 0 and filter_num != (conjun_num + 1):
                    # assert 'and' in filters
                    idx = 0
                    while idx < len(filters):
                        if filters[idx] == 'and':
                            if idx - 1 == 0:
                                filters.pop(idx)
                                break
                            if filters[idx - 1] in ['and', 'or']:
                                filters.pop(idx)
                                break
                            if idx + 1 >= len(filters) - 1:
                                filters.pop(idx)
                                break
                            if filters[idx + 1] in ['and', 'or']:
                                filters.pop(idx)
                                break
                        idx += 1
                if len(filters) > 0:
                    where_clause = 'WHERE ' + ' '.join(filters).strip()
                    where_clause = where_clause.replace('not_in', 'NOT IN')
                else:
                    where_clause = ''

            if len(having_filters) > 0:
                has_group_by = True

        for agg in ['count(', 'avg(', 'min(', 'max(', 'sum(']:
            if (len(sql_json['select']) > 1 and agg in select_clause_str) or \
                    agg in sup_clause or \
                    agg in order_clause:
                has_group_by = True
                break

        group_by_clause = ''
        if has_group_by:
            if len(table_names) == 1:
                # check none agg
                is_agg_flag = False
                for (agg, col, tab) in sql_json['select']:
                    if agg == 'none':
                        group_by_clause = 'GROUP BY ' + PARSER.col_to_str(agg, col, tab, table_names, N_T)
                    else:
                        is_agg_flag = True

                if is_agg_flag is False and len(group_by_clause) > 5:
                    group_by_clause = "GROUP BY"
                    for (agg, col, tab) in sql_json['select']:
                        group_by_clause = group_by_clause + ' ' + PARSER.col_to_str(agg, col, tab, table_names, N_T)

                if len(group_by_clause) < 5:
                    if 'count(*)' in select_clause_str:
                        current_table = schema
                        for primary in current_table['primary_keys']:
                            if current_table['table_names'][current_table['col_table'][primary]] in table_names:
                                group_by_clause = 'GROUP BY ' + \
                                                  PARSER.col_to_str('none', current_table['schema_content'][primary],
                                                                    current_table['table_names'][current_table['col_table'][primary]],
                                                                    table_names, N_T)
            else:
                # if only one select
                if len(sql_json['select']) == 1:
                    agg, col, tab = sql_json['select'][0]
                    non_lists = [tab]
                    fix_flag = False
                    # add tab from other part
                    for key, value in table_names.items():
                        if key not in non_lists:
                            non_lists.append(key)

                    a = non_lists[0]
                    b = None
                    for non in non_lists:
                        if a != non:
                            b = non
                    if b:
                        for pair in current_table['foreign_keys']:
                            t1 = current_table['table_names'][current_table['col_table'][pair[0]]]
                            t2 = current_table['table_names'][current_table['col_table'][pair[1]]]
                            if t1 in [a, b] and t2 in [a, b]:
                                if pre_table_names and t1 not in pre_table_names:
                                    assert t2 in pre_table_names
                                    t1 = t2
                                group_by_clause = 'GROUP BY ' + \
                                                  PARSER.col_to_str('none', current_table['schema_content'][pair[0]],
                                                                    t1, table_names, N_T)
                                fix_flag = True
                                break

                    if fix_flag is False:
                        agg, col, tab = sql_json['select'][0]
                        group_by_clause = 'GROUP BY ' + PARSER.col_to_str(agg, col, tab, table_names, N_T)

                else:
                    # check if there are only one non agg
                    non_agg, non_agg_count = None, 0
                    non_lists = []
                    for (agg, col, tab) in sql_json['select']:
                        if agg == 'none':
                            non_agg = (agg, col, tab)
                            non_lists.append(tab)
                            non_agg_count += 1

                    non_lists = list(set(non_lists))
                    # print(non_lists)
                    if non_agg_count == 1:
                        group_by_clause = 'GROUP BY ' + \
                                          PARSER.col_to_str(non_agg[0], non_agg[1], non_agg[2], table_names, N_T)
                    elif non_agg:
                        find_flag = False
                        fix_flag = False
                        find_primary = None
                        if len(non_lists) <= 1:
                            for key, value in table_names.items():
                                if key not in non_lists:
                                    non_lists.append(key)
                        if len(non_lists) > 1:
                            a = non_lists[0]
                            b = None
                            for non in non_lists:
                                if a != non:
                                    b = non
                            if b:
                                for pair in current_table['foreign_keys']:
                                    t1 = current_table['table_names'][current_table['col_table'][pair[0]]]
                                    t2 = current_table['table_names'][current_table['col_table'][pair[1]]]
                                    if t1 in [a, b] and t2 in [a, b]:
                                        if pre_table_names and t1 not in pre_table_names:
                                            assert t2 in pre_table_names
                                            t1 = t2
                                        group_by_clause = 'GROUP BY ' + \
                                                          PARSER.col_to_str('none',
                                                                            current_table['schema_content'][pair[0]],
                                                                            t1, table_names, N_T)
                                        fix_flag = True
                                        break
                        tab = non_agg[2]
                        assert tab in current_table['table_names']

                        for primary in current_table['primary_keys']:
                            if current_table['table_names'][current_table['col_table'][primary]] == tab:
                                find_flag = True
                                find_primary = (current_table['schema_content'][primary], tab)
                        if fix_flag is False:
                            if find_flag is False:
                                # rely on count *
                                foreign = []
                                for pair in current_table['foreign_keys']:
                                    if current_table['table_names'][current_table['col_table'][pair[0]]] == tab:
                                        foreign.append(pair[1])
                                    if current_table['table_names'][current_table['col_table'][pair[1]]] == tab:
                                        foreign.append(pair[0])

                                for pair in foreign:
                                    if current_table['table_names'][current_table['col_table'][pair]] in table_names:
                                        group_by_clause = 'GROUP BY ' + \
                                                          PARSER.col_to_str('none', current_table['schema_content'][pair],
                                                                            current_table['table_names'][current_table['col_table'][pair]], table_names, N_T)
                                        find_flag = True
                                        break
                                if find_flag is False:
                                    for (agg, col, tab) in sql_json['select']:
                                        if 'id' in col.lower():
                                            group_by_clause = 'GROUP BY ' + PARSER.col_to_str(agg, col, tab, table_names, N_T)
                                            break
                                    if len(group_by_clause) > 5:
                                        pass
                                    else:
                                        raise RuntimeError('fail to convert')
                            else:
                                group_by_clause = 'GROUP BY ' + PARSER.col_to_str('none', find_primary[0],
                                                                                  find_primary[1], table_names, N_T)
        intersect_clause = ''
        if 'intersect' in sql_json:
            sql_json['intersect']['sql'] = sql_json['sql']
            intersect_clause = 'INTERSECT ' + PARSER.to_str(sql_json['intersect'], len(table_names) + 1, schema, table_names)
        union_clause = ''
        if 'union' in sql_json:
            sql_json['union']['sql'] = sql_json['sql']
            union_clause = 'UNION ' + PARSER.to_str(sql_json['union'], len(table_names) + 1, schema, table_names)
        except_clause = ''
        if 'except' in sql_json:
            sql_json['except']['sql'] = sql_json['sql']
            except_clause = 'EXCEPT ' + PARSER.to_str(sql_json['except'], len(table_names) + 1, schema, table_names)

        # print(current_table['table_names_original'])
        table_names_replace = {}
        for a, b in zip(current_table['table_names_original'], current_table['table_names']):
            table_names_replace[b] = a
        new_table_names = {}
        for key, value in table_names.items():
            if key is None:
                continue
            new_table_names[table_names_replace[key]] = value
        from_clause = PARSER.infer_from_clause(new_table_names, schema, all_columns).strip()

        sql = ' '.join([select_clause_str, from_clause, where_clause, group_by_clause, have_clause, sup_clause, order_clause,
                        intersect_clause, union_clause, except_clause])

        return sql


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--table_path', type=str, help='table path', default="../../data/nl2sql/tables.json")
    arg_parser.add_argument('--input_path', type=str, help='predicted logical form',
                            default='../../data/mysemQL/dev.json')
    arg_parser.add_argument('--kb_relatedto', type=str, help='conceptNet data',
                            default="../../data/conceptNet/english_RelatedTo.pkl")
    arg_parser.add_argument('--kb_isa', type=str, help='conceptNet data',
                            default="../../data/conceptNet/english_IsA.pkl")
    arg_parser.add_argument('--output_path', type=str, help='output data',
                            default='../../data/predictedsql/train.json')
    args = arg_parser.parse_args()

    preporcess = PREPROCESS(args.table_path, args.kb_relatedto, args.kb_isa)
    postprocess = POSTPROCESS()
    parser = PARSER()
    # loading dataSets
    data = JSON.load(args.input_path)
    schemas = SCHEMA(args.table_path).table_dict
    results = []
    count = 0
    exception_count = 0
    with open(args.output_path, 'w', encoding='utf8') as f:
        for d in data:
            entry = dict()
            entry["db_id"] = copy.deepcopy(d["db_id"])
            entry["question"] = copy.deepcopy(d["question"])
            entry['question_toks'] = copy.deepcopy(d['question_toks'])
            entry["query"] = copy.deepcopy(d["query"])
            entry['model_result'] = copy.deepcopy(d["rule_label"])
            entry['rule_label'] = copy.deepcopy(d["rule_label"])

            preporcess.build_one(entry)
            schema = schemas[entry["db_id"]]
            postprocess.build(entry, schema)
            try:
                result, transformed_sql = parser.transform(entry, schema)
                results.append((entry["db_id"], entry["query"], result))
                f.write(result + '\n')
                count += 1
            except Exception as e:
                result, transformed_sql = parser.transform(entry, schema,
                                                           origin='Root1(3) Root(5) Sel(0) N(0) A(3) C(0) T(0)')
                exception_count += 1
                f.write(result + '\n')
                count += 1
                print(e)
                print('Exception')
                print(traceback.format_exc())
                print('===\n\n')
    print(count, exception_count)
