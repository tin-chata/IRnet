# Developed by J. Henning Viljoen at Chata Technologies.
""" This module will be able to generate a list of random Formal Representation (FR) queries, that will aim to maximise the probability that the FR generated are correct.
It will also generate the English answer, and questions that go with it."""


import random
import requests
import os
import argparse


from . import permutor_utils as utilities
from . import gen_ans_from_fr
from .data_utils import DbUtil
from .template_paraphraser import template_paraphraser

import pandas as pd

DUCKLING_TIME = 'DUCKLING_TIME'
DUCKLING_CURRENCY = 'DUCKLING_CURRENCY'
DUCKLING_AMOUNT = 'DUCKLING_AMOUNT'
NegativeDUCKLING_AMOUNT = '-DUCKLING_AMOUNT'
ValueLabelSubstring = 'VALUE_LABEL'

TrainingSetHuman = 'training_set_human'
DevSetHuman = 'dev_set_human'
TestSetHuman = 'test_set_human'
PermutorTrainingSet = 'permutor_training_set'
PermutorTestSet = 'permutor_test_set'
ReverseTranslationTrainingSet = 'reverse_translation_training_set'
ReverseTranslationTestSet = 'reverse_translation_test_set'

SelectEng = {'select': ['all', 'show me all', 'find all', 'list', 'list all', 'show us all']}

EqualTo = '='
GreaterThan = '>'
LessThan = '<'
DUCKLING_Ops = [EqualTo, GreaterThan, LessThan]
DUCKLING_Ops_Eng = {EqualTo: 'equal to', GreaterThan: 'greater than', LessThan: 'less than'}

DEFAULT = 'DEFAULT'
bar_plot = 'bar_plot'
bar_h_plot = 'bar_h_plot'
stacked_bar = 'stacked_bar'
stacked_column = 'stacked_column'
pie_chart = 'pie_chart'
heatmap = 'heatmap'
bubble = 'bubble'
line_plot = 'line_plot'
PlotLabels = [DEFAULT,
                bar_plot,
                bar_h_plot,
                stacked_bar,
                stacked_column,
                pie_chart,
                heatmap,
                bubble,
                line_plot]
CumProbDistPlotLabels = [0.82, 0.84, 0.86, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98]  #Cumulative probability distribution for the graph plots.
PlotLabelsEng = {DEFAULT: 'chart',
                bar_plot: 'column chart',
                bar_h_plot: 'bar chart',
                stacked_bar: 'stacked bar chart',
                stacked_column: 'stacked column chart',
                pie_chart: 'pie chart',
                heatmap: 'heat map',
                bubble: 'bubble chart',
                line_plot: 'line chart'}
COLUMN_TYPE = 'COLUMN_TYPE'
BAR_TYPE = 'BAR_TYPE'
STACKED_BAR_TYPE = 'STACKED_BAR_TYPE'
STACKED_COLUMN_TYPE = 'STACKED_COLUMN_TYPE'
PIE_TYPE = 'PIE_TYPE'
HEATMAP_TYPE = 'HEATMAP_TYPE'
BUBBLE_TYPE = 'BUBBLE_TYPE'
LINE_TYPE = 'LINE_TYPE'
PLOT_TYPES = {
    bar_plot: COLUMN_TYPE,
    bar_h_plot: BAR_TYPE,
    stacked_bar: STACKED_BAR_TYPE,
    stacked_column: STACKED_COLUMN_TYPE,
    pie_chart: PIE_TYPE,
    heatmap: HEATMAP_TYPE,
    bubble: BUBBLE_TYPE,
    line_plot: LINE_TYPE
}
CATEGORICAL = 'CATEGORICAL'
NUMERICAL = 'NUMERICAL'
OR_MORE = 'OR_MORE'
ALLOWABLE_DATA_TYPES = {
    COLUMN_TYPE: [{CATEGORICAL: 1, NUMERICAL: [1, OR_MORE]}],
    BAR_TYPE: [{CATEGORICAL: 1, NUMERICAL: [1, OR_MORE]}],
    STACKED_BAR_TYPE: [{CATEGORICAL: 2, NUMERICAL: [1, OR_MORE]}],
    STACKED_COLUMN_TYPE: [{CATEGORICAL: 2, NUMERICAL: [1, OR_MORE]}],
    PIE_TYPE: [{CATEGORICAL: 1, NUMERICAL: [1]}],
    HEATMAP_TYPE: [{CATEGORICAL: 2, NUMERICAL: [1]}],
    BUBBLE_TYPE: [{CATEGORICAL: 1, NUMERICAL: [3]}, {CATEGORICAL: 2, NUMERICAL: [1]}],
    LINE_TYPE: [{CATEGORICAL: 1, NUMERICAL: [1, OR_MORE]}, {CATEGORICAL: 2, NUMERICAL: [1]}]
}

LenDUCKLING_Ops = len(DUCKLING_Ops)
MaxNrOEToSampleForSelect = 1 #This can be increased later
MaxNrAggOpsToSample = 1
MaxNrOEToSampleForAggregation = 1
MaxNrGroupByOEsToSample = 2
MaxNrConditionLabelsToSample = 3
SelectAggregation = ['select', 'aggregation'] #All queries generated will be either of these categories.
CompareNoCompare = ['compare', ''] #have a compare or do not have a compare in the particular clause.
CompareNoCompareEng = {'compare' : 'compare', '' : ''}
QE_BASE_URI = 'http://localhost:9000'
#QE_BASE_URI = 'http://mystic-sound-143520.appspot.com'
#QE_BASE_URI = 'https://services.backend.dev.chata.ai'
prj_api_key = 'CHATAbcc13283ef1444019cfee3ba2194bbf7'
API = 'QBO'
QE_Project_ID = 1
NrQueriesToGenerate = 50 #5000 #500
PopulateDataBase = False
PopulateTestSet = False
ProbabilityInitialCompare = 0.05 #0.05 #Probability that a query will start with a compare statement
ProbabilitySelect = 0.4 #0.3
CumProbDistRowLimit = [0.8, 0.85, 0.9, 0.95] #Cumulative probability distribution for the Row Limit.  First value is the probability of not having a rowlimit in the query.
RowLimitLabels = ['1', '-1', DUCKLING_AMOUNT, NegativeDUCKLING_AMOUNT]

DefaultHumanQueries = [] #this list will normally be a parameter from a different module containing the human generated training queries.


class generate_fr_queries(object):
    def __init__(self, human_query_list, nr_queries_to_generate = NrQueriesToGenerate, populate_database=PopulateDataBase, table_to_populate=PermutorTrainingSet ,RunningFolderPath=str(),
                 NL_Entities_filename=str(),
                 nl_entity_groupablebyCSV_filename=str(),
                 nl_entity_filterablebyCSV_filename=str(),
                 template_paraphraserCSV=str(),
                 ParticularPatternLookedFor=str(),
                 ConnectionStringPostgres=str(),
                 training_set_human=str(),
                 dev_set_human=str(),
                 test_set_human=str(),
                 permutor_training_set=str(),
                 permutor_test_set=str(),
                 reverse_translation_training_set=str(),
                 reverse_translation_test_set=str()
                 ):
        self.RunningFolderPath = RunningFolderPath
        self.GeneratedCorpusCSV = self.RunningFolderPath + '/data/nl2fr/generated_corpus.csv'
        self.ConnectionStringPostgres = ConnectionStringPostgres

        self.fr_query_list = list()
        self.eng_ans_list = list()
        self.eng_query_list = list()

        self.eng_ans_and_fr_query_list = list()

        self.new_english_queries = list()
        self.new_fr_queries = list()

        self.project_id = QE_Project_ID
        self.init_stack() #init stack property
        self.fr_string = str()
        self.eng_answer_string = str()
        self.eng_query_string = str()
        self.human_query_list = human_query_list
        self.nr_queries_to_generate = nr_queries_to_generate
        self.populate_database = populate_database
        self.table_to_populate = table_to_populate
        self.mechanical_reverse = gen_ans_from_fr.fr_to_ans(RunningFolderPath=self.RunningFolderPath, NL_Entities_filename=NL_Entities_filename,
                              nl_entity_groupablebyCSV_filename=nl_entity_groupablebyCSV_filename, nl_entity_filterablebyCSV_filename=nl_entity_filterablebyCSV_filename)
        self.template_para = template_paraphraser(self.mechanical_reverse, RunningFolderPath=self.RunningFolderPath, template_paraphraserCSV=template_paraphraserCSV,
                                                  ParticularPatternLookedFor=ParticularPatternLookedFor)
        self.training_set_human = training_set_human
        self.dev_set_human = dev_set_human
        self.test_set_human = test_set_human
        self.permutor_training_set = permutor_training_set
        self.permuter_test_set = permutor_test_set
        self.reverse_translation_training_set = reverse_translation_training_set
        self.reverse_translation_test_set = reverse_translation_test_set
        """
        self.current_corpus = data_preparation(ConnectionStringPostgres=ConnectionStringPostgres, training_set_human = training_set_human,
            dev_set_human = dev_set_human,
            test_set_human = test_set_human,
            permutor_training_set = permutor_training_set,
            permutor_test_set = permutor_test_set,
            reverse_translation_training_set = reverse_translation_training_set,
            reverse_translation_test_set = reverse_translation_test_set,
            RootFolder=self.RunningFolderPath)
        """

        #self.current_corpus.get_training_set_list()
        #self.current_corpus.get_dev_test_lists()
        #self.current_corpus.get_reverse_translation_training_set_lists()
        #self.current_corpus.get_reverse_translation_test_set_lists()
        #self.prob_dist_oe = dict()
        #self.generate_prob_dist_oe()


    def init_stack(self):
        self.stack = {
            'initial_compare': list(),
            'select': list(),
            'aggregation': list(),
            'groupby': list(),
            'condition': list(),
            'plot': list(),
            'rowlimit': list()
        }


    def generate_prob_dist_oe(self): #generate a cumulative probability distribution for sample OEs to generate queries for the training corpus
        self.prob_dist_oe = gen_ans_from_fr.OE_Eng #first entry will be zero
        sum_of_prob_weights = 0.0
        for key in self.prob_dist_oe.keys():
            sum_of_prob_weights += gen_ans_from_fr.OE_Eng[key][1]
        for key in self.prob_dist_oe.keys():
            self.prob_dist_oe[key][1] = self.prob_dist_oe[key][1]/sum_of_prob_weights


    def generate_initial_compare(self):
        p = random.random()
        if p < ProbabilityInitialCompare:
            if (len(self.stack['groupby'])) == 1 and (len(self.stack['rowlimit']) == 0):
                self.stack['initial_compare'].append(CompareNoCompare[0])


    def generate_select(self):
        if not self.stack['initial_compare']:
            nr_entities = random.randrange(0, MaxNrOEToSampleForSelect) + 1
            oe_nodes = list()
            for node in self.mechanical_reverse.oe_eng.nodes():
                try:
                    if self.mechanical_reverse.oe_eng.nodes[node]['PermutorInclude'] == 1:
                        if self.mechanical_reverse.oe_eng.nodes[node]['NodeType'] == 'OntologicalEntity':
                            oe_nodes.append(node)
                        else: #node is an attribute
                            for outgoing_node in self.mechanical_reverse.oe_eng.successors(node):
                                outgoing_edges = self.mechanical_reverse.oe_eng.get_edge_data(node, outgoing_node)
                                for outgoing_edge_key in outgoing_edges:
                                    if outgoing_edges[outgoing_edge_key]['relationship'] == 'FilterableBy':
                                        oe_nodes.append(node)
                except:
                    pass
            entities = random.sample(oe_nodes, nr_entities)
            self.stack['select'] = entities
            if len(self.stack['select']) > 1:
                for oe in self.stack['select'][1:]:
                    pass


    def generate_aggregation(self):
        nr_aggops = random.randrange(0, MaxNrAggOpsToSample) + 1
        #aggops = random.sample(gen_ans_from_fr.AggOpsEng.keys(), nr_aggops)
        aggops = utilities.sample_from_discrete_cum_nr_of_times_distinct(list(gen_ans_from_fr.AggOpsEng.keys()), gen_ans_from_fr.CumProbDistAggOps, nr_aggops)
        #entities = random.sample(gen_ans_from_fr.OE_Eng.keys(), nr_aggops)
        #entities = utilities.sample_from_ontological_entities_nr_of_times_distinct(self.prob_dist_oe, MaxNrMainEntitiesToSample)
        nr_entities = random.randrange(0, MaxNrOEToSampleForAggregation) + 1
        oe_nodes = list()
        for node in self.mechanical_reverse.oe_eng.nodes():
            try:
                if self.mechanical_reverse.oe_eng.nodes[node]['PermutorInclude'] == 1:
                    if self.mechanical_reverse.oe_eng.nodes[node]['NodeType'] == 'OntologicalEntity':
                        if (self.mechanical_reverse.oe_eng.nodes[node]['DefaultAggregate'] != '') and \
                                (self.mechanical_reverse.oe_eng.nodes[node]['DefaultAggregate'] != 'NULL'):
                            oe_nodes.append(node)
                    else: #The node is an Attribute
                        for outgoing_node in self.mechanical_reverse.oe_eng.successors(node):
                            outgoing_edges = self.mechanical_reverse.oe_eng.get_edge_data(node, outgoing_node)
                            for outgoing_edge_key in outgoing_edges:
                                if (outgoing_edges[outgoing_edge_key]['relationship'] == 'FilterableBy' or outgoing_edges[outgoing_edge_key]['relationship'] == 'GroupableBy') and \
                                        (self.mechanical_reverse.oe_eng.nodes[node]['ClassName'] != 'QUANTITY' or self.mechanical_reverse.oe_eng.nodes[node]['ClassName'] != 'DOLLAR_AMT'):
                                    oe_nodes.append(node)
            except:
                pass
        entities = random.sample(oe_nodes, nr_entities)

        for i in range(nr_aggops):
            self.stack['aggregation'].append(aggops[i])
            if i > 0:
                pass
            self.stack['aggregation'].append(entities[i])


    def generate_select_or_aggregation(self):
        if self.stack['initial_compare']:
            self.generate_aggregation()
        else:
            r = random.random()
            if r <= ProbabilitySelect:
                self.generate_select()
            else:
                self.generate_aggregation()


    def generate_groupby(self):
        if self.stack['aggregation']:
            nl_entities_only = list()
            nl_entities_only_immediate_parent_nodes = list()
            for token in self.stack['aggregation']:
                if (token not in gen_ans_from_fr.AggOpsEng) and (token != gen_ans_from_fr.compare):
                    nl_entities_only.append(token)
                    nl_entities_only_immediate_parent_nodes.append('.'.join(token.split('.')[:-1]))
            nr_groupbys = random.randrange(0, MaxNrGroupByOEsToSample + 1)

            groupbys_possibilities = list()
            for main_entity in nl_entities_only:
                list_for_entity = list()
                for outgoing_node in self.mechanical_reverse.oe_eng.successors(main_entity):
                    incoming_edges = self.mechanical_reverse.oe_eng.get_edge_data(main_entity, outgoing_node)
                    for incoming_edge_key in incoming_edges:
                        if incoming_edges[incoming_edge_key]['relationship'] == 'GroupableBy':
                            list_for_entity.append(outgoing_node)
                groupbys_possibilities.append(list_for_entity)

            all_valid_groupbys = list()
            running_set = set(groupbys_possibilities[0])
            for entity_list in groupbys_possibilities:
                running_set = running_set & set(entity_list)
            all_valid_groupbys = list(running_set)
            len_all_valid_groupbys = len(all_valid_groupbys)

            groupbys = list()
            if nr_groupbys > 0 and len_all_valid_groupbys > 0:
                if nr_groupbys > len(all_valid_groupbys):
                    nr_groupbys = len(all_valid_groupbys)
                nr_date_groupbys = 2 #initial value for while loop
                while nr_date_groupbys > 1:
                    groupbys = random.sample(all_valid_groupbys, nr_groupbys)
                    nr_date_groupbys = 0
                    for one_groupby in groupbys:
                        if ('hour_of_day' in one_groupby) or ('day_of_month' in one_groupby) or ('week' in one_groupby) or \
                                ('month' in one_groupby) or ('fiscal_quarter' in one_groupby) or ('year' in one_groupby):
                            nr_date_groupbys += 1

            for i in range(len(groupbys)):
                applicable_path = groupbys[i]
                neighbour = groupbys[i]
                immediate_parent = '.'.join(groupbys[i].split('.')[:-1])
                try:
                    default_groupby_found = self.mechanical_reverse.oe_eng.nodes[groupbys[i]]['DefaultGroupBy']
                except:
                    print('DefaultGroupby or node not found for ', groupbys[i])
                if (self.mechanical_reverse.oe_eng.nodes[groupbys[i]]['DefaultGroupBy'] == True) and (immediate_parent not in nl_entities_only_immediate_parent_nodes):
                    groupbys[i] = immediate_parent
                else:
                    use_leaf_nodes = False
                    all_main_entities_in_default_group_path = [False]*len(nl_entities_only)
                    for main_entity_i in range(len(nl_entities_only)):
                        parent_entity_0 = self.mechanical_reverse.oe_eng.nodes[groupbys[i]]['ParentEntity']
                        if parent_entity_0 != 'NULL' and parent_entity_0 != '' and parent_entity_0 == nl_entities_only[main_entity_i]:
                            all_main_entities_in_default_group_path[main_entity_i] = True
                        else:
                            if applicable_path.split('.')[-1] in ['hour_of_day', 'day_of_month', 'week', 'month',
                                                                  'fiscal_quarter', 'year']:
                                applicable_path = '.'.join(applicable_path.split('.')[:-1]) + '.date'
                            path_split = applicable_path.split('.')
                            len_path_split = len(path_split)
                            looked_up_value = self.mechanical_reverse.oe_eng.nodes['.'.join(path_split[:-1])]['DefaultDate']
                            while (len_path_split > 1) and (
                                    (applicable_path == looked_up_value) or neighbour == looked_up_value):
                                applicable_path = '.'.join(path_split[:-1])
                                path_split = applicable_path.split('.')
                                len_path_split = len(path_split)
                                if len_path_split > 1:
                                    looked_up_value = self.mechanical_reverse.oe_eng.nodes['.'.join(path_split[:-1])]['DefaultDate']
                                else:
                                    looked_up_value = ''  # in this case the loop should end
                            if applicable_path == nl_entities_only[main_entity_i]: all_main_entities_in_default_group_path[main_entity_i] = True

                    if not False in all_main_entities_in_default_group_path:
                        groupbys[i] = groupbys[i].split('.')[-1]

            #if use_leaf_nodes == True:
            #    for i in range(len(groupbys)):
            #        groupbys[i] = groupbys[i].split('.')[-1]

            aggops_entities_str = ' '.join(groupbys)
            self.stack['groupby'] = groupbys
            for groupby in self.stack['groupby']:
                pass


    def generate_duckling_currency(self, duckling_time_field):
        duckling_op = random.sample(DUCKLING_Ops, 1)[0]
        condition_string = duckling_time_field + ' ' + duckling_op + ' ' + 'DUCKLING_CURRENCY'
        duckling_op_eng = DUCKLING_Ops_Eng[duckling_op]
        return condition_string


    def generate_condition(self):
        nr_conditions = random.randrange(0, MaxNrConditionLabelsToSample + 1)
        nl_entities_only = list()
        nl_entities_only_root_nodes = list()
        list_for_query = list()
        if self.stack['aggregation'] == []:
            list_for_query = self.stack['select']
        else:
            list_for_query = self.stack['aggregation']
        for token in list_for_query:
            if (token not in gen_ans_from_fr.AggOpsEng) and (token != gen_ans_from_fr.compare):
                nl_entities_only.append(token)
                nl_entities_only_root_nodes.append(token.split('.')[0])

        conditions_possibilities = list()
        for main_entity in nl_entities_only:
            dict_for_entity = dict()
            for outgoing_node in self.mechanical_reverse.oe_eng.successors(main_entity):
                incoming_edges = self.mechanical_reverse.oe_eng.get_edge_data(main_entity, outgoing_node)
                for incoming_edge_key in incoming_edges:
                    if incoming_edges[incoming_edge_key]['relationship'] == 'FilterableBy':
                        possible_values = incoming_edges[incoming_edge_key]['EnumerationConditions']
                        dict_for_entity[outgoing_node] = possible_values
            conditions_possibilities.append(dict_for_entity)

        all_valid_conditions = list()
        running_set = set(conditions_possibilities[0].keys())
        for entity_dict in conditions_possibilities:
            running_set = running_set & set(entity_dict.keys())
        all_valid_conditions = list(running_set)
        len_all_valid_conditions = len(all_valid_conditions)

        conditions = list()
        if nr_conditions > 0 and len_all_valid_conditions > 0:
            if nr_conditions > len_all_valid_conditions:
                nr_conditions = len_all_valid_conditions
            conditions = random.sample(all_valid_conditions, nr_conditions)

        len_conditions = len(conditions)
        #condition_keys = random.sample(ConditionLabels.keys(), nr_conditions)
        condition_keys = [str()] * len_conditions
        condition_values = [str()]*len_conditions
        for i in range(len_conditions):
            condition_keys[i] = conditions[i]
            sampled_value = random.sample(conditions_possibilities[0][condition_keys[i]], 1)[0]
            if sampled_value == gen_ans_from_fr.DUCKLING_CURRENCY:
                sampled_opp = random.sample(DUCKLING_Ops, 1)[0]
            else:
                sampled_opp = '='
            condition_values[i] = sampled_opp + ' ' + sampled_value
            #condition_string = str()
            """if condition_key not in ['DUCKLING_CURRENCY', 'DUCKLING_TIME']:
                condition_string += ConditionLabels[condition_key][0] + ' ' + EqualTo + ' ' + condition_key
            else:
                duckling_field = ''
                if len(self.stack['aggregation']) > 0:
                    duckling_field += self.stack['aggregation'][1]
                else:
                    duckling_field += self.stack['select'][0]
                if len(self.stack['aggregation']) > 2:
                    condition_string += OBJECT + ' ' + EqualTo + ' ' + condition_key
                elif condition_key == 'DUCKLING_TIME':
                    condition_string += duckling_field + ' ' + EqualTo + ' ' + condition_key
                elif condition_key == 'DUCKLING_CURRENCY':
                    condition_string = self.generate_duckling_currency(duckling_field)"""

        for i in range(len_conditions):
            applicable_node = condition_keys[i]
            moved_up_with_defaults = False
            searching = True
            while searching == True:
                searching = False
                for node in self.mechanical_reverse.oe_eng.nodes():
                    try:
                        if self.mechanical_reverse.oe_eng.nodes[node]['DefaultDate'] == applicable_node or \
                                self.mechanical_reverse.oe_eng.nodes[node]['DefaultCurrency'] == applicable_node:
                            applicable_node = node
                            searching = True
                            moved_up_with_defaults = True
                    except:
                        pass

            if moved_up_with_defaults == False: #no DefaultDate or DefaultCurrency
                if 'VALUE_LABEL' in condition_values[i]:
                    parent_entity = self.mechanical_reverse.oe_eng.nodes[condition_keys[i]]['ParentEntity']
                    if parent_entity != 'NULL':
                        applicable_node = parent_entity
                else:
                    can_use_leaf_node = True
                    for main_entity in nl_entities_only:
                        if self.mechanical_reverse.oe_eng.nodes[condition_keys[i]]['ParentEntity'] != main_entity: can_use_leaf_node = False
                    if can_use_leaf_node == True: applicable_node = condition_keys[i].split('.')[-1]

            condition_keys[i] = applicable_node
            self.stack['condition'].append(condition_keys[i] + ' ' + condition_values[i])


    def generate_rowlimit(self):
        if len(self.stack['groupby']) == 1:
            rowlimit_label = utilities.sample_from_discrete_cum(RowLimitLabels, CumProbDistRowLimit)
            if len(rowlimit_label) > 0:
                #sign = rowlimit_label[0]
                #if rowlimit_label[0] == '+': sign = ''
                #self.stack['rowlimit'].append(sign + gen_ans_from_fr.DUCKLING_AMOUNT)
                self.stack['rowlimit'].append(rowlimit_label[0])


    def generate_plots(self):
        if (len(self.stack['initial_compare']) == 0) and (len(self.stack['aggregation']) > 0) and (len(self.stack['groupby']) > 0) and (len(self.stack['rowlimit']) == 0):
            plot_label = utilities.sample_from_discrete_cum(PlotLabels, CumProbDistPlotLabels)
            if plot_label and plot_label[0] != DEFAULT:
                numerical = len(self.stack['aggregation']) / 2
                categorical = len(self.stack['groupby'])
                plot_type = PLOT_TYPES[plot_label[0]]
                plot_good = False
                for type_detail in ALLOWABLE_DATA_TYPES[plot_type]:
                    plot_categorical_good = False
                    plot_numercial_good = False
                    if categorical == type_detail[CATEGORICAL]:
                        plot_categorical_good = True
                    for numerical_detail in type_detail[NUMERICAL]:
                        if numerical_detail == OR_MORE:
                            plot_numercial_good = True
                        elif numerical_detail == numerical:
                            plot_numercial_good = True
                    if not plot_good: plot_good = plot_categorical_good and plot_numercial_good
                if not plot_good:
                    return 0 #the plot picked will not work for the type of query, so we exit without inserting in the stack.

            self.stack['plot'] = plot_label
            if self.stack['plot']:
                pass


    def generate_formal_representation(self):
        self.fr_string = ''
        if self.stack['initial_compare']:
            self.fr_string += ' '.join(self.stack['initial_compare']) + ' '
        if self.stack['select']:
            self.fr_string += 'select ' + ' '.join(self.stack['select']) + ' '
        if self.stack['aggregation']:
            self.fr_string += 'aggregation ' + ' '.join(self.stack['aggregation']) + ' '
        if self.stack['groupby']:
            self.fr_string += 'groupby ' + ' '.join(self.stack['groupby']) + ' '
        if self.stack['condition']:
            self.fr_string += 'condition ' + ' '.join(self.stack['condition']) + ' '
        if self.stack['plot']:
            self.fr_string += 'plot ' + ' '.join(self.stack['plot'])
        if self.stack['rowlimit']:
            self.fr_string += 'rowlimit ' + ' '.join(self.stack['rowlimit'])

        self.fr_string = self.fr_string.strip()


    def prepare_query_for_qe(self, one_query):
        query_list = one_query.split()
        nr_duckling_time = 0
        nr_duckling_currency = 0
        nr_duckling_amount = 0
        changed_query = str()
        new_word = str()
        for word in query_list:
            if word == DUCKLING_TIME:
                new_word = DUCKLING_TIME + str(nr_duckling_time)
                nr_duckling_time += 1
            elif word == DUCKLING_AMOUNT:
                new_word = DUCKLING_AMOUNT + str(nr_duckling_amount)
                nr_duckling_amount += 1
            elif word == DUCKLING_CURRENCY:
                new_word = DUCKLING_CURRENCY + str(nr_duckling_currency)
                nr_duckling_currency += 1
            elif ValueLabelSubstring in word:
                new_word = word + str(0)
            else:
                new_word = word
            changed_query += new_word + ' '
        return changed_query.strip()


    def generate_dict(self, one_query):
        dict = {
                "DUCKLING_TIME1": "{\"text\": \"this year\", \"value\": \"2018-01-01T00:00:00.000Z\", \"entity\": \"time\", \"additional_info\": {\"grain\": \"year\", \"value\": \"2018-01-01T00:00:00.000Z\"}}",
                "DUCKLING_CURRENCY0": "{\"text\": \"$500\", \"value\": 500, \"entity\": \"amount-of-money\", \"additional_info\": {\"unit\": \"$\", \"value\": 500}}",
                "ITEM_VALUE_LABEL0": "seals",
                "DUCKLING_TIME0": "{\"text\": \"older than 90 days\", \"value\": {\"to\": \"2017-11-29T08:00:00.000Z\", \"from\": null}, \"entity\": \"time\", \"additional_info\": {\"value\": {\"to\": \"2017-11-29T08:00:00.000Z\", \"from\": null}}}",
                "CUSTOMER_VALUE_LABEL0": "cod",
                "CUSTOMER_VALUE_LABEL1": "jonny depp",
                "ACCOUNT_VALUE_LABEL0" : "travel",
                "INVOICE_VALUE_LABEL0" : "ab",
                "VENDOR_VALUE_LABEL0" : "telus",
                "TRACKING_CATEGORY_VALUE_LABEL0" : "residential",
                "BILL_ADDRESS_VALUE_LABEL0" : "canada",
                "DUCKLING_AMOUNT0": "{\"text\": \"5\", \"value\": 5, \"entity\": \"number\", \"additional_info\": {\"type\": \"value\", \"value\": 5}}",
                "representation": one_query
                }
        return dict


    def query_engine_output_good(self, one_query):
        try:
            post_request = requests.post("{}/qengine/api/v1/sql?project={}&api={}".format(QE_BASE_URI, self.project_id,
                                                              API), json=self.generate_dict(one_query))
        except:
            return False
        if not post_request or post_request.status_code is not 200:
            return False
        return True


    def remove_zeros_from_one_query(self, the_query):
        filtered_query = the_query.replace('0', '')
        return filtered_query


    def remove_zeros_from_list(self, the_list):
        for i in range(len(the_list)):
            the_list[i] = self.remove_zeros_from_one_query(the_list[i])


    def generate_query_list(self, api_id=None, model_id=None):
        nr_unique_queries_generated = 0
        query_addition_per_iteration = 0.0
        prev_iteration_query_added = 0
        for i in range(self.nr_queries_to_generate):
            self.init_stack()
            self.generate_select_or_aggregation()
            self.generate_groupby()
            self.generate_condition()
            self.generate_rowlimit()
            self.generate_initial_compare()
            self.generate_plots()
            #self.add_line_item_s_if_needed()
            self.generate_formal_representation()

            if self.fr_string not in self.fr_query_list and self.remove_zeros_from_one_query(self.fr_string) not in self.human_query_list:
                print('')
                print(i, '|', nr_unique_queries_generated, '|', query_addition_per_iteration, ':', self.fr_string)
                #q_for_qe = self.prepare_query_for_qe(self.fr_string)
                #print(q_for_qe)
                qe_response_good = True #just skipping QE for now.
                #qe_response_good = self.query_engine_output_good(q_for_qe)
                #print(qe_response_good)
                if qe_response_good:
                    reverse_translation_paraphrase_list, full_path_fr = self.mechanical_reverse.translate_and_paraphrase(self.fr_string)
                    print('Full path FR:', full_path_fr)
                    template_paraphraser_paraphrase_list = self.template_para.paraphrase_formal_representation(full_path_fr)
                    if (template_paraphraser_paraphrase_list != None) and len(template_paraphraser_paraphrase_list) > 0:
                        reverse_translation_paraphrase_list = reverse_translation_paraphrase_list + template_paraphraser_paraphrase_list
                    for one_translate_para in reverse_translation_paraphrase_list:
                        self.eng_answer_string = one_translate_para
                        print(self.eng_answer_string)
                        if self.eng_answer_string != self.mechanical_reverse.reverse_translation_error:
                            if self.eng_answer_string not in self.eng_ans_list:
                                self.fr_query_list.append(self.fr_string)
                                self.eng_ans_list.append(self.eng_answer_string)
                                eng_and_fr = [self.eng_answer_string, self.fr_string]
                                if api_id:
                                    eng_and_fr = [api_id, model_id, self.eng_answer_string, self.fr_string]
                                self.eng_ans_and_fr_query_list.append(eng_and_fr)
                                nr_unique_queries_generated += 1
                                if prev_iteration_query_added < i:
                                    query_addition_per_iteration = 1.0/(i - prev_iteration_query_added)
                                    prev_iteration_query_added = i
                        else:
                            print('Reverse translation error for:', self.fr_string)
        print()
        print('Nr unique queries generated:',len(self.eng_ans_and_fr_query_list))
        print('Number of times particular FR hit:', self.template_para.nr_times_particular_pattern_hit)



    def write_csv(self, file_to_write_to, list_to_write, column_header):
        df = pd.DataFrame(list_to_write)
        df.to_csv(file_to_write_to, index=False, header=False)


    def generate_and_test_and_insert(self, api_id=None, model_id=None):
        self.generate_query_list(api_id, model_id)
        self.write_csv(self.GeneratedCorpusCSV, self.eng_ans_and_fr_query_list, ['english_query', 'formal_representation'])
        if self.populate_database:
            if self.table_to_populate == TestSetHuman:
                #self.current_corpus.truncate_populate_specific_table(self.GeneratedCorpusCSV, self.current_corpus.test_set_human)
                DbUtil.truncate_populate_specific_table(self.ConnectionStringPostgres, self.GeneratedCorpusCSV, self.test_set_human, api_id)
            elif self.table_to_populate == DevSetHuman:
                #self.current_corpus.truncate_populate_specific_table(self.GeneratedCorpusCSV, self.current_corpus.dev_set_human)
                DbUtil.truncate_populate_specific_table(self.ConnectionStringPostgres, self.GeneratedCorpusCSV, self.dev_set_human, api_id)
            elif self.table_to_populate == PermutorTrainingSet:
                #self.current_corpus.populate_permutor_training_set_payment(self.GeneratedCorpusCSV)
                print (self.GeneratedCorpusCSV, self.permutor_training_set)
                DbUtil.truncate_populate_specific_table(self.ConnectionStringPostgres, self.GeneratedCorpusCSV, self.permutor_training_set, api_id)
            elif self.table_to_populate == PermutorTestSet:
                #self.current_corpus.truncate_populate_specific_table(self.GeneratedCorpusCSV, self.current_corpus.permutor_test_set)
                DbUtil.truncate_populate_specific_table(self.ConnectionStringPostgres, self.GeneratedCorpusCSV, self.permutor_test_set, api_id)
                pass
            os.remove(self.GeneratedCorpusCSV) #delete the generated csv file

"""TrainingSetHuman = 'training_set_human'
DevSetHuman = 'dev_set_human'
TestSetHuman = 'test_set_human'
PermutorTrainingSet = 'permutor_training_set'
PermutorTestSet = 'permutor_test_set'
ReverseTranslationTrainingSet = 'reverse_translation_training_set'
ReverseTranslationTestSet = 'reverse_translation_test_set'"""


def add_arguments(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument("--nr_iterations", type=int, default=NrQueriesToGenerate, help="Number of iterations of stochastic graph search algorithm to run.")
    parser.add_argument("--populate_database", type="bool", nargs="?", const=True, default=PopulateDataBase, help="Whether to populate the training database automatically once generation is completed.")
    parser.add_argument("--table_to_populate", type=str, default=PermutorTrainingSet, help="Mode of app.")


if __name__ == "__main__":
    prod_queries_parser = argparse.ArgumentParser()
    add_arguments(prod_queries_parser)
    local_flags, unparsed = prod_queries_parser.parse_known_args()
    nr_queries_to_generate = local_flags.nr_iterations
    populate_database = local_flags.populate_database
    table_to_populate = local_flags.table_to_populate

    #qbo/xero
    """query_generator = generate_fr_queries(DefaultHumanQueries,
                                          nr_queries_to_generate=nr_queries_to_generate,
                                          populate_database=populate_database,
                                          RunningFolderPath='/stripe-neural-parser',
                                          NL_Entities_filename='stripe_NL-Entities_mod.csv',
                                          nl_entity_groupablebyCSV_filename='stripe_nl_entity_groupableby.csv',
                                          nl_entity_filterablebyCSV_filename='stripe_nl_entity_filterableby.csv',
                                          template_paraphraserCSV='template_paraphraser.csv',
                                          ParticularPatternLookedFor='aggregation count ENTITY0',
                                          ConnectionStringPostgres='postgresql://payment_training_data_user:uGJHdt0vQ&25@35.226.48.166:5432/payment_training_data',
                                          training_set_human='training_set_human_payment',
                                          dev_set_human='dev_set_human_payment',
                                          test_set_human='test_set_human_payment',
                                          permutor_training_set='permutor_training_set_payment',
                                          permutor_test_set='permutor_test_set',
                                          reverse_translation_training_set='reverse_translation_training_set',
                                          reverse_translation_test_set='reverse_translation_test_set'
                                          )"""

    #stripe
    query_generator = generate_fr_queries(DefaultHumanQueries,
                                          nr_queries_to_generate=nr_queries_to_generate,
                                          populate_database=populate_database,
                                          RunningFolderPath='/stripe-neural-parser',
                                          NL_Entities_filename='stripe_NL-Entities_mod.csv',
                                          nl_entity_groupablebyCSV_filename='stripe_nl_entity_groupableby.csv',
                                          nl_entity_filterablebyCSV_filename='stripe_nl_entity_filterableby.csv',
                                          template_paraphraserCSV='template_paraphraser.csv',
                                          ParticularPatternLookedFor='aggregation count ENTITY0',
                                          ConnectionStringPostgres='postgresql://payment_training_data_user:uGJHdt0vQ&25@35.226.48.166:5432/payment_training_data',
                                          training_set_human='training_set_human_payment',
                                          dev_set_human='dev_set_human_payment',
                                          test_set_human='test_set_human_payment',
                                          permutor_training_set='permutor_training_set_payment',
                                          permutor_test_set='permutor_test_set',
                                          reverse_translation_training_set='reverse_translation_training_set',
                                          reverse_translation_test_set='reverse_translation_test_set'
                                          )

    #property demo project
    """query_generator = generate_fr_queries(DefaultHumanQueries,
                                          nr_queries_to_generate=nr_queries_to_generate,
                                          populate_database=populate_database,
                                          table_to_populate=table_to_populate,
                                          RunningFolderPath='/property-neural-parser',
                                          NL_Entities_filename='property_NL_Entities_mod.csv',
                                          nl_entity_groupablebyCSV_filename='property_nl_entity_groupableby.csv',
                                          nl_entity_filterablebyCSV_filename='property_nl_entity_filterableby.csv',
                                          template_paraphraserCSV='property_template_paraphraser.csv',
                                          ParticularPatternLookedFor='aggregation count ENTITY0',
                                          ConnectionStringPostgres='postgresql://property_training_data_user:gY2QSbkRUUy7zJ2B@35.226.48.166:5432/property_training_data',
                                          training_set_human='training_set_human_property',
                                          dev_set_human='dev_set_human_property',
                                          test_set_human='test_set_human_property',
                                          permutor_training_set='permutor_training_set_property',
                                          permutor_test_set='permutor_test_set_property',
                                          reverse_translation_training_set='reverse_translation_training_set_property',
                                          reverse_translation_test_set='reverse_translation_test_set_property'
                                          )"""

    query_generator.generate_and_test_and_insert()