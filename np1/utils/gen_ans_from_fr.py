"""J. Henning Viljoen, Chata Technologies"""
"""This module will provide a class that can generate the English answer given Formal Representation according to the Reverse Translation spec."""

from enum import Enum
import networkx as nx
import pandas as pd
import math as math
import random

from .data_utils import Util


FilterableBy = 'FilterableBy'


compare = 'compare'
contrast = 'contrast'
help_identifier = 'help_identifier'
select = 'select'
aggregation = 'aggregation'
groupby = 'groupby'
orderby = 'orderby'
condition = 'condition'
having = 'having'
rowlimit = 'rowlimit'
plot = 'plot'

substring_in_list = Util.substring_in_list

Clauses = [help_identifier, select, aggregation, groupby, orderby, condition, having, rowlimit, plot] #compare should not be here since we should not look for it later in the FR string

help_identifier_Eng = {help_identifier: 'help on'}

#For the OE_Eng dict, each value entry will be a list like [English translation, weight probability for random generation selection


OE_ForEndOfLists = ['line_item'] #if these entities are in a select or aggregation clause, we will try and put them at the end of that clause in the reverse translation.

CumProbDistAggOps = [0, 0.6, 0.90, 0.95, 0.975]
AggOpsEng = {'sum': ['total'], 'avg': ['average'], 'count': ['number of', 'how many'], 'min': ['minimum'], 'max': ['maximum']}
TopSynonyms = ['highest', 'top']
BottomSynonyms = ['lowest', 'bottom']



LabelsEng = {"''": 'empty',
             "accepted": 'accepted status',
             "asset": 'asset',
             "c": 'cleared',
             "closed": 'closed status',
             "equity": 'equity',
             "expense": 'expense',
             "expired": 'expired',
             "INVENTORY": 'inventory',
             "liability": 'liability',
             "open": 'open status',
             "pending": 'pending status',
             "r": 'reconciled',
             "rejected": 'rejected status',
             "revenue": 'revenue'}


line_item = 'line_item'


EqualTo = '='
GreaterThan = '>'
GreaterThanOrEqualTo = '>='
LessThan = '<'
Not = '!='
Ops_Eng = {EqualTo: ['for', ''], GreaterThan: ['greater than', 'more than', 'over'], GreaterThanOrEqualTo: ['greater than or equal to'], LessThan: ['less than', 'under'], Not: ['not']}
Ops_Eng_Having = {EqualTo: 'as', GreaterThan: 'greater than', LessThan: 'less than'}
GreaterLessOps = [GreaterThan, LessThan]

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
CumProbDistPlotLabels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  #Cumulative probability distribution for the graph plots.
PlotLabelsEng = {DEFAULT: ['chart', 'plot', 'graph'],
                bar_plot: ['column chart', 'column plot', 'column graph'],
                bar_h_plot: ['bar chart', 'bar plot', 'bar graph'],
                stacked_bar: ['stacked bar chart', 'stacked bar plot', 'stacked bar graph'],
                stacked_column: ['stacked column chart', 'stacked column plot', 'stacked column graph'],
                pie_chart: ['pie chart', 'pie plot', 'pie graph'],
                heatmap: ['heat map', 'heat map plot', 'heat map graph'],
                bubble: ['bubble chart', 'bubble plot', 'bubble graph'],
                line_plot: ['line chart', 'line plot', 'line graph']}

ReverseTranslationError = '500'

DUCKLING_TIME = 'DUCKLING_TIME'
DUCKLING_AMOUNT = 'DUCKLING_AMOUNT'
DUCKLING_CURRENCY = 'DUCKLING_CURRENCY'
DucklingSubString = 'DUCKLING'
ValueLabelSubstring = 'VALUE_LABEL'
EnumStr_true = 'true'
EnumStr_false = 'false'

class condition_clause_label_types(Enum):
    Duckling = 0
    ValueLabel = 1
    Numeric = 2
    Other = 3
    NoCondition = 4
    SpecifiedLabel = 5

OBJECT = 'OBJECT'


DucklingsDefaultKeys = {'DUCKLING_TIME': 'DefaultDate',
                        'DUCKLING_CURRENCY': 'DefaultCurrency',
                        'DUCKLING_AMOUNT': 'DefaultAmount'}

EquationOperations_Eng = {'add': 'plus',
                          'subtract': 'minus',
                          'multiply': 'times',
                          'divide': 'divided by'}

class ModesForReverseTranslation(Enum):
    Full = 0
    Shortened = 1
    ShortenedEnumInFrontFieldInBack = 2
    #Adjectives = 2

EOS = '<eos>'

#QBO/Xero
#FRQueryToBackTranslate = 'compare aggregation sum expense groupby month condition vendor = VENDOR_VALUE_LABEL expense = DUCKLING_TIME'
#FRQueryToBackTranslate = 'aggregation sum account_receivable'
#FRQueryToBackTranslate = 'aggregation sum sale groupby product condition customer = CUSTOMER_VALUE_LABEL sale = DUCKLING_TIME rowlimit DUCKLING_AMOUNT'
#FRQueryToBackTranslate = 'aggregation avg sale sum sale groupby product'
#FRQueryToBackTranslate = 'select bill vendor'
#FRQueryToBackTranslate = 'aggregation sum sale condition sale = DUCKLING_TIME compare sale = DUCKLING_TIME'
#FRQueryToBackTranslate = 'aggregation sum sale groupby year condition customer = CUSTOMER_VALUE_LABEL sale > DUCKLING_CURRENCY'
#FRQueryToBackTranslate = 'select account_payable condition account_payable = DUCKLING_TIME account_payable < DUCKLING_CURRENCY vendor = VENDOR_VALUE_LABEL'
#FRQueryToBackTranslate = 'aggregation sum sale groupby customer condition sale = DUCKLING_TIME having sum sale > DUCKLING_CURRENCY'
#FRQueryToBackTranslate = 'aggregation sum sale avg sale groupby customer plot heatmap'
#FRQueryToBackTranslate = 'aggregation sum expense groupby product rowlimit -DUCKLING_AMOUNT'
#FRQueryToBackTranslate = 'aggregation avg sale groupby service rowlimit -1'
#FRQueryToBackTranslate = 'aggregation sum sale groupby customer rowlimit DUCKLING_AMOUNT'
#FRQueryToBackTranslate = 'aggregation avg sale groupby service rowlimit 1'
#FRQueryToBackTranslate = "aggregation sum sale.line_item groupby product having sum sale.line_item > DUCKLING_CURRENCY"
#FRQueryToBackTranslate = "aggregation sum customer.email groupby product"
#FRQueryToBackTranslate = "select account condition classification = expense"
#FRQueryToBackTranslate = "select reconciliation_details condition reconciliation_details = DUCKLING_TIME status != r reconciliation_details > DUCKLING_CURRENCY"
#FRQueryToBackTranslate = 'select purchase_order condition status = open purchase_order > DUCKLING_CURRENCY purchase_order = DUCKLING_TIME'
#FRQueryToBackTranslate = 'aggregation sum invoice groupby customer having sum invoice < 0'
#FRQueryToBackTranslate = 'aggregation sum profit condition profit = DUCKLING_TIME compare profit = DUCKLING_TIME'
#FRQueryToBackTranslate = 'aggregation sum invoice groupby customer having sum invoice < 0'
#FRQueryToBackTranslate = 'aggregation sum sale.line_item groupby product having sum sale.line_item > DUCKLING_CURRENCY'
#FRQueryToBackTranslate = 'aggregation sum sale.line_item.quantity condition item = ITEM_VALUE_LABEL'
#FRQueryToBackTranslate = 'aggregation sum sale compare sum expense condition OBJECT = DUCKLING_TIME'
#FRQueryToBackTranslate = 'help_identifier users'
#FRQueryToBackTranslate = 'select expense orderby expense.amount rowlimit 1'
#FRQueryToBackTranslate = 'aggregation sum bill groupby week condition due_date = DUCKLING_TIME'
#FRQueryToBackTranslate = 'aggregation sum profit groupby profit.item_name'
#FRQueryToBackTranslate = 'aggregation sum account_receivable groupby account'
#FRQueryToBackTranslate = 'select sale.line_item condition product = ITEM_VALUE_LABEL'
#FRQueryToBackTranslate = 'aggregation sum sale.line_item groupby customer condition service = ITEM_VALUE_LABEL having sum sale.line_item > DUCKLING_CURRENCY'
#FRQueryToBackTranslate = 'aggregation sum sale groupby customer condition service = ITEM_VALUE_LABEL sale = DUCKLING_TIME'
#FRQueryToBackTranslate = 'aggregation sum sale groupby country'
#FRQueryToBackTranslate = 'select reconciliation_details condition status = r reconciliation_details = DUCKLING_TIME reconciliation_details > DUCKLING_CURRENCY'
#FRQueryToBackTranslate = 'aggregation count invoice groupby month condition invoice = DUCKLING_TIME'
#FRQueryToBackTranslate = 'select item expense'
#FRQueryToBackTranslate = 'aggregation sum invoice condition balance = 0 customer = CUSTOMER_VALUE_LABEL invoice = DUCKLING_TIME'
#FRQueryToBackTranslate = 'select transaction'
#FRQueryToBackTranslate = 'select transaction condition account.classification = revenue'
#FRQueryToBackTranslate = 'select transaction condition account.classification = revenue transaction = DUCKLING_TIME'
#FRQueryToBackTranslate = 'aggregation avg expense compare avg sale condition OBJECT = DUCKLING_TIME'
#FRQueryToBackTranslate = 'select reconciliation_details orderby reconciliation_details.transaction_date.date condition status = r rowlimit 5'
#FRQueryToBackTranslate = 'select account_payable.line_item condition tracking_category = TRACKING_CATEGORY_VALUE_LABEL'
#FRQueryToBackTranslate = 'select sale.line_item condition item = ITEM_VALUE_LABEL sale.line_item > DUCKLING_CURRENCY sale = DUCKLING_TIME'
#FRQueryToBackTranslate = 'select sales_receipt.line_item groupby account'
#FRQueryToBackTranslate = 'select sale_tax condition sale_tax = DUCKLING_TIME'
#FRQueryToBackTranslate = 'aggregation avg sales_receipt groupby customer month condition sales_receipt = DUCKLING_TIME'
#FRQueryToBackTranslate = 'aggregation sum gross_profit subtract aggregation sum expense'
#FRQueryToBackTranslate = 'aggregation sum gross_profit divide aggregation sum sale'
#FRQueryToBackTranslate = 'aggregation avg account_payable groupby week condition account_payable = DUCKLING_TIME'
#FRQueryToBackTranslate = 'aggregation sum expense groupby vendor condition expense = DUCKLING_TIME divide aggregation sum expense condition expense = DUCKLING_TIME'
#FRQueryToBackTranslate = 'select sales orderby sale.due_date.date'
#FRQueryToBackTranslate = 'select bill.line_item condition tracking_category = TRACKING_CATEGORY_VALUE_LABEL'
#FRQueryToBackTranslate = 'select estimate.line_item condition customer = CUSTOMER_VALUE_LABEL estimate.txnstatus = pending item = ITEM_VALUE_LABEL estimate = DUCKLING_TIME estimate.line_item > DUCKLING_CURRENCY'
#FRQueryToBackTranslate = 'select estimate.line_item condition estimate.txnstatus = closed item = ITEM_VALUE_LABEL estimate = DUCKLING_TIME estimate.line_item > DUCKLING_CURRENCY'
#FRQueryToBackTranslate = 'select estimate condition customer = CUSTOMER_VALUE_LABEL estimate.txnstatus = rejected estimate < DUCKLING_CURRENCY estimate = DUCKLING_TIME'
#FRQueryToBackTranslate = 'aggregation sum estimate condition estimate.txnstatus = expired divide aggregation sum estimate'
#FRQueryToBackTranslate = 'select reconciliation_details condition status = c'
#FRQueryToBackTranslate = 'aggregation sum estimate.line_item groupby account fiscal_quarter condition estimate.txnstatus = pending'
#FRQueryToBackTranslate = "aggregation sum sale.line_item groupby month condition item = ITEM_VALUE_LABEL sale = DUCKLING_TIME"
#FRQueryToBackTranslate = "aggregation sum sale.line_item groupby sale.transaction_date.month condition item = ITEM_VALUE_LABEL sale = DUCKLING_TIME"
#FRQueryToBackTranslate = "aggregation sum invoice.amount groupby currency"
#FRQueryToBackTranslate = "aggregation sum invoice groupby month"
#FRQueryToBackTranslate = "aggregation sum invoice.balance groupby month"
#FRQueryToBackTranslate = "compare aggregation sum profit groupby item_name month condition profit = DUCKLING_TIME"
#FRQueryToBackTranslate = "aggregation sum sale groupby province"
#FRQueryToBackTranslate = "aggregation sum sale.line_item.quantity groupby month condition item = ITEM_VALUE_LABEL"
#FRQueryToBackTranslate = "aggregation sum sale.line_item groupby product province condition sale = DUCKLING_TIME plot heatmap"
#FRQueryToBackTranslate = 'aggregation sum budget.actual_summary.budget_amount compare sum budget.actual_summary.actual_amount groupby budget_name budget_parentid account.classification account condition account.classification = revenue budget.actual_summary.budget_month = DUCKLING_TIME'
#FRQueryToBackTranslate = 'compare aggregation sum sale.line_item groupby product month'
FRQueryToBackTranslate = 'aggregation sum account condition account.account_type_subclass = current__asset divide aggregation sum account condition account.account_type_subclass = current__liability'

#Stripe
#FRQueryToBackTranslate = 'aggregation avg application_fee condition amount_refunded < DUCKLING_CURRENCY account.charges_enabled = false'
#FRQueryToBackTranslate = 'aggregation avg application_fee groupby refunded hour_of_day condition account.charges_enabled = true application_fee < DUCKLING_CURRENCY application_fee = DUCKLING_TIME'
#FRQueryToBackTranslate = 'aggregation avg charge groupby application_fee customer condition status = failed application_fee.amount_refunded = DUCKLING_CURRENCY charge = DUCKLING_TIME'
#FRQueryToBackTranslate = 'select bank_account condition bank_account = ACCOUNT_VALUE_LABEL'
#FRQueryToBackTranslate = 'select account condition charges_enabled = false account.tos_acceptance.date = DUCKLING_TIME account.payout_schedule.weekly_anchor = saturday'
#FRQueryToBackTranslate = 'select account condition charges_enabled = true account.payout_schedule.delay_days = DUCKLING_AMOUNT'
#FRQueryToBackTranslate = 'select account condition charges_enabled = true type = custom account.legal_entity.verification.status = verified'
#FRQueryToBackTranslate = 'select balance_transaction condition type = advance_funding status = pending net < DUCKLING_CURRENCY'
#FRQueryToBackTranslate = 'aggregation avg application_fee groupby currency year condition amount_refunded < DUCKLING_CURRENCY balance_transaction > DUCKLING_CURRENCY refunded = true plot DEFAULT'
#FRQueryToBackTranslate = 'aggregation avg application_fee_refund groupby application_fee year condition application_fee_refund = DUCKLING_TIME'

#Property
#FRQueryToBackTranslate = "aggregation avg property groupby assessmentids"
#FRQueryToBackTranslate = "aggregation avg property condition priceperarea < DUCKLING_CURRENCY garagespaces = DUCKLING_AMOUNT"


DrawGraph = False

FRTestSource = 'Single_Query' #'data_prep'


class fr_to_ans(object):
    def __init__(self, RunningFolderPath=str(), NL_Entities_filename=str(), nl_entity_groupablebyCSV_filename=str(), nl_entity_filterablebyCSV_filename = str()):
        self.RunningFolderPath = RunningFolderPath
        self.NL_EntitiesCSV = self.RunningFolderPath + '/' + NL_Entities_filename
        self.nl_entity_groupablebyCSV = self.RunningFolderPath + '/' + nl_entity_groupablebyCSV_filename
        self.nl_entity_filterablebyCSV = self.RunningFolderPath + '/' + nl_entity_filterablebyCSV_filename

        self.oe_eng = nx.MultiDiGraph()  #nx.Graph() #nx.DiGraph()
        self.init_oe_eng()

        self.init_stacks_and_ans()

        self.fr_to_translate = str()
        self.fr_to_translate_list = list()
        self.fr_to_translate_full_path = str()
        self.fr_to_translate_list_full_path = list()

        self.len_fr_to_translate_list = 0
        self.fr_queries_to_translate = list()
        self.reverse_translation_error = ReverseTranslationError
        self.condition_label_type = condition_clause_label_types.NoCondition #the type of condition label for one condition at a time.
        self.reverse_translation_mode = ModesForReverseTranslation.Full
        self.reset_line_item_properties()


    def init_objects_and_attributes(self):
        df = pd.read_csv(self.NL_EntitiesCSV)
        df.columns = ['Type', 'ID', 'Parent_Entity', 'NL_Entity_Name', 'default_select', 'default_group_by', \
                      'default_aggregate', 'default_date', 'default_amount', 'default_currency', \
                      'class_name', 'condition', 'value', 'english_plural', 'english_singular', 'english_plural_short', \
                      'english_singular_short', 'english_plural_synonym1', 'english_singular_synonym1', 'permutor_include']
        for index, row in df.iterrows():
            type = row['Type']
            if type == 'A':
                node_type = 'OntologicalAttribute'
            elif type == 'O':
                node_type = 'OntologicalEntity'
            id = row['ID']

            parent_entity = row['Parent_Entity']
            try:
                if math.isnan(parent_entity): parent_entity = ''
            except:
                pass

            english_plural0 = row['english_plural']
            try:
                if math.isnan(english_plural0): english_plural0 = ''
            except: pass
            english_plural = list()
            english_plural.append(english_plural0)
            english_plural1 = row['english_plural_synonym1']
            try:
                if math.isnan(english_plural1): english_plural1 = ''
            except:
                pass
            english_plural.append(english_plural1)

            english_singular0 = row['english_singular']
            try:
                if math.isnan(english_singular0): english_singular0 = ''
            except: pass
            english_singular = list()
            english_singular.append(english_singular0)
            english_singular1 = row['english_singular_synonym1']
            try:
                if math.isnan(english_singular1): english_singular1 = ''
            except:
                pass
            english_singular.append(english_singular1)

            english_plural_short = row['english_plural_short']
            try:
                if math.isnan(english_plural_short): english_plural_short = ''
            except:
                pass

            english_singular_short = row['english_singular_short']
            try:
                if math.isnan(english_singular_short): english_singular_short = ''
            except:
                pass

            default_select = row['default_select']
            default_group_by = row['default_group_by']
            try:
                if math.isnan(default_group_by): default_group_by = ''
            except:
                pass

            default_aggregate = row['default_aggregate']
            try:
                if math.isnan(default_aggregate): default_aggregate = ''
            except:
                pass


            default_date = row['default_date']
            try:
                if math.isnan(default_date): default_date = ''
            except:
                pass

            default_amount = row['default_amount']
            try:
                if math.isnan(default_amount): default_amount = ''
            except:
                pass

            default_currency = row['default_currency']
            try:
                if math.isnan(default_currency): default_currency = ''
            except:
                pass

            class_name = row['class_name']
            try:
                if math.isnan(class_name): class_name = ''
            except:
                pass

            permutor_include = row['permutor_include']
            try:
                if math.isnan(parent_entity): permutor_include = ''
            except:
                pass


            self.oe_eng.add_node(id, NodeType=node_type, ParentEntity=parent_entity ,DefaultGroupBy=default_group_by, EnglishNameSingular=english_singular, \
                                 EnglishNamePlural=english_plural, \
                                 EnglishNameSingularShort=english_singular_short, EnglishNamePluralShort=english_plural_short, DefaultAggregate=default_aggregate, \
                                 DefaultDate=default_date, DefaultAmount=default_amount, DefaultCurrency=default_currency, ClassName=class_name, PermutorInclude=permutor_include)
            if len(parent_entity) > 0: #if type == 'A':
                self.oe_eng.add_edge(parent_entity, id, relationship='HasAttribute', DefaultSelect=default_select, DefaultGroupBy=default_group_by)


    """def init_joinable_edges(self):
        df = pd.read_csv(nl_entity_association_csvCSV)
        df.columns = ['id', 'nl_entity1_id', 'nl_entity2_id', 'predicate', 'context', 'description', 'domain_id']
        for index, row in df.iterrows():
            entity1 = row['nl_entity1_id']
            entity2 = row['nl_entity2_id']
            self.oe_eng.add_edge(entity1, entity2, relationship='Joinable')"""


    def init_groupableby_edges(self):
        df = pd.read_csv(self.nl_entity_groupablebyCSV)
        df.columns = ['nl_entity1_id', 'nl_entity2_id']
        for index, row in df.iterrows():
            entity1 = row['nl_entity1_id']
            entity2 = row['nl_entity2_id']
            self.oe_eng.add_edge(entity1, entity2, relationship='GroupableBy')


    def parse_enumeration_conditions(self, enumeration_conditions_raw):
        if enumeration_conditions_raw == '':
            return []
        else:
            string_to_parse = enumeration_conditions_raw[1:-1]
            list_to_return = list()
            string_to_list = string_to_parse.split(',')
            for token in string_to_list:
                list_to_return.append(token.strip())
            return list_to_return


    def init_filterableby_edges(self):
        df = pd.read_csv(self.nl_entity_filterablebyCSV)
        df.columns = ['nl_entity1_id', 'filterableby', 'EnumerationConditions', 'EnumerationConditionsEnglish']
        for index, row in df.iterrows():
            entity1 = row['nl_entity1_id']
            entity2 = row['filterableby']
            enumeration_conditions = row['EnumerationConditions']
            try:
                if math.isnan(enumeration_conditions): enumeration_conditions = ''
            except: pass
            enumeration_conditions_english = row['EnumerationConditionsEnglish']
            try:
                if math.isnan(enumeration_conditions_english): enumeration_conditions_english = ''
            except:
                pass
            self.oe_eng.add_edge(entity1, entity2, relationship=FilterableBy, \
                                 EnumerationConditions = self.parse_enumeration_conditions(enumeration_conditions), \
                                 EnumerationConditionsEnglish = self.parse_enumeration_conditions(enumeration_conditions_english))


    def init_oe_eng(self):
        self.init_objects_and_attributes()
        #self.init_joinable_edges() We will leave this one out for now and see how it goes.
        self.init_groupableby_edges()
        self.init_filterableby_edges()

        if DrawGraph == True:
            pos = nx.spring_layout(self.oe_eng)
            nx.draw(self.oe_eng, pos, with_labels=True, node_size=50, font_size=6, font_color='black', edge_color='gray')  #font_weight='bold'
            edge_labels = nx.get_edge_attributes(self.oe_eng, 'relationship')
            nx.draw_networkx_edge_labels(self.oe_eng, pos, labels=edge_labels, font_size=4.5)
            #plt.show()


    def reset_line_item_properties(self):
        pass
        """self.need_line_item = False  # line_item needs to be appended to teh OEs that need it in the relevant places.
        self.OE_entity_with_line_item_attribute_found = False
        self.line_item_satisfied = True  # if need_line_item is true, and this one is False, then an exception will be raised."""


    def init_stacks_and_ans(self):
        self.ans_string = ''
        self.fr_stack = {
            compare: list(),
            help_identifier: list(),
            select: list(),
            aggregation: list(),
            groupby: list(),
            orderby: list(),
            condition: list(),
            having: list(),
            rowlimit: list(),
            plot: list()
        }
        self.ans_stack = {
            compare: '',
            help_identifier: '',
            select: '',
            aggregation: '',
            groupby: '',
            orderby: '',
            condition: '',
            having: '',
            rowlimit: '',
            plot: ''
        }


    def get_fr_queries_to_translate(self):
        pass
        #prep = data_prep()
        #prep.get_eng_sql_lists()
        #prep.remove_reverse_duplicates()
        #self.fr_queries_to_translate = prep.fr_reverse_queries


    def index_in_fr_list(self, clause): #returns the index of the first entry, as well as the number of times it shows up
        if clause in self.fr_to_translate_list:
            return self.fr_to_translate_list.index(clause), self.fr_to_translate_list.count(clause)
        else:
            return -1, self.fr_to_translate_list.count(clause)


    def get_entity(self, token, in_condition_or_having_clause): #token is the total parent.child string
        try:
            token_to_append = random.sample(self.oe_eng.nodes[token]['EnglishNamePlural'],1)[0]
        except:
            token_to_append = ''
        entities_str = token_to_append.strip()
        return entities_str


    def compare_clause(self):
        i, nr_clauses = self.index_in_fr_list(compare)
        if nr_clauses > 1: raise Exception('Too many compare clauses')
        elif i == 0:
            self.fr_stack[compare].append(compare)
            self.ans_stack[compare] += ' ' + compare


    def help_identifier_clause(self):
        i, nr_clauses = self.index_in_fr_list(help_identifier)
        index_found = i
        if nr_clauses > 1: raise Exception('Too many help clauses')
        elif i == 0:
            j = 0
            self.ans_stack[help_identifier] += help_identifier_Eng[help_identifier]
            while i < self.len_fr_to_translate_list - 1 and self.fr_to_translate_list[i + 1] not in Clauses:
                i += 1
                j += 1
                if j > 1: self.ans_stack[help_identifier] += ' ' + 'and'
                value_label = self.fr_to_translate_list[i]
                self.fr_stack[help_identifier].append(value_label)
                self.ans_stack[help_identifier] += ' ' + value_label
            if i == index_found: raise Exception('No detail specified in the help_identifier clause')


    def select_clause(self):
        i, nr_clauses = self.index_in_fr_list(select)
        index_found = i
        if nr_clauses > 1: raise Exception('Too many select clauses')
        elif i >= 0:
            j = 0
            self.ans_stack[select] += ' all'
            while i < self.len_fr_to_translate_list - 1 and self.fr_to_translate_list[i + 1] not in Clauses:
                i += 1
                j += 1
                if j > 1: self.ans_stack[select] += ' ' + 'and'
                entity = self.fr_to_translate_list[i]
                self.fr_stack[select].append(entity)
                self.ans_stack[select] += ' ' + self.get_entity(entity, False)
                #if self.need_line_item and self.OE_entity_with_line_item_attribute_found == True and self.line_item_satisfied == False: raise Exception('aggregation/select clause need line_item for this query')
            if i == index_found: raise Exception('No entity specified in the select clause')
            if len(self.fr_stack[aggregation]) > 0: raise Exception('select clause added but aggregation clause already present')


    def aggregation_clause(self):
        i, nr_clauses = self.index_in_fr_list(aggregation)
        index_found = i
        if nr_clauses > 1: raise Exception('Too many aggregation clauses')
        elif i >= 0:
            j = 0
            while i < self.len_fr_to_translate_list - 1 and self.fr_to_translate_list[i + 1] not in Clauses:
                i += 1
                j += 1
                if j > 1:
                    if self.fr_to_translate_list[i] == compare:
                        self.fr_stack[aggregation].append(compare)
                        self.ans_stack[aggregation] += ' ' + 'compared to'
                        i += 1
                    elif self.fr_to_translate_list[i] == contrast:
                        self.fr_stack[aggregation].append(contrast)
                        self.ans_stack[aggregation] += ' ' + 'contrasted with'
                        i += 1
                    else:
                        self.ans_stack[aggregation] += ' ' + 'and'
                aggopp = self.fr_to_translate_list[i]
                self.fr_stack[aggregation].append(aggopp)
                self.ans_stack[aggregation] += ' ' + random.sample(AggOpsEng[aggopp],1)[0]
                i += 1
                entity = self.fr_to_translate_list[i]
                self.fr_stack[aggregation].append(entity)
                #if len(entity.split('.')) == 1:
                #    default_aggregate_entity = self.
                self.ans_stack[aggregation] += ' ' + self.get_entity(entity, False)
                #if self.need_line_item and self.OE_entity_with_line_item_attribute_found == True and self.line_item_satisfied == False: raise Exception('aggregation/select clause need line_item for this query')
            if i == index_found: raise Exception('No entity specified in the aggregation clause')
            if len(self.fr_stack[select]) > 0: raise Exception('aggregation clause added but select clause already present')


    def check_for_line_item(self): #will check for line item in all the places where it needs to be
        pass


    def groupby_clause_old(self):
        i, nr_clauses = self.index_in_fr_list(groupby)
        index_found = i
        if nr_clauses > 1: raise Exception('Too many groupby clauses')
        elif i >= 0:
            nl_entities_only = list()
            for token in self.fr_stack[aggregation]:
                if (token not in AggOpsEng) and (token != compare) and (token != contrast):
                    nl_entities_only.append(token)
            while i < self.len_fr_to_translate_list - 1 and self.fr_to_translate_list[i + 1] not in Clauses:
                i += 1
                self.ans_stack[groupby] += ' ' + 'by'
                entity = self.fr_to_translate_list[i]
                entity_english_name = ''
                for aggregation_entity in nl_entities_only:
                    valid_groupable_found = False
                    default_groupable_found = False
                    parent_node = str()

                    for neighbour in self.oe_eng.neighbors(aggregation_entity):
                        applicable_path = neighbour
                        leaf_node = neighbour.split('.')[-1]
                        immediate_parent = '.'.join(neighbour.split('.')[:-1])
                        if default_groupable_found == False:
                            edges = self.oe_eng.get_edge_data(aggregation_entity, neighbour)
                            if edges != None:
                                for edge_key in edges:
                                    if edges[edge_key]['relationship'] == 'GroupableBy':
                                        neighbour_is_default_groupable = False
                                        neighbour_is_default_date = False
                                        for incoming_node in self.oe_eng.predecessors(neighbour):
                                            incoming_edges = self.oe_eng.get_edge_data(incoming_node, neighbour)
                                            for incoming_edge_key in incoming_edges:
                                                if incoming_edges[incoming_edge_key]['relationship'] == 'HasAttribute':
                                                    if incoming_edges[incoming_edge_key]['DefaultGroupBy'] == True:
                                                        neighbour_is_default_groupable = True
                                                        parent_node = incoming_node
                                        if neighbour_is_default_groupable == False:
                                            if applicable_path.split('.')[-1] in ['hour_of_day', 'day_of_month', 'week', 'month', 'fiscal_quarter', 'year']:
                                                applicable_path = '.'.join(applicable_path.split('.')[:-1]) + '.date'
                                            path_split = applicable_path.split('.')
                                            len_path_split = len(path_split)
                                            looked_up_value = self.oe_eng.nodes['.'.join(path_split[:-1])]['DefaultDate']
                                            while (len_path_split > 1) and ((applicable_path == looked_up_value) or neighbour == looked_up_value):
                                                applicable_path = '.'.join(path_split[:-1])
                                                if applicable_path == aggregation_entity and looked_up_value == self.oe_eng.nodes[aggregation_entity]['DefaultDate']: #when the groupable found is in the default date object
                                                    neighbour_is_default_date = True
                                                path_split = applicable_path.split('.')
                                                len_path_split = len(path_split)
                                                if len_path_split > 1:
                                                    looked_up_value = self.oe_eng.nodes['.'.join(path_split[:-1])]['DefaultDate']
                                                else:
                                                    looked_up_value = ''  # in this case the loop should end


                                        if (neighbour == entity) or (neighbour_is_default_groupable == True and parent_node == entity) or \
                                                ((applicable_path == aggregation_entity) and (leaf_node == entity)) or \
                                                ((immediate_parent == aggregation_entity) and (leaf_node == entity)): # or (neighbour.split('.')[-1] == entity):
                                            valid_groupable_found = True
                                            if neighbour_is_default_date and (self.reverse_translation_mode != ModesForReverseTranslation.Full):
                                                entity_english_name = self.oe_eng.nodes[neighbour]['EnglishNameSingularShort']
                                            else:
                                                entity_english_name = random.sample(self.oe_eng.nodes[neighbour]['EnglishNameSingular'], 1)[0]
                                            if parent_node == entity: default_groupable_found = True


                    if valid_groupable_found == False: raise Exception('No GroupableBy edge for groupby for `' + aggregation_entity + '` to groupby `' + entity + '`')

                self.fr_stack[groupby].append(entity)
                self.ans_stack[groupby] += ' ' + entity_english_name #self.get_groupby_entity(entity)
            if i == index_found: raise Exception('No entity specified in the groupby clause')





    def orderby_clause(self):
        i, nr_clauses = self.index_in_fr_list(orderby)
        index_found = i
        if nr_clauses > 1: raise Exception('Too many orderby clauses')
        elif i >= 0:
            while i < self.len_fr_to_translate_list - 1 and self.fr_to_translate_list[i + 1] not in Clauses:
                i += 1
                self.ans_stack[orderby] += ' ' + 'ordered by'
                entity = self.fr_to_translate_list[i]
                self.fr_stack[orderby].append(entity)
                self.ans_stack[orderby] += ' ' + self.get_entity(entity, False)
            if i == index_found: raise Exception('No entity specified in the orderby clause')


    """def label_is_valid(self, label):
        if DucklingSubString in label:
            self.condition_label_type = condition_clause_label_types.Duckling
            return True
        if ValueLabelSubstring in label:
            self.condition_label_type = condition_clause_label_types.ValueLabel
            return True
        for string in LabelsEng:
            if string == label:
                self.condition_label_type = condition_clause_label_types.SpecifiedLabel
                return True
        try:
            int(label)
            self.condition_label_type = condition_clause_label_types.Numeric
            return True
        except:
            return False
        return False"""


    def condition_clause_old(self):
        i, nr_clauses = self.index_in_fr_list(condition)
        index_found = i
        fr_entity_i = i #this is used to help generate the new full path fr needed.
        if nr_clauses > 1: raise Exception('Too many condition clauses')
        elif i >= 0:
            nl_entities_only = list()
            list_for_query = list()
            node_in_stack = str()
            if self.fr_stack[aggregation] == []:
                list_for_query = self.fr_stack[select]
                node_in_stack = select
            else:
                list_for_query = self.fr_stack[aggregation]
                node_in_stack = aggregation

            for token in list_for_query:
                if (token not in AggOpsEng) and (token != compare) and (token != contrast):
                    nl_entities_only.append(token)
            j = 0
            while i < self.len_fr_to_translate_list - 1 and self.fr_to_translate_list[i + 1] not in Clauses:
                i += 1
                j += 1
                has_greater_less_opp = False
                if self.fr_to_translate_list[i] == compare:
                    self.ans_stack[condition] += ' ' + 'compared to'
                    i += 2
                else:
                    fr_entity = self.fr_to_translate_list[i]
                    fr_entity_i = i

                    self.fr_stack[condition].append(fr_entity)
                    if 'due_date' in fr_entity:
                        self.ans_stack[condition] += ' ' + 'due'
                    else:
                        entity = self.get_entity(fr_entity, True)
                    i += 1
                    opp = self.fr_to_translate_list[i]
                    if opp in GreaterLessOps:  has_greater_less_opp = True
                    self.fr_stack[condition].append(opp)
                    if not ((opp == EqualTo) and (self.reverse_translation_mode == ModesForReverseTranslation.Shortened)):
                        self.ans_stack[condition] += ' ' + random.sample(Ops_Eng[opp], 1)[0]

                i += 1
                label = self.fr_to_translate_list[i]
                label_english = str() #label #initialisation
                #if label in LabelsEng:
                #    label = LabelsEng[label]
                #    self.condition_label_type = condition_clause_label_types.SpecifiedLabel
                #elif not self.label_is_valid(label): raise Exception('Condition clause label is not valid')

                for agg_select_entity in nl_entities_only:
                    label = self.fr_to_translate_list[i]
                    valid_filterable_found = False
                    exact_match_neighbour_found = False
                    if fr_entity == OBJECT: applicable_fr_entity = agg_select_entity
                    else: applicable_fr_entity = fr_entity
                    for neighbour in self.oe_eng.neighbors(agg_select_entity):
                        if exact_match_neighbour_found == False:
                            edges = self.oe_eng.get_edge_data(agg_select_entity, neighbour)
                            if edges != None:
                                for edge_key in edges:
                                    if edges[edge_key]['relationship'] == FilterableBy:
                                        applicable_path = neighbour
                                        applicable_path2 = str() #initialisation
                                        leaf_node = str()
                                        path_split = applicable_path.split('.')
                                        neighbour_is_default_date = False
                                        neighbour_is_default_currency = False
                                        VALUE_LABEL_found = False
                                        parent_entity = path_split[0]
                                        if label in DucklingsDefaultKeys:
                                            applicable_key = DucklingsDefaultKeys[label]
                                            len_path_split = len(path_split)
                                            looked_up_value = self.oe_eng.nodes['.'.join(path_split[:-1])][applicable_key]
                                            while (len_path_split > 1) and ((applicable_path == looked_up_value) or neighbour == looked_up_value):
                                                applicable_path = '.'.join(path_split[:-1])
                                                path_split = applicable_path.split('.')
                                                len_path_split = len(path_split)
                                                if len_path_split > 1 : looked_up_value = self.oe_eng.nodes['.'.join(path_split[:-1])][applicable_key]
                                                else: looked_up_value = '' #in this case the loop should end
                                            if applicable_path == agg_select_entity:
                                                if applicable_key == 'DefaultDate':
                                                    neighbour_is_default_date = True
                                                elif applicable_key == 'DefaultCurrency':
                                                    neighbour_is_default_currency = True
                                        elif 'VALUE_LABEL' in label:
                                            VALUE_LABEL_found = True
                                            if len(path_split) > 1:
                                                applicable_path = '.'.join(path_split[:-1])
                                            else:
                                                applicable_path = path_split[0]
                                            applicable_path2 = path_split[-1]
                                        else:
                                            if label[0] == "'" and label[-1] == "'":
                                                applicable_path = parent_entity
                                            else:
                                                applicable_path = path_split[-1]
                                            """if len(path_split) == 1: value_label_entity = path_split[0]
                                            else: value_label_entity = path_split[-2]
                                            parent_entity = path_split[0]
                                            #if 'VALUE_LABEL' in label: # == value_label_entity.upper() + '_VALUE_LABEL':
                                            applicable_path = parent_entity
                                            #leaf_node = path_split[-1]"""

                                        if applicable_path.split('.')[-1] == applicable_fr_entity or applicable_path == applicable_fr_entity \
                                                or neighbour == applicable_fr_entity or leaf_node == applicable_fr_entity or applicable_path2 == applicable_fr_entity:
                                            if neighbour == applicable_fr_entity: exact_match_neighbour_found = True
                                            self.fr_to_translate_list_full_path[fr_entity_i] = neighbour
                                            if label in edges[edge_key]['EnumerationConditions']:
                                                valid_filterable_found = True
                                                label_english = str()
                                                english_enum_for_filter = edges[edge_key]['EnumerationConditionsEnglish']
                                                if ('.'.join(neighbour.split('.')[:-1]) == agg_select_entity) and (self.reverse_translation_mode != ModesForReverseTranslation.Full) and \
                                                    (not utilities.substring_in_list(DucklingSubString, english_enum_for_filter)) and \
                                                     (not utilities.substring_in_list(ValueLabelSubstring, english_enum_for_filter)) and \
                                                      (not utilities.substring_in_list(EnumStr_true, english_enum_for_filter)) and \
                                                       (not utilities.substring_in_list(EnumStr_false, english_enum_for_filter)): #if label is an API enum
                                                    match_index = 0
                                                    for main_eng_entity_looked_for in self.oe_eng.nodes[agg_select_entity]['EnglishNamePlural']:
                                                        an_index = self.ans_stack[node_in_stack].find(main_eng_entity_looked_for)
                                                        if an_index >= 0: match_index = an_index

                                                    self.ans_stack[node_in_stack] = self.ans_stack[node_in_stack][:match_index] + edges[edge_key]['EnumerationConditionsEnglish'][edges[edge_key]['EnumerationConditions'].index(label)] + \
                                                                                    ' ' + self.ans_stack[node_in_stack][match_index:]
                                                    if self.reverse_translation_mode == ModesForReverseTranslation.ShortenedEnumInFrontFieldInBack:
                                                        label_english = applicable_path.split('.')[-1]
                                                    exact_match_neighbour_found = True #since if this state is found, then no need to look for other matches for the neighbour
                                                else:
                                                    if ((neighbour_is_default_date == False) and (neighbour_is_default_currency == False) and (VALUE_LABEL_found == False)) \
                                                            or (self.reverse_translation_mode == ModesForReverseTranslation.Full): #not default curr/date, not full
                                                        label_english = random.sample(self.oe_eng.nodes[neighbour]['EnglishNameSingular'], 1)[0] + ' of '
                                                    label_english += edges[edge_key]['EnumerationConditionsEnglish'][edges[edge_key]['EnumerationConditions'].index(label)]
                                                    exact_match_neighbour_found = True
                                            else: # 'CONSTANT' in edges[edge_key]['EnumerationConditions']:
                                                valid_filterable_found = True
                                                if label == "''": label = 'nothing'
                                                label_english = applicable_path.split('.')[-1] + ' of ' + str(label)
                                            #else:
                                            #    pass
                                                #raise Exception('Label not in filterable lits of label options as per edge enumeration.')
                    if valid_filterable_found == False: raise Exception(
                            'No FilterableBy edge for filtering for `' + agg_select_entity + '` to filter by `' + applicable_fr_entity + '`, or filterable node not truncated, or VL wrong.')

                """if (self.condition_label_type != condition_clause_label_types.SpecifiedLabel) and (self.condition_label_type != condition_clause_label_types.Numeric) and\
                        (self.condition_label_type != condition_clause_label_types.Duckling) and \
                        (fr_entity != ConditionLabels[label][0]): raise Exception('Non Duckling label and entity in condition clause not the same type')"""
                #if self.need_line_item and self.OE_entity_with_line_item_attribute_found == True and self.line_item_satisfied == False and label != DUCKLING_TIME and \
                #    fr_entity != 'estimate.txnstatus' : raise Exception('condition clause need line_item for this query')
                #if self.need_line_item and self.OE_entity_with_line_item_attribute_found == True and self.line_item_satisfied == True and label == DUCKLING_TIME: raise Exception( 'condition clause should not have line_item for DUCKLING_TIME for this query')
                #if has_greater_less_opp and (self.condition_label_type != condition_clause_label_types.Numeric and self.condition_label_type != condition_clause_label_types.Duckling):
                #    raise Exception('Cannot have < or > if label is not a Duckling or a number')
                self.fr_stack[condition].append(label)
                self.ans_stack[condition] += ' ' + label_english
            if i == index_found: raise Exception('No entity specified in the condition clause')

    #groupby clause new -----------------------------------
    def look_for_default_date_groupable_in_an_entity(self, aggregation_entity, neighbour, entity, valid_groupable_found, neighbour_is_default_date):
        possible_detault_date = self.oe_eng.nodes[aggregation_entity]['DefaultDate']
        if possible_detault_date != '':
            new_default_date = str()
            while possible_detault_date != '':
                new_default_date = possible_detault_date
                possible_detault_date = self.oe_eng.nodes[possible_detault_date]['DefaultDate']
            if len(new_default_date) > 0:
                default_date_common_path = '.'.join(new_default_date.split('.')[:-1])
                for neighbour in self.oe_eng.nodes():
                    if default_date_common_path in neighbour:
                        leaf_node = neighbour.split('.')[-1]
                        if (entity == neighbour) or (entity == leaf_node):
                            valid_groupable_found = True
                            neighbour_is_default_date = True
                    if valid_groupable_found: break
        return neighbour, valid_groupable_found, neighbour_is_default_date


    def search_entity_for_aggregation_entity(self, aggregation_entity, neighbour, entity, valid_groupable_found, neighbour_is_default_date):
        if not valid_groupable_found:  # look on same level
            aggregation_entity_path_split = aggregation_entity.split('.')
            if len(aggregation_entity_path_split) > 1:
                parent_aggregation_entity = '.'.join(aggregation_entity_path_split[:-1])
                for outgoing_node in self.oe_eng.successors(parent_aggregation_entity):
                    outgoing_edges = self.oe_eng.get_edge_data(parent_aggregation_entity, outgoing_node)
                    for outgoing_edge_key in outgoing_edges:
                        if outgoing_edges[outgoing_edge_key]['relationship'] == 'HasAttribute':
                            neighbour = outgoing_node
                            leaf_node = outgoing_node.split('.')[-1]
                            if (entity == neighbour) or (entity == leaf_node):
                                valid_groupable_found = True
                                break
                    if valid_groupable_found: break

        if not valid_groupable_found:  # look on the child level for default groupable
            for outgoing_node in self.oe_eng.successors(aggregation_entity):
                outgoing_edges = self.oe_eng.get_edge_data(aggregation_entity, outgoing_node)
                for outgoing_edge_key in outgoing_edges:
                    if outgoing_edges[outgoing_edge_key]['relationship'] == 'HasAttribute':
                        if self.oe_eng.nodes[outgoing_node]['DefaultGroupBy']:
                            neighbour = outgoing_node
                            leaf_node = neighbour.split('.')[-1]
                            if (entity == neighbour) or (entity == leaf_node):
                                valid_groupable_found = True
                    if valid_groupable_found: break
                if valid_groupable_found: break

        if not valid_groupable_found:  # look in the same entity for default date:
            neighbour, valid_groupable_found, neighbour_is_default_date = \
                self.look_for_default_date_groupable_in_an_entity(aggregation_entity, neighbour, entity, \
                                                                  valid_groupable_found, neighbour_is_default_date)

        if not valid_groupable_found:  # look on the child level for any entity that might fit
            for outgoing_node in self.oe_eng.successors(aggregation_entity):
                outgoing_edges = self.oe_eng.get_edge_data(aggregation_entity, outgoing_node)
                for outgoing_edge_key in outgoing_edges:
                    if outgoing_edges[outgoing_edge_key]['relationship'] == 'HasAttribute':
                        neighbour = outgoing_node
                        leaf_node = neighbour.split('.')[-1]
                        if (entity == neighbour) or (entity == leaf_node):
                            valid_groupable_found = True
                    if valid_groupable_found: break
                if valid_groupable_found: break

        if not valid_groupable_found:  # look on same level, for default dates
            aggregation_entity_path_split = aggregation_entity.split('.')
            if len(aggregation_entity_path_split) > 1:
                parent_aggregation_entity = '.'.join(aggregation_entity_path_split[:-1])
                neighbour, valid_groupable_found, neighbour_is_default_date = \
                    self.look_for_default_date_groupable_in_an_entity(parent_aggregation_entity, neighbour, entity, \
                                                                      valid_groupable_found,
                                                                      neighbour_is_default_date)

        if not valid_groupable_found:  # look on the 2nd child level for any entity that might fit
            for outgoing_node in self.oe_eng.successors(aggregation_entity):
                outgoing_edges = self.oe_eng.get_edge_data(aggregation_entity, outgoing_node)
                for outgoing_edge_key in outgoing_edges:
                    if outgoing_edges[outgoing_edge_key]['relationship'] == 'HasAttribute':
                        for outgoing_node2 in self.oe_eng.successors(outgoing_node):
                            outgoing_edges2 = self.oe_eng.get_edge_data(outgoing_node, outgoing_node2)
                            for outgoing_edge_key2 in outgoing_edges2:
                                if outgoing_edges2[outgoing_edge_key2]['relationship'] == 'HasAttribute':
                                    neighbour = outgoing_node2
                                    leaf_node = neighbour.split('.')[-1]
                                    if (entity == neighbour) or (entity == leaf_node):
                                        valid_groupable_found = True
                                if valid_groupable_found: break
                            if valid_groupable_found: break
                        if valid_groupable_found: break
                    if valid_groupable_found: break
                if valid_groupable_found: break

        return neighbour, valid_groupable_found, neighbour_is_default_date


    def groupby_clause(self): #New groupby clause ------------------------------------------------------------------------------------------
        i, nr_clauses = self.index_in_fr_list(groupby)
        index_found = i
        if nr_clauses > 1: raise Exception('Too many groupby clauses')
        elif i >= 0:
            nl_entities_only = list()
            for token in self.fr_stack[aggregation]:
                if (token not in AggOpsEng) and (token != compare) and (token != contrast):
                    nl_entities_only.append(token)
            while i < self.len_fr_to_translate_list - 1 and self.fr_to_translate_list[i + 1] not in Clauses:
                i += 1
                self.ans_stack[groupby] += ' ' + 'by'
                entity = self.fr_to_translate_list[i]
                entity_english_name = ''

                for aggregation_entity in nl_entities_only:
                    valid_groupable_found = False
                    default_groupable_found = False
                    parent_node = str()
                    neighbour_is_default_groupable = False
                    neighbour_is_default_date = False
                    neighbour = str()

                    for neighbour in self.oe_eng.nodes():  # look in root of tree
                        if neighbour == entity:
                            valid_groupable_found = True
                            break

                    neighbour, valid_groupable_found, neighbour_is_default_date = self.search_entity_for_aggregation_entity(aggregation_entity, neighbour, entity, \
                                                         valid_groupable_found, neighbour_is_default_date)

                    if not valid_groupable_found:
                        aggregation_entity_path_split = aggregation_entity.split('.')
                        if len(aggregation_entity_path_split) > 1:
                            parent_aggregation_entity = '.'.join(aggregation_entity_path_split[:-1])
                            neighbour, valid_groupable_found, neighbour_is_default_date = self.search_entity_for_aggregation_entity(
                                parent_aggregation_entity, neighbour, entity, \
                                valid_groupable_found, neighbour_is_default_date)
                            if not valid_groupable_found:
                                if len(aggregation_entity_path_split) > 2:
                                    parent_aggregation_entity = '.'.join(aggregation_entity_path_split[:-2])
                                    neighbour, valid_groupable_found, neighbour_is_default_date = self.search_entity_for_aggregation_entity(
                                        parent_aggregation_entity, neighbour, entity, \
                                        valid_groupable_found, neighbour_is_default_date)

                    if valid_groupable_found:
                        self.fr_to_translate_list_full_path[i] = neighbour
                        if neighbour_is_default_date and (self.reverse_translation_mode != ModesForReverseTranslation.Full):
                            entity_english_name = self.oe_eng.nodes[neighbour]['EnglishNameSingularShort']
                        else:
                            entity_english_name = random.sample(self.oe_eng.nodes[neighbour]['EnglishNameSingular'], 1)[0]
                        #if parent_node == entity: default_groupable_found = True

                    if valid_groupable_found == False: raise Exception('No group by node found for for `' + aggregation_entity + '` to groupby `' + entity + '`')

                self.fr_stack[groupby].append(entity)
                self.ans_stack[groupby] += ' ' + entity_english_name #self.get_groupby_entity(entity)
            if i == index_found: raise Exception('No entity specified in the groupby clause')


    #New condition clause ------------------------------------------------------------------------------------------------------------------------------------------
    def look_for_default_date_filterable_in_an_entity(self, aggregation_entity, neighbour, entity, valid_filterable_found, neighbour_is_default_date):
        possible_detault_date = self.oe_eng.nodes[aggregation_entity]['DefaultDate']
        if possible_detault_date != '':
            new_default_date = str()
            while possible_detault_date != '':
                new_default_date = possible_detault_date
                possible_detault_date = self.oe_eng.nodes[possible_detault_date]['DefaultDate']
            if len(new_default_date) > 0:
                default_date_common_path = '.'.join(new_default_date.split('.')[:-1])
                for neighbour in self.oe_eng.nodes():
                    if default_date_common_path in neighbour:
                        leaf_node = neighbour.split('.')[-1]
                        if (entity == neighbour) or (entity == leaf_node):
                            valid_filterable_found = True
                            neighbour_is_default_date = True
                    if valid_filterable_found: break
        return neighbour, valid_filterable_found, neighbour_is_default_date


    def look_for_default_currency_filterable_in_an_entity(self, aggregation_entity, neighbour, entity, valid_filterable_found, neighbour_is_default_currency):
        possible_default_currency = self.oe_eng.nodes[aggregation_entity]['DefaultCurrency']
        if possible_default_currency != '':
            new_default_currency = str()
            while possible_default_currency != '':
                new_default_currency = possible_default_currency
                possible_default_currency = self.oe_eng.nodes[possible_default_currency]['DefaultCurrency']
            if len(new_default_currency) > 0:
                default_date_common_path = '.'.join(new_default_currency.split('.')[:-1])
                for neighbour in self.oe_eng.nodes():
                    if neighbour in new_default_currency:
                        leaf_node = neighbour.split('.')[-1]
                        if (entity == neighbour) or (entity == leaf_node):
                            valid_filterable_found = True
                            neighbour_is_default_currency = True
                    if valid_filterable_found: break
        return neighbour, valid_filterable_found, neighbour_is_default_currency


    def search_applicable_fr_entity_for_agg_select_entity(self, agg_select_entity, neighbour, applicable_fr_entity, \
                        valid_filterable_found, neighbour_is_default_date, neighbour_is_default_currency):

        if not valid_filterable_found:  # look on same level
            aggregation_entity_path_split = agg_select_entity.split('.')
            if len(aggregation_entity_path_split) > 1:
                parent_aggregation_entity = '.'.join(aggregation_entity_path_split[:-1])
                for outgoing_node in self.oe_eng.successors(parent_aggregation_entity):
                    outgoing_edges = self.oe_eng.get_edge_data(parent_aggregation_entity, outgoing_node)
                    for outgoing_edge_key in outgoing_edges:
                        if outgoing_edges[outgoing_edge_key]['relationship'] == 'HasAttribute':
                            neighbour = outgoing_node
                            leaf_node = outgoing_node.split('.')[-1]
                            if (applicable_fr_entity == neighbour) or (applicable_fr_entity == leaf_node):
                                valid_filterable_found = True
                                break
                    if valid_filterable_found: break


        if not valid_filterable_found:  # look in the same entity for default date:
            neighbour, valid_filterable_found, neighbour_is_default_date = \
                self.look_for_default_date_filterable_in_an_entity(agg_select_entity, neighbour, applicable_fr_entity, \
                                                                  valid_filterable_found, neighbour_is_default_date)

        if not valid_filterable_found:  # look in the same entity for default currency:
            neighbour, valid_filterable_found, neighbour_is_default_currency = \
                self.look_for_default_currency_filterable_in_an_entity(agg_select_entity, neighbour, applicable_fr_entity, \
                                                                  valid_filterable_found, neighbour_is_default_currency)

        if not valid_filterable_found:  # look on the child level for any entity that might fit
            for outgoing_node in self.oe_eng.successors(agg_select_entity):
                outgoing_edges = self.oe_eng.get_edge_data(agg_select_entity, outgoing_node)
                for outgoing_edge_key in outgoing_edges:
                    if outgoing_edges[outgoing_edge_key]['relationship'] == 'HasAttribute':
                        neighbour = outgoing_node
                        leaf_node = neighbour.split('.')[-1]
                        if (applicable_fr_entity == neighbour) or (applicable_fr_entity == leaf_node):
                            valid_filterable_found = True
                    if valid_filterable_found: break
                if valid_filterable_found: break

        if not valid_filterable_found:  # look on same level, for default dates
            aggregation_entity_path_split = agg_select_entity.split('.')
            if len(aggregation_entity_path_split) > 1:
                parent_aggregation_entity = '.'.join(aggregation_entity_path_split[:-1])
                neighbour, valid_filterable_found, neighbour_is_default_date = \
                    self.look_for_default_date_filterable_in_an_entity(parent_aggregation_entity, neighbour, applicable_fr_entity, \
                                                                       valid_filterable_found,
                                                                      neighbour_is_default_date)

        if not valid_filterable_found:  # look on same level, for default currency
            aggregation_entity_path_split = agg_select_entity.split('.')
            if len(aggregation_entity_path_split) > 1:
                parent_aggregation_entity = '.'.join(aggregation_entity_path_split[:-1])
                neighbour, valid_filterable_found, neighbour_is_default_currency = \
                    self.look_for_default_currency_filterable_in_an_entity(parent_aggregation_entity, neighbour, applicable_fr_entity, \
                                                                       valid_filterable_found,
                                                                       neighbour_is_default_currency)

        if not valid_filterable_found:  # look on the 2nd child level for any entity that might fit
            for outgoing_node in self.oe_eng.successors(agg_select_entity):
                outgoing_edges = self.oe_eng.get_edge_data(agg_select_entity, outgoing_node)
                for outgoing_edge_key in outgoing_edges:
                    if outgoing_edges[outgoing_edge_key]['relationship'] == 'HasAttribute':
                        for outgoing_node2 in self.oe_eng.successors(outgoing_node):
                            outgoing_edges2 = self.oe_eng.get_edge_data(outgoing_node, outgoing_node2)
                            for outgoing_edge_key2 in outgoing_edges2:
                                if outgoing_edges2[outgoing_edge_key2]['relationship'] == 'HasAttribute':
                                    neighbour = outgoing_node2
                                    leaf_node = neighbour.split('.')[-1]
                                    if (applicable_fr_entity == neighbour) or (applicable_fr_entity == leaf_node):
                                        valid_filterable_found = True
                                if valid_filterable_found: break
                            if valid_filterable_found: break
                        if valid_filterable_found: break
                    if valid_filterable_found: break
                if valid_filterable_found: break

        return neighbour, valid_filterable_found, neighbour_is_default_date, neighbour_is_default_currency


    def condition_clause(self): #New condition clause
        i, nr_clauses = self.index_in_fr_list(condition)
        index_found = i
        fr_entity_i = i #this is used to help generate the new full path fr needed.
        if nr_clauses > 1: raise Exception('Too many condition clauses')
        elif i >= 0:
            nl_entities_only = list()
            list_for_query = list()
            node_in_stack = str()
            if self.fr_stack[aggregation] == []:
                list_for_query = self.fr_stack[select]
                node_in_stack = select
            else:
                list_for_query = self.fr_stack[aggregation]
                node_in_stack = aggregation

            for token in list_for_query:
                if (token not in AggOpsEng) and (token != compare) and (token != contrast):
                    nl_entities_only.append(token)
            j = 0
            while i < self.len_fr_to_translate_list - 1 and self.fr_to_translate_list[i + 1] not in Clauses:
                i += 1
                j += 1
                has_greater_less_opp = False
                if self.fr_to_translate_list[i] == compare:
                    self.ans_stack[condition] += ' ' + 'compared to'
                    i += 2
                else:
                    fr_entity = self.fr_to_translate_list[i]
                    fr_entity_i = i

                    self.fr_stack[condition].append(fr_entity)
                    if 'due_date' in fr_entity:
                        self.ans_stack[condition] += ' ' + 'due'
                    else:
                        entity = self.get_entity(fr_entity, True)
                    i += 1
                    opp = self.fr_to_translate_list[i]
                    if opp in GreaterLessOps:  has_greater_less_opp = True
                    self.fr_stack[condition].append(opp)
                    if not ((opp == EqualTo) and (self.reverse_translation_mode == ModesForReverseTranslation.Shortened)):
                        self.ans_stack[condition] += ' ' + random.sample(Ops_Eng[opp], 1)[0]

                i += 1
                label = self.fr_to_translate_list[i]
                label_english = str() #label #initialisation

                for agg_select_entity in nl_entities_only:
                    label = self.fr_to_translate_list[i]
                    valid_filterable_found = False
                    exact_match_neighbour_found = False
                    neighbour_is_default_date = False
                    neighbour_is_default_currency = False
                    neighbour = str()
                    VALUE_LABEL_found = False
                    if fr_entity == OBJECT: applicable_fr_entity = agg_select_entity
                    else: applicable_fr_entity = fr_entity

                    for neighbour in self.oe_eng.nodes():  # look in root of tree
                        if neighbour == applicable_fr_entity:
                            valid_filterable_found = True
                            exact_match_neighbour_found = True
                            if self.oe_eng.nodes[neighbour]['DefaultCurrency'] != '':
                                neighbour_is_default_currency = True
                            if self.oe_eng.nodes[neighbour]['DefaultDate'] != '':
                                neighbour_is_default_date = True
                            break

                    if not valid_filterable_found:
                        neighbour, valid_filterable_found, neighbour_is_default_date, neighbour_is_default_currency = self.search_applicable_fr_entity_for_agg_select_entity(
                            agg_select_entity, neighbour, applicable_fr_entity, \
                            valid_filterable_found, neighbour_is_default_date, neighbour_is_default_currency)

                    if not valid_filterable_found:
                        agg_select_entity_path_split = agg_select_entity.split('.')
                        if len(agg_select_entity_path_split) > 1:
                            parent_agg_select_entity_entity = '.'.join(agg_select_entity_path_split[:-1])
                            neighbour, valid_filterable_found, neighbour_is_default_date, neighbour_is_default_currency = self.search_applicable_fr_entity_for_agg_select_entity(
                                parent_agg_select_entity_entity, neighbour, applicable_fr_entity, \
                                valid_filterable_found, neighbour_is_default_date, neighbour_is_default_currency)
                            if not valid_filterable_found:
                                if len(agg_select_entity_path_split) > 2:
                                    parent_agg_select_entity_entity = '.'.join(agg_select_entity_path_split[:-2])
                                    neighbour, valid_filterable_found, neighbour_is_default_date, neighbour_is_default_currency = self.search_applicable_fr_entity_for_agg_select_entity(
                                        parent_agg_select_entity_entity, neighbour, applicable_fr_entity, \
                                        valid_filterable_found, neighbour_is_default_date, neighbour_is_default_currency)

                    if label in DucklingsDefaultKeys:
                        applicable_key = DucklingsDefaultKeys[label]
                        if neighbour_is_default_date and (applicable_key == 'DefaultDate'):
                            neighbour_is_default_date = True
                        elif neighbour_is_default_currency and (applicable_key == 'DefaultCurrency'):
                            neighbour_is_default_currency = True
                        else:
                            neighbour_is_default_date = False
                            neighbour_is_default_currency = False
                    elif 'VALUE_LABEL' in label:
                        VALUE_LABEL_found = True

                    if valid_filterable_found:
                        self.fr_to_translate_list_full_path[fr_entity_i] = neighbour
                        edges = self.oe_eng.get_edge_data(agg_select_entity, neighbour)
                        if edges != None:
                            for edge_key in edges:
                                if edges[edge_key]['relationship'] == FilterableBy:

                                    if label in edges[edge_key]['EnumerationConditions']:
                                        valid_filterable_found = True
                                        label_english = str()
                                        english_enum_for_filter = edges[edge_key]['EnumerationConditionsEnglish']
                                        if ('.'.join(neighbour.split('.')[:-1]) == agg_select_entity) and (self.reverse_translation_mode != ModesForReverseTranslation.Full) and \
                                            (not utilities.substring_in_list(DucklingSubString, english_enum_for_filter)) and \
                                                    (not utilities.substring_in_list(ValueLabelSubstring, english_enum_for_filter)) and \
                                                    (not utilities.substring_in_list(EnumStr_true, english_enum_for_filter)) and \
                                                    (not utilities.substring_in_list(EnumStr_false, english_enum_for_filter)): #if label is an API enum
                                            match_index = 0
                                            for main_eng_entity_looked_for in self.oe_eng.nodes[agg_select_entity]['EnglishNamePlural']:
                                                an_index = self.ans_stack[node_in_stack].find(main_eng_entity_looked_for)
                                                if an_index >= 0: match_index = an_index

                                            self.ans_stack[node_in_stack] = self.ans_stack[node_in_stack][:match_index] + edges[edge_key]['EnumerationConditionsEnglish'][edges[edge_key]['EnumerationConditions'].index(label)] + \
                                                                                ' ' + self.ans_stack[node_in_stack][match_index:]
                                            if self.reverse_translation_mode == ModesForReverseTranslation.ShortenedEnumInFrontFieldInBack:
                                                label_english = neighbour.split('.')[-1]
                                        else:
                                            if ((neighbour_is_default_date == False) and (neighbour_is_default_currency == False) and (VALUE_LABEL_found == False)) \
                                                    or (self.reverse_translation_mode == ModesForReverseTranslation.Full): #not default curr/date, not full
                                                label_english = random.sample(self.oe_eng.nodes[neighbour]['EnglishNameSingular'], 1)[0] + ' of '
                                            label_english += edges[edge_key]['EnumerationConditionsEnglish'][edges[edge_key]['EnumerationConditions'].index(label)]
                                            exact_match_neighbour_found = True
                                    else:
                                        if label == "''": label = 'nothing'
                                        label_english = neighbour.split('.')[-1] + ' of ' + str(label)
                                else:
                                    if label_english == '':
                                        if label == "''": label = 'nothing'
                                        if ((neighbour_is_default_date == False) and (neighbour_is_default_currency == False) and (VALUE_LABEL_found == False)) \
                                                or (self.reverse_translation_mode == ModesForReverseTranslation.Full):  # not default curr/date, not full
                                            label_english = self.oe_eng.nodes[neighbour]['EnglishNameSingular'][0] + ' of '
                                        label_english += str(label)
                        else:
                            if label == "''": label = 'nothing'
                            if ((neighbour_is_default_date == False) and (neighbour_is_default_currency == False) and (VALUE_LABEL_found == False)) or (self.reverse_translation_mode == ModesForReverseTranslation.Full):  # not default curr/date, not full
                                label_english = self.oe_eng.nodes[neighbour]['EnglishNameSingular'][0] + ' of '
                            label_english += str(label)

                    else: raise Exception(
                            'No FilterableBy edge for filtering for `' + agg_select_entity + '` to filter by `' + applicable_fr_entity + '`, or filterable node not truncated, or VL wrong.')

                self.fr_stack[condition].append(label)
                self.ans_stack[condition] += ' ' + label_english
            if i == index_found: raise Exception('No entity specified in the condition clause')


    def having_clause(self):
        i, nr_clauses = self.index_in_fr_list(having)
        index_found = i
        if nr_clauses > 1: raise Exception('Too many having clauses')
        elif i >= 0:
            j = 0
            while i < self.len_fr_to_translate_list - 1 and self.fr_to_translate_list[i + 1] not in Clauses:
                self.ans_stack[having] += ' ' + 'having'
                i += 1
                j += 1
                aggopp = self.fr_to_translate_list[i]
                self.fr_stack[having].append(aggopp)
                self.ans_stack[having] += ' ' + random.sample(AggOpsEng[aggopp],1)[0]
                i += 1
                entity = self.fr_to_translate_list[i]
                self.fr_stack[having].append(entity)
                self.ans_stack[having] += ' ' + self.get_entity(entity, True)
                i += 1
                opp = self.fr_to_translate_list[i]
                self.fr_stack[having].append(opp)
                self.ans_stack[having] += ' ' + Ops_Eng_Having[opp]
                i += 1
                label = self.fr_to_translate_list[i]
                self.fr_stack[having].append(label)
                self.ans_stack[having] += ' ' + label
                #if self.need_line_item and self.OE_entity_with_line_item_attribute_found == True and self.line_item_satisfied == False: raise Exception('aggregation/select clause need line_item for this query')
            if i == index_found: raise Exception('No entity specified in the having clause')


    def plot_clause(self):
        i, nr_clauses = self.index_in_fr_list(plot)
        index_found = i
        if nr_clauses > 1: raise Exception('Too many plot clauses')
        elif i >= 0:
            while i < self.len_fr_to_translate_list - 1 and self.fr_to_translate_list[i + 1] not in Clauses:
                i += 1
                self.ans_stack[plot] += ' ' + 'in a'
                plot_type = self.fr_to_translate_list[i]
                self.fr_stack[plot].append(plot_type)
                plot_string = str()
                if self.reverse_translation_mode == ModesForReverseTranslation.Full:
                    plot_string = PlotLabelsEng[plot_type][0]
                else:
                    plot_string = random.sample(PlotLabelsEng[plot_type], 1)[0]
                self.ans_stack[plot] += ' ' + plot_string
            if i == index_found: raise Exception('No plot type specified in the plot clause')


    def rowlimit_clause(self):
        i, nr_clauses = self.index_in_fr_list(rowlimit)
        index_found = i
        if nr_clauses > 1: raise Exception('Too many rowlimit clauses')
        elif i >= 0:
            j = 0
            while i < self.len_fr_to_translate_list - 1 and self.fr_to_translate_list[i + 1] not in Clauses:
                i += 1
                j += 1
                if j > 1: raise Exception('More than one number supplied in rowlimit clause')
                number = self.fr_to_translate_list[i]
                self.fr_stack[rowlimit].append(number)
                token = ''
                start_index = 0
                if number[0] == '-':
                    if self.reverse_translation_mode == ModesForReverseTranslation.Shortened: token = random.sample(BottomSynonyms, 1)[0]
                    else: token = BottomSynonyms[0]
                    start_index = 1
                else:
                    if self.reverse_translation_mode == ModesForReverseTranslation.Shortened: token = random.sample(TopSynonyms, 1)[0]
                    else: token = TopSynonyms[0]
                self.ans_stack[rowlimit] += token
                actual_number = number[start_index:]
                if actual_number != '1':
                    self.ans_stack[rowlimit] += ' ' + actual_number
            if i == index_found: raise Exception('No number specified in the rowlimit clause')


    def gen_ans_string(self):
        self.ans_string = self.ans_stack[rowlimit] + \
                            self.ans_stack[compare] + \
                            self.ans_stack[help_identifier] + \
                            self.ans_stack[select] + \
                            self.ans_stack[aggregation] + \
                            self.ans_stack[groupby] + \
                            self.ans_stack[orderby] + \
                            self.ans_stack[condition] + \
                            self.ans_stack[having] + \
                            self.ans_stack[plot]
        self.ans_string = self.ans_string.strip()
        self.ans_string = ' '.join(self.ans_string.split())


    def translate_one_query(self, fr_to_translate): #method will return a tuple: (translated_query, original_FR_readied for QE)
        try:
            self.init_stacks_and_ans()
            self.fr_to_translate = fr_to_translate
            self.fr_to_translate_list = self.fr_to_translate.split()
            self.fr_to_translate_list_full_path = self.fr_to_translate_list #this is an ititial init, entities will be swopped out for the full entity at the right points.
            self.len_fr_to_translate_list = len(self.fr_to_translate_list)
            #print('fr_to_translate_list:', self.fr_to_translate_list)


            self.compare_clause()
            self.help_identifier_clause()
            self.select_clause()
            self.aggregation_clause()

            self.orderby_clause()
            self.groupby_clause()  # we run this first to see if line_item is to be sorted out or not
            self.condition_clause()

            self.having_clause()
            self.rowlimit_clause()
            self.plot_clause()

            if len(self.fr_stack[select]) == 0 and len(self.fr_stack[aggregation]) == 0 and len(self.fr_stack[help_identifier]) == 0: raise Exception('Need a select clause or an aggregation clause')
            if len(self.fr_stack[compare]) > 0 and len(self.fr_stack[groupby]) == 0: raise Exception('Change (compare) queries need a groupby clause')

            #print('fr_stack:', self.fr_stack)
            #print('ans_stack:', self.ans_stack)
            self.gen_ans_string()
            self.fr_to_translate_full_path = (' '.join(self.fr_to_translate_list_full_path)).strip()
        except Exception as inst:
            print(inst)
            self.ans_string = self.reverse_translation_error #something went wrong in the reverse translation.
        self.reset_line_item_properties()
        return self.ans_string


    def translate(self, fr_to_translate):
        #EquationOperations_Eng
        #ReverseTranslationError
        try:
            sub_query = str()
            token_list = (fr_to_translate.strip() + ' ' + EOS).split()
            total_query_translated = str()
            len_token_list = len(token_list)
            for i in range(len_token_list):
                token = token_list[i]
                if (token in EquationOperations_Eng.keys()) or (token == EOS):
                    sub_query = sub_query.strip()
                    new_tanslation = self.translate_one_query(sub_query)
                    sub_query = ''
                    if new_tanslation == ReverseTranslationError:
                        raise Exception("Problem in reverse translation")
                    else:
                        if token == EOS: end_of_new_translation = ''
                        else: end_of_new_translation = ' ' + EquationOperations_Eng[token] + ' '
                        total_query_translated += new_tanslation + end_of_new_translation
                else:
                    sub_query += ' ' + token

            return total_query_translated.strip()

        except:
            return ReverseTranslationError


    def translate_and_paraphrase(self, fr_to_translate):
        output_list = list()
        for mode in ModesForReverseTranslation:
            self.reverse_translation_mode = mode
            rev_translate_para = self.translate(fr_to_translate)
            if rev_translate_para not in output_list:
                output_list.append(rev_translate_para)
        return output_list, self.fr_to_translate_full_path


    def translate_list(self):
        i = 0
        if FRTestSource == 'data_prep':
            self.get_fr_queries_to_translate()
        elif FRTestSource == 'Single_Query':
            self.fr_queries_to_translate = [FRQueryToBackTranslate]
        for fr_query in self.fr_queries_to_translate:
            print(' ')
            print(i,':')
            print(fr_query)
            output = self.translate(fr_query)
            print(output)
            i += 1


    def translate_and_paraphrase_list(self):
        i = 0
        if FRTestSource == 'data_prep':
            self.get_fr_queries_to_translate()
        elif FRTestSource == 'Single_Query':
            self.fr_queries_to_translate = [FRQueryToBackTranslate]
        for fr_query in self.fr_queries_to_translate:
            print(' ')
            print(i, ':')
            print('Original FR:')
            print(fr_query)
            output, full_path_fr = self.translate_and_paraphrase(fr_query)
            print('Full path FR:')
            print(full_path_fr)
            print('Reverse translations and paraphrases:')
            for output_translation_paraphrase in output:
                print(output_translation_paraphrase)
            i += 1


if __name__ == "__main__":
    #Stripe
    """ans_generator = fr_to_ans(RunningFolderPath='/stripe-neural-parser',
                              NL_Entities_filename='stripe_NL-Entities_mod.csv',
                              nl_entity_groupablebyCSV_filename='stripe_nl_entity_groupableby.csv',
                              nl_entity_filterablebyCSV_filename='stripe_nl_entity_filterableby.csv')"""

    #QBO/Xero
    ans_generator = fr_to_ans(RunningFolderPath='/rnn_runtime',
                              NL_Entities_filename='accounting_NL-Entities_mod_2.csv',
                              nl_entity_groupablebyCSV_filename='accounting_nl_entity_groupableby.csv',
                              nl_entity_filterablebyCSV_filename='accounting_nl_entity_filterableby.csv')

    #Property
    """ans_generator = fr_to_ans(RunningFolderPath='/projects/chata_ai/property_neural_parser',
                              NL_Entities_filename='property_NL_Entities_mod.csv',
                              nl_entity_groupablebyCSV_filename='property_nl_entity_groupableby.csv',
                              nl_entity_filterablebyCSV_filename='property_nl_entity_filterableby.csv')"""


    ans_generator.translate_and_paraphrase_list()