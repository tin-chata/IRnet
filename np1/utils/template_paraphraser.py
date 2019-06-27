#J. Henning Viljoen Chata Technologies Inc. 2018

import pandas as pd
import random

from . import gen_ans_from_fr


VALUE_LABEL = 'VALUE_LABEL'
PlaceHolderTokens = ['ENTITY', 'ENTITY_PLURAL', VALUE_LABEL, 'FILTERABLE']
FRTokensNotToBeInParaphrase = ['FILTERABLE_TIME', 'FILTERABLE']
PLURAL = 'PLURAL'


class template_paraphraser(object):
    def __init__(self, mechanical_reverse, RunningFolderPath=str(), template_paraphraserCSV=str(), ParticularPatternLookedFor=str()):
        self.RunningFolderPath = RunningFolderPath
        self.template_paraphraserCSV = self.RunningFolderPath + '/' + template_paraphraserCSV
        self.ParticularPatternLookedFor = ParticularPatternLookedFor

        self.template_data_structure = dict()
        self.set_up_template()
        self.distinct_paraphrases()
        self.mechanical_reverse = mechanical_reverse
        self.nr_times_particular_pattern_hit = 0 #Tracker when testing to see how many times a particular pattern was hit by the permtutor
        pass


    def set_up_template(self):
        df = pd.read_csv(self.template_paraphraserCSV)
        df.columns = ['formal_representation', 'english_paraphrase']
        for index, row in df.iterrows():
            formal_representation = row['formal_representation']
            english_paraphrase = row['english_paraphrase']
            if formal_representation in self.template_data_structure:
                self.template_data_structure[formal_representation].append(english_paraphrase)
            else:
                self.template_data_structure[formal_representation] = list()
                self.template_data_structure[formal_representation].append(english_paraphrase)


    def distinct_paraphrases(self):
        for fr_key in self.template_data_structure:
            self.template_data_structure[fr_key] = list(set(self.template_data_structure[fr_key]))


    def extract_numeral_at_end_of_token(self, token):
        numeral = int()
        try:
            numeral = int(token[-1])
        except:
            numeral = -1
        return numeral


    def extract_numeral_for_token_list(self, token_list):
        new_token_list = list()
        for token in token_list:
            numeral = self.extract_numeral_at_end_of_token(token)
            if numeral == -1:
                new_token = [token, numeral]
            else:
                new_token = [token[:-1], numeral]
            new_token_list.append(new_token)
        return new_token_list


    def diff_between_strings(self, s0, s1):
        s0_split = s0.strip().split(' ')
        s1_split = s1.strip().split(' ')
        s0_split = self.extract_numeral_for_token_list(s0_split)
        s1_split = self.extract_numeral_for_token_list(s1_split)
        s0_unique_words = list()
        s1_unique_words = list()
        s0len = len(s0_split)
        s1len = len(s1_split)

        longest_length = max(s0len, s1len)
        for i in range(longest_length):
            if i >= s0len:
                s1_unique_words.append(s1_split[i])
            elif i >= s1len:
                s0_unique_words.append(s0_split[i])
            else:
                if s0_split[i] != s1_split[i]:
                    s0_unique_words.append(s0_split[i])
                    s1_unique_words.append(s1_split[i])
        return [s0_unique_words, s1_unique_words]


    def shorten_diff_matrix(self, diffs): #if FR tokens that should be in the paraphrase, are in the diffs for a matched FR query, they are deleted here
        new_diffs = [[], []]
        for i in range(len(diffs[1])):
            fr_token = diffs[0][i]
            template_fr_token = diffs[1][i]
            if (template_fr_token[0] not in FRTokensNotToBeInParaphrase):
                new_diffs[0].append(fr_token)
                new_diffs[1].append(template_fr_token)
        return new_diffs


    def sort_fr_dif_list_according_to_numerals(self, fr_dif_list, entities_with_numerals):
        list_of_numerals = list()
        for entity in entities_with_numerals:
            numeral = self.extract_numeral_at_end_of_token(entity)
            if numeral != -1:
                list_of_numerals.append(numeral)
            else:
                return fr_dif_list
        sorted_fr_dif_list = [fr_dif_list[i] for i in list_of_numerals]
        return sorted_fr_dif_list


    def return_index_with_nr(self, token_list, nr_wanted):
        i = 0
        for token in token_list:
            if token[1] == nr_wanted:
                return i
            i += 1
        return -1


    def consolidate_diff_matrix(self, diffs):
        new_diffs = [[], []]
        entity_numbers_consolidated = list()
        for i in range(len(diffs[1])):
            fr_template_diff_nr = diffs[1][i][1]
            if fr_template_diff_nr not in entity_numbers_consolidated:
                new_diffs[1].append(diffs[1][i])
                new_diffs[0].append(diffs[0][i])
                entity_numbers_consolidated.append(fr_template_diff_nr)
            else:
                previously_added_index = self.return_index_with_nr(new_diffs[1], fr_template_diff_nr)
                if new_diffs[0][previously_added_index][0] !=   diffs[0][i][0]:
                    return []
        return new_diffs


    def process_paraphrase(self, split_paraphrase):
        new_split_paraphrase = list()
        nr_valid_tokens = 0
        for token in split_paraphrase:
            i = 0
            new_token = ''
            valid_token_found = False
            while (i < len(PlaceHolderTokens)) and (valid_token_found == False):
                valid_token = PlaceHolderTokens[i]
                if valid_token in token:
                    valid_token_found = True
                i += 1
            if valid_token_found:
                numeral = self.extract_numeral_at_end_of_token(token)
                if numeral == -1:
                    new_token = [token, nr_valid_tokens]
                else:
                    new_token = [token[:-1], numeral]
                nr_valid_tokens += 1
            else:
                new_token = [token]
            new_split_paraphrase.append(new_token)

        return new_split_paraphrase


    def add_numerals_to_token_list(self, token_list):
        i = 0
        new_token_list = list()
        for token in token_list:
            new_token = [token, i]
            new_token_list.append(new_token)
            i += 1
        return new_token_list


    def find_index_of_entity_in_fr_template_diff(self, diffs, entity_nr):
        i = 0
        index_found = -1
        for token in diffs[1]:
            if token[1] == entity_nr:
                index_found = i
            i += 1
        return index_found


    def replace_place_holders_in_paraphrase(self, diffs, english_paraphrase):
        len_diffs = len(diffs[1])
        split_paraphrase = english_paraphrase.strip().split()
        processed_paraphrase_list = list()
        processed_paraphrase_str = str()
        diff_index = 0
        nr_replacements = 0
        nr_to_be_replaced = len(diffs[1])
        fr_dif_list = diffs[0]
        fr_template_dif_list = diffs[1]
        split_paraphrase = self.process_paraphrase(split_paraphrase)
        for token in split_paraphrase:
            if len(token) > 1:
                entity_nr = token[1]
                index_found = self.find_index_of_entity_in_fr_template_diff(diffs, entity_nr)
                if index_found == -1: return ''
                fr_token = fr_dif_list[index_found][0]
                english_token = str()
                if VALUE_LABEL in fr_token:
                    english_token = fr_token
                elif PLURAL in token[0]:
                    try:
                        english_token = random.sample(self.mechanical_reverse.oe_eng.nodes[fr_token]['EnglishNamePlural'], 1)[0]
                    except:
                        english_token = fr_token
                else:
                    try:
                        english_token = random.sample(self.mechanical_reverse.oe_eng.nodes[fr_token]['EnglishNameSingular'], 1)[0]
                    except:
                        english_token = fr_token
                processed_paraphrase_list.append(english_token)
                nr_replacements += 1
                if diff_index < len_diffs - 1:
                    diff_index += 1
            else:
                processed_paraphrase_list.append(token[0])
        #if nr_replacements == nr_to_be_replaced:
        processed_paraphrase_str = ' '.join(processed_paraphrase_list)
        return processed_paraphrase_str


    def paraphrase_formal_representation(self, formal_representation):
        try:
            #print('Paraphrasing: "' + formal_representation + '"')
            for fr_in_data_structure in self.template_data_structure:
                diffs = self.diff_between_strings(formal_representation, fr_in_data_structure)
                #print(diffs)
                fr_matches_data_template_entry = True
                if (len(diffs[0]) == len(diffs[1])):
                    i = 0
                    for token in diffs[1]:
                        if token[0] not in PlaceHolderTokens:
                            fr_matches_data_template_entry = False
                        elif VALUE_LABEL in token[0]:
                            if VALUE_LABEL not in diffs[0][i][0]:
                                fr_matches_data_template_entry = False
                        i += 1
                else:
                    fr_matches_data_template_entry = False
                if fr_matches_data_template_entry:
                    print('FR template matched in template_paraphraser:', fr_in_data_structure)
                    if fr_in_data_structure == self.ParticularPatternLookedFor: self.nr_times_particular_pattern_hit += 1
                    diffs = self.shorten_diff_matrix(diffs)
                    diffs = self.consolidate_diff_matrix(diffs)
                    print(diffs)
                    if diffs == []: return []
                    paraphrase_list = self.template_data_structure[fr_in_data_structure]
                    processed_paraphrase_list = list()
                    #print(paraphrase_list)
                    for english_paraphrase in paraphrase_list:
                        processed_paraphrase = self.replace_place_holders_in_paraphrase(diffs, english_paraphrase)
                        if len(processed_paraphrase) > 0: processed_paraphrase_list.append(processed_paraphrase)
                    processed_paraphrase_list = list(set(processed_paraphrase_list))
                    processed_paraphrase_list.sort()
                    #for paraphrase in processed_paraphrase_list: print(paraphrase)
                    return processed_paraphrase_list
        except:
            return []


def main():
    #Stripe
    mechanical_reverse = gen_ans_from_fr.fr_to_ans(RunningFolderPath='/stripe-neural-parser',
                                                   NL_Entities_filename='stripe_NL-Entities_mod.csv',
                                                   nl_entity_groupablebyCSV_filename='stripe_nl_entity_groupableby.csv',
                                                   nl_entity_filterablebyCSV_filename='stripe_nl_entity_filterableby.csv')
    a_template_paraphraser = template_paraphraser(mechanical_reverse, RunningFolderPath='/stripe-neural-parser',
                                                  template_paraphraserCSV='template_paraphraser.csv',
                                                  ParticularPatternLookedFor='aggregation count ENTITY0')

    #Property
    """mechanical_reverse = gen_ans_from_fr.fr_to_ans(RunningFolderPath='/projects/chata_ai/property_neural_parser',
                                                   NL_Entities_filename='property_NL_Entities_mod.csv',
                                                   nl_entity_groupablebyCSV_filename='property_nl_entity_groupableby.csv',
                                                   nl_entity_filterablebyCSV_filename='property_nl_entity_filterableby.csv')
    a_template_paraphraser = template_paraphraser(mechanical_reverse, RunningFolderPath='/projects/chata_ai/property_neural_parser',
                                                  template_paraphraserCSV='property_template_paraphraser.csv',
                                                  ParticularPatternLookedFor='aggregation avg property groupby assessmentids')"""


    #para_list = a_template_paraphraser.paraphrase_formal_representation('aggregation sum charge groupby day')
    #para_list = a_template_paraphraser.paraphrase_formal_representation('select charge condition charge = DUCKLING_TIME')
    #para_list = a_template_paraphraser.paraphrase_formal_representation('select charge')
    #para_list = a_template_paraphraser.paraphrase_formal_representation('aggregation sum invoice groupby customer month')
    #para_list = a_template_paraphraser.paraphrase_formal_representation('aggregation sum invoice groupby customer month condition invoice = DUCKLING_TIME')
    #para_list = a_template_paraphraser.paraphrase_formal_representation('select invoice condition customer = CUSTOMER_VALUE_LABEL')
    #para_list = a_template_paraphraser.paraphrase_formal_representation('aggregation sum invoice groupby month condition invoice = DUCKLING_TIME')
    #para_list = a_template_paraphraser.paraphrase_formal_representation('aggregation sum invoice condition invoice = DUCKLING_TIME')
    #para_list = a_template_paraphraser.paraphrase_formal_representation('aggregation sum dispute')
    #para_list = a_template_paraphraser.paraphrase_formal_representation('aggregation sum invoice groupby customer rowlimit 1')
    #para_list = a_template_paraphraser.paraphrase_formal_representation('aggregation avg invoice groupby customer')
    para_list = a_template_paraphraser.paraphrase_formal_representation('aggregation avg application_fee_refund groupby application_fee year condition application_fee_refund = DUCKLING_TIME')

    nr_paraphrases = 0
    if para_list != None:
        for par in para_list:
            print(par)
            nr_paraphrases += 1
    print()
    print('Nr paraphrases:', nr_paraphrases)



if __name__ == "__main__":
    main()
