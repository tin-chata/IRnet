#J. Henning Viljoen.  Chata Technologies.

"""This is a module to house various semi- or fully stand-alone functions that can be called to perform various tasks
that might be needed by the other modules in the application"""

import random

def sample_from_discrete_cum(list_for_sample, cum_prob_dist):
    """This function will sample from a given list according to the given cumulative probability distribution.
    Typically the first probability category, is the prob that none of the items in the list is selected.  So
    The len of cum_prob_dist should be one more than that of list_for_sample.
    Returns: a list containing either nothing, or the element chosen"""

    cum_prob_dist_internal = [0.0] + cum_prob_dist + [1.0]
    r = random.random()
    for bracket in range(len(cum_prob_dist_internal) - 1):
        if r >= cum_prob_dist_internal[bracket] and r <= cum_prob_dist_internal[bracket + 1]:
            if bracket == 0:
                return []
            else:
                return [list_for_sample[bracket - 1]]
    return [] #Default in case of an error - should never run


def sample_from_discrete_cum_nr_of_times_distinct(list_for_sample, cum_prob_dist, nr_distinct_samples):
    """This function will sample nr_distinct_samples from the list according to cum_prob_dist.  All samples have to be
    different, so if samples repeat, they will sampled again until different, or the nr of samples is more than the
    length of the list"""
    len_list_for_sample = len(list_for_sample)
    historical_samples_list = list()
    nr_samples_drawn = 0
    sample = 0
    for i in range(nr_distinct_samples):
        j = 0
        sample = sample_from_discrete_cum(list_for_sample, cum_prob_dist)[0]
        while (sample in historical_samples_list and len(historical_samples_list) < len_list_for_sample):
            sample = sample_from_discrete_cum(list_for_sample, cum_prob_dist)[0]
        historical_samples_list.append(sample)
    return historical_samples_list


def sample_from_ontological_entities_nr_of_times_distinct(prob_dist_oe, nr_distinct_samples):
    """This function will sample nr_distinct_samples from the dict prob_dist_oe,  This distribution is not cumulative."""
    list_for_sample = list()
    cum_prob_dist = list()
    cummalitive_prob = 0.0
    for key in prob_dist_oe.keys():
        list_for_sample.append(key)
        cum_prob_dist.append(cummalitive_prob)
        cummalitive_prob += prob_dist_oe[key][1]
    return sample_from_discrete_cum_nr_of_times_distinct(list_for_sample, cum_prob_dist, nr_distinct_samples)


def substring_in_list(substring, list_of_strings):
    for entry in list_of_strings:
        if substring in entry: return True
    return False