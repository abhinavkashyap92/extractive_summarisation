import numpy as np
def facility_function(vocab_indices, summary_indices, sentence_similarity, config):
    """
     This is the facility function that is described in the paper
     Graph-Based Submodular selection for extractive Summarization
    Args:
        vocab_indices: list
            Indices of the vocabulary usually (0, num_sentences)
            This is not necessary for facility function
            as it iterates over the entire vocab to calculate the value
            It is included for completeness
        summary_indices: list
            Indices of the sentences that are in summary
            sentence_similarity: ndarray
        config: dictionary:
            If there is any config that needs to be used can be passed
        The similarity matrix between all sentences

    Returns: The facility function value - A float number
    """

    # A VECTORISED IMPLEMENTATION OF THE FACILITY FUNCTION
    # THE SPECIAL CASE OF SUMMARY BEING 0 HAS TO BE HANDLED SPECIALLY
    if len(summary_indices) == 0:
        facility_function_value = 0.0
    else:
        facility_function_value = np.sum(np.max(sentence_similarity[:,summary_indices],
                                                axis=1))
    return facility_function_value

def penalty_function(vocab_indices, summary_indices, sentence_similarity, config):
    """
        This is the penalty function that is described in the paper
        Graph-Based Submodular selection for extractive Summarization
    Args:
        vocab_indices: list
        summary_indices: list
        sentence_similarity: ndarray
        config: dictionary
        Some of the methods require some hyper parameters
        to be set

        This penalises redundancy
    Returns: The value of the graph cut function
    """
    penalty_lambda = config["penalty_lambda"]
    sentence_similartiy_ = np.copy(sentence_similarity)
    np.fill_diagonal(sentence_similartiy_, 0.0)

    if len(summary_indices) == 0:
        fn_value = 0.0
    else:
        v_not_in_s = list(set(vocab_indices) - set(summary_indices))
        rows = v_not_in_s
        cols = summary_indices
        # USING THE ADVANCED INDEXING OF THE NUMPY ARRAY
        fn_value = np.sum(sentence_similarity[np.ix_(rows, cols)]) - \
                   penalty_lambda * np.sum(sentence_similartiy_[np.ix_(summary_indices, summary_indices)])

    return fn_value

def graph_cut_function(vocab_indices, summary_indices, sentence_similarity, config):
    """
    Args:
        vocab_indices: list
            Indices of the sentences
        summary_indices: list
            Indices of the sentences in the summary
        sentence_similarity: list
            similarity matrix between sentences
        config: dict
            If there is any configuration that is required to implement
         this function then it is based a dict
    Returns: float
    """
    if len(summary_indices) ==0:
        fn_value = 0.0
    else:
        v_not_in_s = list(set(vocab_indices) - set(summary_indices))
        rows = v_not_in_s
        cols = summary_indices
        fn_value = np.sum(sentence_similarity[np.ix_(rows, cols)])

    return fn_value

def coverage_function(vocab_indices, summary_indices, sentence_similarity, config):
    """
    This is the coverage function that is described in the paper
    A CLASS OF SUB MODULAR FUNCTIONS FOR DOCUMENT SUMMARIZATION
    L(S) = \sum_{i \in V} min(C_{i}(S), \alpha C_{i}(V))
    Paste the formula in latexit and you can see it
    The C that is used here is simple summation that compares
    the similarity between sentence i and S or V accordingly
    Args:
        vocab_indices: list

        summary_indices:
        sentence_similarity:
        config:

    Returns: float - The function value
    """
    saturation_alpha = config["saturation_alpha"]
    if len(summary_indices) == 0:
        fn_value = 0
    else:
        rows = vocab_indices
        a = saturation_alpha * np.sum(sentence_similarity, axis=1)
        x = np.sum(sentence_similarity[np.ix_(rows,summary_indices)], axis=1)
        fn_value = np.sum(np.minimum(x, a))

    return fn_value

def diversity_function(vocab_indices, summary_indices, sentence_similarity, config):
    """
    This is the coverage function that is described in the paper
    A CLASS OF SUB MODULAR FUNCTIONS FOR DOCUMENT SUMMARIZATION
    L(S) = \sum_{i \in V} min(C_{i}(S), \alpha C_{i}(V))
    Paste the formula in latexit and you can see it
    The C that is used here is simple summation that compares
    the similarity between sentence i and S or V accordingly
    Args:
        vocab_indices: list
        summary_indices: list
        sentence_similarity: ndarray
        config:
        membership
        The ground set is divided into various clusters
        The membership of these clusters is given here
        A list of lists [[1, 2], [3]]
        First cluster contains 1,2
        second cluster contains 3

    Returns: float - The function value
    """

    fn_value = None
    membership = config['membership']
    K = len(membership)
    N, N = sentence_similarity.shape
    if len(summary_indices) == 0:
        fn_value = 0.0

    else:
        fn_value = 0.0
        for k in xrange(K):
            s_intersection_pk = list(set(membership[k]).intersection(set(summary_indices)))
            fn_value += np.sqrt((1. / N) * np.sum(sentence_similarity[s_intersection_pk,:]))

    return fn_value

def coverage_diversity(vocab_indices, summary_indices, sentence_similarity, config):
    """
    Combining the coverage and diversity function according to the paper
    A CLASS OF SUB MODULAR FUNCTIONS FOR DOCUMENT SUMMARIZATION
    Args:
        vocab_indices: list
        summary_indices: list
        similarity_matrix: ndarray
        config:
        saturation_alpha - required for coverage function
        membership - required for diversity function
        reg: required to regularise the coverage with diversity
    Returns: float_value
    """
    reg = config['reg']
    fn_value = coverage_function(vocab_indices, summary_indices, sentence_similarity, config) + \
               reg * diversity_function(vocab_indices, summary_indices, sentence_similarity, config)
    return fn_value
