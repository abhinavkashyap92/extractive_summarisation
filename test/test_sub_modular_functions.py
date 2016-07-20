import random
import numpy as np
import pytest
from epicure.extractive_caption_summarization.yelp_word2Vec.summarization.sub_modular_functions import *

NUM_SENTENCES = 3

@pytest.fixture(scope="module")
def similarity_matrix(request):
    """
        The dummy similarity matrix consists of NUM_SENTENCES* NUM_SENTENCES
        entries where similarity of a sentence with itself is 0
    """
    return np.array([[1.0, 0.2, 0.8], [0.2, 1.0, 0.6], [0.8, 0.6, 1.0]])

@pytest.fixture(scope="module")
def vocab_indices(request):
    return np.arange(NUM_SENTENCES).tolist()

@pytest.fixture(scope="module")
def empty_summary_indices(request):
    return []

@pytest.fixture(scope="module")
def summary_contains_first(request):
    """
    The summary contains the first sentence only
    """
    return [0]

@pytest.fixture(scope="module")
def summary_contains_first_third(request):
    """
        The summary contains the first and third sentences
    """
    return [0, 2]


class TestSubmodularFunctions():

    def test_facility_function_null(self, similarity_matrix, vocab_indices, empty_summary_indices):
        """
            WHEN THE SUMMARY SET IS NULL THEN THE FACILITY FUNCTION
            SHOULD RETURN 0
            THIS IS THE NORMALISED PROPERTY OF THE FUNCTION
        """
        value = facility_function(vocab_indices, empty_summary_indices, similarity_matrix, {})
        assert value == 0.0

    def test_facility_function_first(self, similarity_matrix, vocab_indices, summary_contains_first):
        """
            FOR THE DUMMY DATA - IF THE SUMMARY CONTAINS ONLY THE FIRST SENTENCE THEN
            THE FACILITY FUNCTION SHOULD RETURN 2.0
            THIS IS CALCULATED BY HAND
        """
        value = facility_function(vocab_indices, summary_contains_first, similarity_matrix, {})
        assert value == 2.0

    def test_facility_function_first_third(self, similarity_matrix, vocab_indices, summary_contains_first_third):
        value = facility_function(vocab_indices, summary_contains_first_third, similarity_matrix, {})
        assert value == 2.6

    def test_penalty_function_null(self, similarity_matrix, vocab_indices, empty_summary_indices):
        config = {'penalty_lambda': 0.1}
        value = penalty_function(vocab_indices, empty_summary_indices, similarity_matrix, config)
        assert value == 0.0

    def test_penalty_function_first(self, similarity_matrix, vocab_indices, summary_contains_first):
        config = {'penalty_lambda': 0.1}
        value = penalty_function(vocab_indices, summary_contains_first, similarity_matrix, config)
        assert value == 1.0

    def test_penalty_function_first_third(self, similarity_matrix, vocab_indices, summary_contains_first_third):
        config = {'penalty_lambda': 0.1}
        value = penalty_function(vocab_indices, summary_contains_first_third, similarity_matrix, config)
        assert value == 0.64

    def test_graph_cut_function_null_value(self, similarity_matrix, vocab_indices, empty_summary_indices):
        config = {}
        value = graph_cut_function(vocab_indices, empty_summary_indices, similarity_matrix, config)
        assert value == 0.0

    def test_graph_cut_function_first(self, similarity_matrix, vocab_indices, summary_contains_first):
        config = {}
        value = graph_cut_function(vocab_indices, summary_contains_first, similarity_matrix, config)
        assert value == 1.0

    def test_graph_cut_function_first_third(self, similarity_matrix, vocab_indices, summary_contains_first_third):
        config = {}
        value = graph_cut_function(vocab_indices, summary_contains_first_third, similarity_matrix, config)
        assert value == 0.80

    def test_coverage_function_null_value(self, similarity_matrix, vocab_indices, empty_summary_indices):
        config = {'saturation_alpha': 0.9}
        value = coverage_function(vocab_indices, empty_summary_indices, similarity_matrix, config)
        assert value == 0.0

    def test_coverage_function_first(self, similarity_matrix, vocab_indices, summary_contains_first):
        config = {'saturation_alpha': 0.9}
        value = coverage_function(vocab_indices, summary_contains_first, similarity_matrix, config)
        assert value == 2.0

    def test_coverage_function_first_third(self, similarity_matrix, vocab_indices, summary_contains_first_third):
        config = {'saturation_alpha': 0.9}
        value = coverage_function(vocab_indices, summary_contains_first_third, similarity_matrix, config)
        assert value == 4.4

    def test_diversity_function(self, similarity_matrix, vocab_indices, empty_summary_indices):
        config = {'membership':[]}
        value = diversity_function(vocab_indices, empty_summary_indices, similarity_matrix, config)
        assert value == 0.0

    def test_diversity_function_first(self, similarity_matrix, vocab_indices, summary_contains_first):
        config = {'membership': [[0, 1], [2]]}
        value = diversity_function(vocab_indices, summary_contains_first, similarity_matrix, config)
        assert value == np.sqrt(2./3)

    def test_diversity_function_first_third(self, similarity_matrix, vocab_indices, summary_contains_first_third):
        config = {'membership':[[0,1], [2]]}
        value = diversity_function(vocab_indices, summary_contains_first_third, similarity_matrix, config)
        assert value == np.sqrt(2./3) + np.sqrt(2.4/3)

