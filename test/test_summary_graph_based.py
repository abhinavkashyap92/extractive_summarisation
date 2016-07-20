import pytest
import numpy as np
from epicure.extractive_caption_summarization.yelp_word2Vec.summarization.summary_graph_based import *
from epicure.extractive_caption_summarization.yelp_word2Vec.summarization.sub_modular_functions import *
@pytest.fixture(scope="module")
def similarity_matrix(request):
    """
        The dummy similarity matrix consists of NUM_SENTENCES* NUM_SENTENCES
        entries where similarity of a sentence with itself is 0
    """
    return np.array([[1.0, 0.2, 0.8], [0.2, 1.0, 0.6], [0.8, 0.6, 1.0]])

class TestSummaryGraphBased():

    def test_graph_based_summary_facility(self, similarity_matrix):
        """
            Create random sentences
            Set the similarity matrix - Although this violates the Object Oriented principles
            this is done for the ease of testing
        """
        CONFIG = {'sentence_similarity_type': 'cosine_sklearn',
                  'budget': 1}
        summarizer =GraphBasedSummarization(sentences=np.random.randn(3, 50), config=CONFIG)
        # Over-riding the object attribute for the ease of testing
        # This has been done by hand. The summary should be 2
        summarizer.similarity_matrix = similarity_matrix
        indices = summarizer.summarize(facility_function)
        assert indices[0] == 2
