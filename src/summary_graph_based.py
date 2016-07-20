"""
This implements the paper
Graph based Submodular Selection for Extractive Summarization paper
by Hui Lin Jeff Bilmes and Shasha Xie 2009
"""
import copy
import numpy as np
from sentence_similarity import SentenceSimilarity
from sub_modular_functions import *
class GraphBasedSummarization():
    """
    This implements the graph based extractive summarization
    of sentences
    """
    # I AM SUMMARIZING THE CAPTIONS OF THE YELP DATASET
    # INTO FOOD DRINKS AND THE AMBIENCE ASPECTS OF A RESTAURANT
    # THE IDEA IS TO VISUALLY SUMMARIZE A GIVEN RESTAURANT
    # THERE ARE OTHER ASPECTS TO A RESTAURANT LIKE PRICE AND SERVICE
    # MAKING VISUAL SUMMARIES FOR THIS DOESNT SEEM TO MAKE SENSE
    def __init__(self, sentences, config):
        """

        Args:
            sentences: shape N, D
            N - number of sentences
            D - dimensions of the features
            config:
            sentence_similarity_type: What kind of similarity between sentences need to be used
            ["cosine", "cosine_sklearn"]
            budget; The number of sentences in the  summary
        """
        self.sentences = sentences
        self.num_sentences, self.dimensions = sentences.shape
        self.config = config
        self.sentence_similarity_type = config["sentence_similarity_type"]
        self.budget = config["budget"]
        if len(sentences) <= config["budget"]:
            raise ValueError("You would want the summary to be small and concise! For this "
                             "the budget needs to be lesser than the number of sentences "
                             "that are already there")
        self.similarity_matrix = self.__get_similarity_matrix()

    def __get_similarity_matrix(self):

        sentence_similarity = SentenceSimilarity(self.sentences)
        similarity_matrix = sentence_similarity.get_similarity(type = self.sentence_similarity_type)
        return similarity_matrix

    def get_similarity_matrix(self):
        """

        Returns: The similarity matrix that is used in
        finding the summary
        """
        try:
            return self.similarity_matrix
        except AttributeError:
            raise AttributeError("Similarity matrix is not defined")

    def summarize(self, sub_modular_func):
        """
        Args:
            sub_modular_func: function object that implements the submodular function
            All the submodular functions take in three arguments
            vocab_indices, summary_indices, similarity_matrix

        Summarise the sentences using the facility location
        function.
        Facility location function is a set function
        \sum_{i \in V} \max_{j \in S} w_{i,j}
        Place the above latexit it to see the formula

        Returns: a set S consisting of the summary
        """

        summary = set([])
        vocab_indices = np.arange(self.num_sentences).tolist()
        config = copy.deepcopy(self.config)

        # 1. WHILE THE NUMBER OF SENTENCES IN THE SUMMARY IS NOT BUDGET
        # 2. GO THROUGH ALL THE SENTENCES THAT ARE NOT IN SUMMARY
        # 3. FOR EVERY SENTENCE IN CALCULATE THE MARGINAL GAIN
        # 4. SELECT THE SENTENCE WITH MAXIMUM MARGINAL GAIN FOR THE SUMMARY
        print "*" * 80
        while len(summary) < self.budget:
            sentences_not_in_summary = list(set(np.arange(self.num_sentences).tolist()) - summary)
            summary_indices = summary
            best_index = None
            best_marginal_gain = -np.inf
            for index in sentences_not_in_summary:
                # 1. SUBMODULAR  FUNCTION WITHOUT ADDING NEW ELEMENT TO SUMMARY
                # 2. SUBMODULAR  FUNCTION BY ADDING THE NEW ELEMENT TO SUMMARY
                # 3. REMOVE THE ELEMENT ADDED TO THE SUMMARY
                # 3. CALCULATE THE MARGINAL GAIN
                # 4. SELECT THE ONE WITH MAXIMUM MARGINAL GAIN TO BE ADDED TO THE SUMMARY
                f_S = sub_modular_func(vocab_indices, list(summary_indices), self.similarity_matrix, config)
                summary_indices.add(index)
                f_S_add_s = sub_modular_func(vocab_indices, list(summary_indices), self.similarity_matrix, config)
                summary_indices.remove(index)
                marginal_gain = f_S_add_s - f_S
                if marginal_gain > best_marginal_gain:
                    best_marginal_gain = marginal_gain
                    best_index =index
            summary.add(best_index)
            print "Percentage complete %f" % ((100) * (len(summary)/float(self.budget)),)
        print "*" * 80
        return list(summary)
