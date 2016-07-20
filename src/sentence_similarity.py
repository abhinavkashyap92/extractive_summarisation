"""
    Given two sentences this finds similarities of different kinds
    between two sentences
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SentenceSimilarity():
    """
        GIVEN A CONTINUOUS REPRESENTATION OF SENTENCES
        THIS FINDS THE SIMILARITY BETWEEN SENTENCES

        SIMILARITY BETWEEN EVERY OTHER SENTENCE
        GIVEN SENTENCES OF THE SHAPE (N,D)
        THIS RETURNS A SIMILARITY MATRIX OF THE SHAPE
        (N, N)
    """
    def __init__(self, sentences):
        """

        Args:
            sentences: shape (N, D)
             N - number of sentences
             D - d dimensional vector
        """
        self.types = ['cosine']
        self.sentences = sentences
        self.num_sentences, self.num_dimensions = self.sentences.shape
        if self.num_sentences < 2:
            raise ValueError("Ha Ha :p Dumb!! Please give more than two sentences to calculate "
                             "the similarity between 'Sentences'")

    def get_similarity(self, type="cosine"):
        if type == "cosine":
            similarity = self.__get_cosine_similarity()

        elif type == "cosine_sklearn":
            similarity = self.__get_cosine_similarity_sklearn()

        return similarity

    def __get_cosine_similarity(self):
        """
        Returns: Cosine similarity matrix of sentences
        """
        similarity = np.zeros((self.num_sentences, self.num_sentences))
        for i in xrange(self.num_sentences):
            for j in xrange(self.num_sentences):
                first_sentence = self.sentences[i]
                second_sentence = self.sentences[j]
                numerator = np.dot(first_sentence, second_sentence)
                denominator = np.linalg.norm(first_sentence) * np.linalg.norm(second_sentence)
                similarity[i][j] = numerator / denominator

        return similarity

    def __get_cosine_similarity_sklearn(self):
        """
        Sklearn has pairwise cosine similarity method
        Returns: similarity matrix using cosine distance
        """

        similarity = cosine_similarity(self.sentences, self.sentences)
        return similarity
