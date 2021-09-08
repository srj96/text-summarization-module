from nltk.cluster.util import cosine_distance
import numpy as np

class SummarizeMethod:

    '''Here we will be defining the methods for building a cosine similarity matrix in 
    order to compare the sentences in the article. Afte the matrix is created we will then, thus,
    ranks them according to similarity coefficient and then summarize the text from the given input  
    text data.
    '''
    '''Similarity Calculation'''
    # Input Arguments : Positional Sentences from the matrix #

    @classmethod
    def sentSimCalc(cls,sent_pos_1,sent_pos_2):
        sent_1 = [w for w in sent_pos_1]
        sent_2 = [w for w in sent_pos_2]
        
        word_corpus = list(set(sent_1 + sent_2))
        
        vector_1 = [0]*len(word_corpus)
        vector_2 = [0]*len(word_corpus)
        
        for word in sent_1:
            vector_1[word_corpus.index(word)] = 1 + vector_1[word_corpus.index(word)]
        
        for word in sent_2:
            vector_2[word_corpus.index(word)] = 1 + vector_2[word_corpus.index(word)]
            
        cls.cos_sim = 1-cosine_distance(vector_1,vector_2)
        
        return(cls.cos_sim)
    
    '''Similarity Matrix '''
    # Input Argument : new cleaned sentence list

    @staticmethod
    def simMatrix(sentence_lst):
        sim_matrix = np.zeros((len(sentence_lst),len(sentence_lst)))
        for ix1 in range(len(sentence_lst)):
            for ix2 in range(len(sentence_lst)):
                if ix1 == ix2:
                    continue
                sim_matrix[ix1][ix2] = SummarizeMethod.sentSimCalc(sentence_lst[ix1], sentence_lst[ix2])
        
        return(sim_matrix)

    

