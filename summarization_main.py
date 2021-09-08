import sys
# from flask import Flask, render_template 
import gradio as gr 
import numpy as np
import pandas as pd
import networkx as nx
from newsfetch.news import newspaper
from summarization_data_preproc import SummDataClean
from summarization_method import SummarizeMethod

class BaseClass:
    def __init__(self, **kwargs):
        if 'data' in kwargs:
            self.data = kwargs['data']
        # if 'num' in kwargs:
        #     self.num = kwargs['num']
        
        self.sent_lst , new_sent_lst = SummDataClean.sentFilter(old_data = self.data)
        self.similarity_matrix = SummarizeMethod.simMatrix(new_sent_lst)

    '''Summarization of the Text'''

    def summarizedResult(self):
        summarized_text = []
        sent_sim_graph = nx.from_numpy_array(self.similarity_matrix)
        sim_scores = nx.pagerank(sent_sim_graph)
        sent_rank = sorted(((sim_scores[ix],iv) for ix,iv in enumerate(self.sent_lst)),reverse = True)

        rank_num = (25*len(sent_rank))/100
        for ix in range(0,int(rank_num)):
            summarized_text.append("".join(sent_rank[ix][1]))
        
        return(". ".join(summarized_text))

if __name__ == "__main__":

     # url = 'https://www.hindustantimes.com/india-news/india-s-covid-19-positivity-rate-on-constant-downward-trend-health-ministry/story-JE3b8bHBqncaZYLYWMSfxK.html'
   
    def url_summ(url):
        news = newspaper(url) 
        obj = BaseClass(data = news.article)
        return(news.headline, obj.summarizedResult())

    gr.Interface(
        url_summ,
        [
            gr.inputs.Textbox(type = "str", label = 'Enter the URL'),
            # gr.inputs.Textbox(numeric= True, type= "number", label = "Enter the summary percent")
        ],

        [
            gr.outputs.Textbox(label = 'Headline of the Article'),
            gr.outputs.Textbox(label = 'Summary of the News Article')]
    
    ).launch()





    # print("\n\n")
    # print("Summarized Text is")
    # print("\n")
    # print(obj.summarizedResult())
