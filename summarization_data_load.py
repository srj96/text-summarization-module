import sys
import textwrap
import numpy as np
import pandas as pd 
import networkx as nx
from summarization_data_preproc import SummDataClean
from summarization_method import SummarizeMethod
from topic_model_class import TopicModel

# Dashboard/GUI Components
import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

class BaseClass:
    def __init__(self, **kwargs):
        if 'data' in kwargs:
            self.data = kwargs['data']
        
        self.sent_lst , new_sent_lst = SummDataClean.sentFilter(old_data = self.data)
        self.topics = TopicModel.topicWords(self.sent_lst)
        self.similarity_matrix = SummarizeMethod.simMatrix(new_sent_lst)

    '''Summarization of the Text'''

    def summarizedResult(self):
        summarized_text = []
        sent_sim_graph = nx.from_numpy_array(self.similarity_matrix)
        sim_scores = nx.pagerank(sent_sim_graph)
        sent_rank = sorted(((sim_scores[ix],iv) for ix,iv in enumerate(self.sent_lst)),reverse = True)

        rank_num = int((25*len(sent_rank))/100)
        for ix in range(0,rank_num):
            summarized_text.append("".join(sent_rank[ix][1]))
        
        return(". ".join(summarized_text))
    
    '''Trending Topic Extraction'''
    def topicResult(self):
        return(self.topics)

if __name__ == "__main__":
    
    '''GUI Components for demo purpose'''
    
    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
        
    app.layout = html.Div([html.H1(children = 'Text Summarization and Trending Topic Module',style = {'backgroundColor' : 'lime'}),
        html.Div(dcc.Input(id='input-on-submit', type='text')),
        html.Button('Submit', id = 'submit-val', n_clicks = 0),
        html.Button('Reset',id = 'reset_button', n_clicks = 0),
        html.Div(id = 'output', style = {'backgroundColor': 'lightcyan'}),
        html.Hr(),
        html.Div(id = 'new_output', style = {'backgroundColor': 'yellow'}),
        html.Hr(),
        html.Div([dcc.Graph(id = 'graph_output', animate = False)])
    ])
    
    @app.callback(
        dash.dependencies.Output('output', component_property = 'children'),
        [dash.dependencies.Input('submit-val', 'n_clicks')],
        [dash.dependencies.State('input-on-submit', 'value')])
    def update_output(n_clicks,value):
        r = html.P(children=[html.Strong('Original Text : ')], style = {'backgroundColor' : 'white'})
        return(r,value)
    
    @app.callback(
        dash.dependencies.Output('input-on-submit','value'),
        [dash.dependencies.Input('reset_button','n_clicks')])
    def button_update(n_clicks):
        if n_clicks > 1 :
            return(' ')

    @app.callback(
        dash.dependencies.Output('new_output', component_property = 'children'),
        [dash.dependencies.Input('submit-val', 'n_clicks')],
        [dash.dependencies.State('input-on-submit', 'value')])
    def update_output_1(n_clicks,value):
        t = html.P(children = [html.Strong('Summarized Text : ')], style = {'backgroundColor' : 'white'})
        obj = BaseClass(data = value)
        new_text = obj.summarizedResult()

        return(t,'{}'.format(new_text))
    
    @app.callback(
        dash.dependencies.Output('graph_output', component_property = 'figure'),
        [dash.dependencies.Input('submit-val', 'n_clicks')],
        [dash.dependencies.State('input-on-submit', 'value')])
    def update_output_2(n_clicks,value):
        obj = BaseClass(data = value)

        wgt_lst = []
        name_lst = []

        for _ , topic in obj.topicResult():
            wgt_sum = sum([w[1] for w in topic])
            wgt_lst.extend([round((w[1]/wgt_sum)*100) for w in topic])
            name_lst.extend([w[0] for w in topic])
        
        index = list(np.arange(0,len(name_lst)))
        
        return({
            'data' : [
                go.Scatter(
                    x = index , y = wgt_lst,
                    mode='markers + text',
                    text = name_lst,
                    textposition='top center',
                    marker = dict(color= ['blue','green','orange','red','yellowgreen','blueviolet','brown','gray','purple']),
                    marker_size = list(10*np.arange(len(name_lst),0,-1))
                    )],
             'layout': dict(title = 'Topic Modeling Graph',  
                       xaxis = dict(title = 'Trending Topic Names'),
                       yaxis = dict(title = 'Percentage of Trend'))
        })
        
    
    app.run_server(debug=True)
    

    

