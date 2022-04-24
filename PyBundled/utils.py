# import  logistic regression , random forest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# import confusion matrix , classification report, accuracy_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np 
import pprint
# compute false negative rate

import pandas as pd
def prepare_training_data(data):
    #print(f'data cols {data.columns}')
    # define sample count 
    pos_count = len(data[data['is_bundled']==1])
    sample_count = int(pos_count*0.8) 

    # sample 
    data_1 = data[data['is_bundled']==1].sample(sample_count)
    data_0 = data[data['is_bundled']==0].sample(sample_count)
    data_train = pd.concat([data_0,data_1]).values
    # prepare data 
    X = data.drop(['is_bundled'], axis=1)
    y = data['is_bundled']
    return X,y

def get_model_results(model,data):
    # prepare data
    X = data.drop(['is_bundled'], axis=1)
    y = data[['is_bundled']]
    # predict
    y_pred = model.predict(X)
    # get index of false negatives
    temp_table = y.copy()
    temp_table['pred'] = y_pred
    fn_index = temp_table[(temp_table['is_bundled']==1)&(temp_table['pred']==0)].index



    # get confusion matrix
    cm = confusion_matrix(y, y_pred)
    # get classification report
    repport = classification_report(y, y_pred)

    # get model features
    features =model.feature_names_in_

    return  cm, repport , features , fn_index

def run_experiment(data,config):
    model_type = config['model_type']
    feat_gen_func = config['feat_gen_func']
    model_params = config['model_params']
    cach_data = config['cach_data']
    data = data.copy()
    

    # define model
    if model_type == 'logistic':
        model = LogisticRegression(**model_params)
    elif model_type == 'randomforest':
        model = RandomForestClassifier(**model_params)
    
    # prepare data
    if isinstance(cach_data, pd.DataFrame): 
        data_enriched = cach_data
    else:
        data_enriched = feat_gen_func(data)
        cach_data = data_enriched
        
    
    X,y = prepare_training_data(data_enriched)
    # fit model
    model.fit(X, y)
    # get results
    cm, repport , features ,fn_index= get_model_results(model,data_enriched)
    return  cm, repport , features ,fn_index, cach_data


class Span_extractor:

    def __init__(self, nlp,sentence: str,deep_extraction=True,debug=False, verify_treshold=0.7):
        self.sentence = sentence
        self.verify_treshold = verify_treshold
        self.debug=debug
        self.deep_extraction = deep_extraction
        self.nlp = nlp
        self.doc = self.nlp(sentence)
        # loggers 
        self.spans_logger = []
    
    def is_valid_span(self, span):
        """check if span is valid

        Args:
            span (_type_): span to check

        Returns:
            _type_: True if valid, False otherwise
        """
        def RepresentsInt(s):
            try: 
                int(s)
                return True
            except ValueError:
                return False

        def is_propn(span):
            span_str = str(span)
            if len(span_str.split(' ')) == 1:
                token = [token for token in self.doc if token.text == span_str][0]
                print(token , token.pos_)
                try : 
                    
                    if token.pos_ in  ['PROPN','CCONJ']:
                        return True
                except:
                    pass
            return False
                
        if RepresentsInt(str(span)):
            return False
        if is_propn(span):
            return False
        
        
        
        
        return True
    
    def extract_spans(self,spacy_span):
        """extract NP,VP span from e sentence text

        Args:
            sentence (_type_): text (with 1 sentence)

        Returns:
            _type_: list of VP,NP spans
        """
        spans = []
        for child in spacy_span._.children:
            if self.debug : 
                print(child , list(child._.labels))
            intersection = set(child._.labels).intersection(set(['S','NP','VP','PP','ADJP','ADVP','NML']))
            if len(intersection)!=0:
                spans.append(child)
        # verify re-itteration
        span_len = []
        for span in spans:
            span_len.append(len(str(span).split(' ')))
        # normalise len
        span_len = [x/len(self.sentence.split(' ')) for x in span_len]
        spans_to_verify = [idx for idx, x in enumerate(span_len) if x > self.verify_treshold]
        return spans , spans_to_verify
    
    def itterate(self):
        """run the extractor

        Returns:
            _type_: list of VP,NP spans
        """
        sent = list(self.doc.sents)[0]
        spans,spans_to_verify = self.extract_spans(sent)
        if self.debug : 
            print('->',sent)
            pprint.pprint(spans)
            print(spans_to_verify)
            print('------')
        spans_to_log = [str(span) for idx ,span in enumerate(spans) if ( idx not in spans_to_verify)   and self.is_valid_span((span))]
        self.spans_logger.extend(spans_to_log)

        if self.deep_extraction :

            while spans_to_verify !=[] : 
                for span_id in spans_to_verify:
                    span = spans[span_id]
                    spans,spans_to_verify = self.extract_spans(span)
                    if self.debug : 
                        print('->',span)
                        pprint.pprint(spans)
                        print(spans_to_verify)
                        print('------')
                    spans_to_log = [str(span) for idx ,span in enumerate(spans) if ( idx not in spans_to_verify)   and self.is_valid_span((span))]
                    self.spans_logger.extend(spans_to_log)
    
    def get_spans(self):
        """get the extracted spans

        Returns:
            _type_: list of VP,NP spans
        """
        spans_logger_str = ' '.join(self.spans_logger)
        ratio = len(spans_logger_str.split(' ') )/len(self.sentence.split(' '))
        if ratio <0.3 :
            return ('PROBLEM' , self.sentence)
        return ('GOOD',self.spans_logger)
        
      
        
    


        






