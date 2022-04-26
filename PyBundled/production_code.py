import numpy as np 
import spacy
import pickle 

import pandas as pd

print('loading spacy pipeline ...')
# load spacy pipeline
nlp = spacy.load("en_core_web_lg",)

print('loading model ...')
# load classifier 
f = open('../model/rf_final.pkl', 'rb')
clf = pickle.load(f)




# data enriching 
def split_into_sentences(text):
    doc = nlp(text)
    spans = []
    span =[]
    for token in doc : 
        if token.pos_=='PUNCT' and  token.text in [',','.','?','!',';'] :
            span_text = ' '.join(span)
            spans.append(span_text)
            span=[]
        else:
            span.append(token.text)
    span_text = ' '.join(span)
    spans.append(span_text)
    spans = [s for s in spans if s!='']
    return spans

# count number of stop words  
def count_stop_words(text):
    doc = nlp(text)
    return sum(token.is_stop for token in doc)

def compute_similarity(pair):
    text1,text2 = pair
    doc1 = nlp(text1)
    doc2 = nlp(text2)
  
    return  doc1.similarity(doc2)

def get_sim_metrics(text):
    text_chunks = split_into_sentences(text)
    text_chunks_pairs = [text_chunks[i:i+2] for i in range(len(text_chunks)-1)]
    if len(text_chunks)<=1:
        return len(text_chunks),1,1
    similarity_scores = [compute_similarity(pair) for pair in text_chunks_pairs]
   # print('sims',similarity_scores)
    return len(text_chunks),  np.mean(similarity_scores) , np.min(similarity_scores)

 

# apply to data 
def similarity_feat_gen(data):
    data['sentence_count'], data['similarity_mean'] ,data['similarity_min']= zip(*data['text'].map(get_sim_metrics))
    data['stop_word_count'] = data['text'].apply(count_stop_words)
    return data[['sentence_count','similarity_mean','similarity_min','stop_word_count']]


## main function 
def main(text):
    data = pd.DataFrame({'text':[text]})
    data_enriched = similarity_feat_gen(data)
   # print(clf.score)
    # predict
    y = clf.predict(data_enriched)
    if len(y)==1 : 
       # print(data_enriched.values)
        return y[0]
    else:
        return y


if __name__=='__main__':
    test_text = " how white it is (lack of access for students of color); how expensive it is (related to whiteness); too much talk and not enough action"
   # print('text:',test_text)
    prediction = main(test_text)
   # print('prediction:',prediction)



