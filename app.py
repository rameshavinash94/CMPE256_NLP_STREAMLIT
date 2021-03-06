
import spacy
import pandas as pd
from ContextExtraction import ContextExtraction
from DocumentRetrival import DocumentRetrival
from DataWrangling import DataWrangler
from ContextSimilarity import ContextSimilarity
from MLModel import MLModel
import streamlit as st
import requests
from flatten_json import flatten
from sentence_transformers import SentenceTransformer, util
import re
import os

nlp = spacy.load('en_core_web_lg')
  # Load Universal Sentence Encoder and later find context similarity for ranking paragraphs

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

with st.form(key='my_form'):
    question = st.text_input('Type your query', 'who is mark zuckerberg?')
    submit_button = st.form_submit_button(label='Submit')

if submit_button:
  #create a Document retrival object
  doc_retrive_obj = DocumentRetrival(nlp)

  #call UserInput func to get query input
  query = doc_retrive_obj.UserInput(question)

  #call preprocess func to preprocess query if required
  doc_retrive_obj.PreprocessUserInput()

  #call Retrive func with required top_n docs for retrival from Wiki
  pages = doc_retrive_obj.Retrive(3)

  #create a Extraction retrival object
  context_extract_obj = ContextExtraction(nlp)

  # Create a spacy matcher for the user query to parse the pages
  context_extract_obj.AddPhraseMatcher(query)

  # extract necessary context
  context_extract_obj.RetriveMatch(pages)

  # convert to pandas df
  text = context_extract_obj.StoreFindingAsDf()

  #create a Data Wrangler object
  data_wrangler_obj = DataWrangler(nlp)

  #cleaned Dataframe
  cleaned_df = data_wrangler_obj.DataWranglerDf(text)

  #create a Context Similarity object
  context_similarity_obj = ContextSimilarity(model)

  #find the Similarites of Different context
  con_list = context_similarity_obj.ContextSimilarity(query,cleaned_df['Wikipedia_Paragraphs'])

  context_similarity_df = context_similarity_obj.ConvertToDf(con_list)

  Merged_Df = context_similarity_obj.MergeDf(context_similarity_df,cleaned_df)

  #retreive top N rows from dataframe
  TopNDf = context_similarity_obj.TopNSimilarityDf(Merged_Df,top_n=20)

  #create a ML Model object
  ML_Model_obj = MLModel()

  #call the Roberta model
  roberta_finding = ML_Model_obj.RobertaModel(TopNDf,query)

  #final Df post model prediction
  Final_DF = ML_Model_obj.ConverttoDf()

  #filtering only top N out of it.
  Results = ML_Model_obj.TopNDf(Final_DF,top_n=5)
  Results['Imageapi'] = 'https://en.wikipedia.org/w/api.php?action=query&titles='+ Results['Wiki_Page'].astype('str').str.extract(pat = "('.*')").replace("'", '', regex=True) + '&prop=pageimages&format=json&pithumbsize=100'
  Results['Wiki_Page'] = 'https://en.wikipedia.org/wiki/' + Results['Wiki_Page'].astype('str').str.extract(pat = "('.*')").replace("'", '', regex=True)
  Results['Wiki_Page'] = Results['Wiki_Page'].replace(" ", '_', regex=True)
  for index, row in Results.iterrows():
    st.markdown('**{0}**'.format(row['Prediction'].upper()))
    r = requests.get(row['Imageapi'])
    test = r.json()
    flat_json = flatten(test)
    for x,y in flat_json.items():
      if re.findall('https.*',str(y)):
        st.image(y)
    st.markdown('_wiki:_ **{0}**'.format(row['Wiki_Page']))
    cont = '<p style="font-family:sans-serif; color:black; font-size: 8px;">{0}</p>'.format(row['Context'])
    st.write(cont,unsafe_allow_html=True)
