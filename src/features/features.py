import spacy
import lftk
import xml.etree.ElementTree as ET
import pandas as pd
import os
import numpy as np


nlp = spacy.load('en_core_web_sm')

def extract_text_from_xml(file_path):
    text = []
    tree = ET.parse(file_path)
    root = tree.getroot()
    for element in root.findall('.//answer'):
        text.append(element.text)
    return text

def process_directory(directory_path):
    texts = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.xml'):
            file_path = os.path.join(directory_path, filename)
            text = extract_text_from_xml(file_path)
            text = [word for word in text if word is not None]
            texts.append(text)
    return texts

def process_text_and_extract_features(texts):
    for text in texts:
        doc = nlp(' '.join(text))
    LFTK = lftk.Extractor(docs=doc)
    LFTK.customize(stop_words=True, punctuations=False, round_decimal=3)
    extracted_features = LFTK.extract(features='*')
    return pd.DataFrame(list(extracted_features.items()), columns=['features', 'r'])

mistral_corpus_path = '../../corpora/mistralai-corpus'
cefr_corpus_path = '../../corpora/cefr-asag-extracted'

mistral_texts = process_directory(mistral_corpus_path)
mistral_features = process_text_and_extract_features(mistral_texts)
cefr_texts = process_directory(cefr_corpus_path)
cefr_features = process_text_and_extract_features(cefr_texts)

df_features = pd.merge(mistral_features, cefr_features, on='features', how='inner')
df_features.index = np.arange(1, len(df_features) + 1)
df_features.head(20)