# ANSI colour codes
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
MAGENTA = '\033[95m'
CYAN = '\033[96m'
RESET = '\033[0m'

# Message to the user
print(GREEN + "Please wait..." + RESET)

# Importing libraries
import sys
from SPARQLWrapper import SPARQLWrapper, JSON
import torch
import torch.nn as nn
import pandas as pd
from unidecode import unidecode
import difflib
from tqdm import tqdm
import spacy
nlp = spacy.load('en_core_web_lg')
from transformers import BertTokenizer, BertModel, logging
logging.set_verbosity_error()

# Initialise variables and GPU
device = 'mps' if (torch.backends.mps.is_available()) else 'cuda' if ( torch.cuda.is_available()) else 'cpu'
BERT_MODEL = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, do_lower_case=True)

# Firstly read the dictionary I created
df = pd.read_csv("dataset/entity_dict.csv", sep = ',')
Entities = df['Entity']
Entity_ids = df['Id']

# Remove accents from the 'Entity' column
df['Entity'] = df['Entity'].apply(lambda x: unidecode(x))

# Create a dictionary with entities as keys and entity_ids as values
entity_to_id = {entity: entity_id for entity, entity_id in zip(df['Entity'], df['Id'])}
entity_list = list(entity_to_id.keys())

# This dictionary for similar words for extreme cases
entity_docs = {}

# Create a dictionary with docs as keys and entity_ids as values
for entity_name, id in tqdm(entity_to_id.items(), desc='Building Knowledge Base'):
    doc = nlp(entity_name)
    if doc.has_vector:
        entity_docs[doc] = id

# Then read the relations
df = pd.read_csv("dataset/relation_vocab.csv")

# Create a list with the relation vocabulary
relation_vocab = df['Relation'].to_list()

del df

class BERT_SPAN(torch.nn.Module):
    def __init__(self, bert_model, vocab_size):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model)

        self.start_head = nn.Sequential(
            nn.Dropout(p=0.15),
            nn.Linear(self.bert.config.hidden_size, 1),
            nn.Flatten(),
            nn.Softmax(dim=1)
        )

        self.end_head = nn.Sequential(
            nn.Dropout(p=0.15),
            nn.Linear(self.bert.config.hidden_size, 1),
            nn.Flatten(),
            nn.Softmax(dim=1)
        )
        
        self.vocab_size = vocab_size

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        
        start_ent = self.start_head(sequence_output)
        end_ent = self.end_head(sequence_output)
        
        return start_ent * attention_mask, end_ent * attention_mask

class BERT_REL(torch.nn.Module):
    def __init__(self, bert_model, vocab_size):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model)

        self.relation_head = nn.Sequential(
            nn.Dropout(0.15),
            nn.Linear(self.bert.config.hidden_size, vocab_size),
            nn.Softmax(dim=1)
        ) 
        self.vocab_size = vocab_size

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)        
        return  self.relation_head(outputs[1])

def preprocess(question):
    question = unidecode(question.replace("?", "").replace("'s", ""))
    ids = []
    mask = []
    
    # Encode the question
    encoding = tokenizer.encode_plus(
        text = question,
        return_attention_mask=True,
        add_special_tokens=False
    )

    ids.append(encoding['input_ids'])
    mask.append(encoding['attention_mask'])

    return torch.tensor(ids), torch.tensor(mask)

def relation_prediction(model, ids, mask, relation_vocab):
        
    # Predict 
    relation_logits = model(input_ids=ids, attention_mask=mask)
    _ , prediction = torch.max(relation_logits, dim=1)
    
    return relation_vocab[prediction]

# In case the entity is slightly off by a word or summat
def find_closest_match(entity, entity_docs, entity_to_id, entity_list):
        
    doc = nlp(entity)
    if doc.has_vector:

        scores = {}
        # Find similarity scores between entity and each entity in dictionary
        for doc_ent, id in entity_docs.items():
            scores[id] = doc.similarity(doc_ent)

        best_score = max(scores.values())
        return [id for id, score in scores.items() if score == best_score][0].split()[0]
    
    # Else simply find the best match from the rest
    best_match = difflib.get_close_matches(entity, entity_list, n=1, cutoff=0.12)
    if best_match != []:
        return entity_to_id[best_match[0]][0]
    return None

endpoint_url = "https://query.wikidata.org/sparql"

def entity_prediction(model, ids, mask, entity_docs, entity_to_id, entity_list):

    # Probabilities from the model
    start_logits, end_logits = model(input_ids=ids, attention_mask=mask)

    # Find the start index with the highest probability
    start_pred = torch.argmax(start_logits, dim=1)

    # Create a mask with the same shape as the matrix
    start_mask = torch.zeros(len(mask[0])).to(device)
    start_mask[start_pred.item():] = 1

    # Find the end index with the highest probability
    masked_end_logits = end_logits * start_mask
    end_pred = torch.argmax(masked_end_logits, dim=1)

    # Get the entity from the tokens
    tokens = ids[0][start_pred.item():end_pred.item()+1]
    entity = tokenizer.decode(tokens)

    # Initially check if it's present in its current form (without accents etc)
    if entity in entity_to_id:
        return entity_to_id[entity]
    else:
        entityid = get_entityid(entity)

        if entityid != []:
            return entityid[0]

    # Fallback to the most similar from the dictionary
    return find_closest_match(entity, entity_docs, entity_to_id, entity_list)

def get_results(endpoint_url, query):
    user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
    sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()

def get_entityid(label):
    query = """SELECT distinct ?item ?itemLabel ?itemDescription WHERE{  
    ?item ?label "%s"@en.
    ?article schema:about ?item .
    ?article schema:inLanguage "en" .
    ?article schema:isPartOf <https://en.wikipedia.org/>.	
    SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }    
    }""" % (label.title())

    results = get_results(endpoint_url, query)

    entityids = []
    for result in results["results"]["bindings"]:
        entityids.append(result["item"]["value"].split(sep='/')[-1])
    return entityids

def query_builder(entityid, relation):

    # Check whether the question relation is inverse
    inverse = False
    if (relation[0] == 'R'):
        relation = relation.replace('R', 'P')
        inverse = True

    if inverse:
        query = """
        SELECT ?item ?itemLabel 
        WHERE 
        {
        ?item wdt:%s wd:%s.
        SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
        }""" % ( relation, entityid)
    else:
        query = """
        SELECT ?item ?itemLabel 
        WHERE 
        {
        wd:%s wdt:%s ?item.
        SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
        }"""  % (entityid, relation)

    return query

def query_executor(query):
    
    endpoint = "https://query.wikidata.org/sparql"
    user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
    sparql = SPARQLWrapper(endpoint, agent=user_agent)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    return results["results"]["bindings"]

models = []
for i in tqdm(range(2) , desc='Loading Entity & Relation Model'):

    if i ==0:
        entity_model = BERT_SPAN(bert_model=BERT_MODEL, vocab_size=len(relation_vocab)).to(device)
        entity_model.load_state_dict(torch.load('./best_span_model.pt'))
    else:
        relation_model = BERT_REL(bert_model=BERT_MODEL, vocab_size=len(relation_vocab)).to(device)
        relation_model.load_state_dict(torch.load('./best_relation_model.pt'))

print(GREEN + "Hey there! Type your question or q to quit" + RESET)

while(True):
    print(MAGENTA + "\nHow may I assist you?" + RESET)
    print(CYAN, end='')
    question = input()
    print(RESET, end='')
    if(question == "q"): 
        break
    ids, mask = preprocess(question)
    ids = ids.to(device)
    mask = mask.to(device)
    relation = relation_prediction(relation_model, ids, mask, relation_vocab)
    entity = entity_prediction(entity_model, ids, mask, entity_docs, entity_to_id, entity_list)
    if entity == None:
        print(MAGENTA + "Nothing found, I'm sorry..." + RESET)
    else: 
        query = query_builder(entityid=entity, relation=relation)
        results = query_executor(query)
        if results== []:
            print(MAGENTA + "Nothing found, I'm sorry..." + RESET)
        else:
            print(YELLOW + results[0]["itemLabel"]["value"] + RESET)