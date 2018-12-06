import os
os.chdir('repo/app')
from docdb_query import docdb_query
import requests
import pickle
import json
import pandas as pd
import numpy as np
url = 'http://localhost:8000/heartBeat'
url = 'http://localhost:8000/representativeStatementsFilter'
headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}

test = requests.get(url)
# times = []
# for i in range(len(respondents)):
# 	print(i)
# 	looking = True
# 	j=0
# 	while looking:
# 		if respondents[i]['log'][j]['idStudyObject'] == '090e5e49-51a7-4aa8-922c-71103a164f1a':
# 			looking=False
# 			times.append(respondents[i]['log'][j]['dateTime'])
# 		j=j+1
# 		if j==len(respondents[i]['log']):
# 			looking=False
# 			times.append(None)
# for i in range(len(respondents)):
# respondents[300]['ideated']
		

query = {'query': "Select c.id, c.label, c.gibberishResult.isGibberish, c.zeroValueResult.isZeroValue from c where c.idStudyStudyObject = '599a1cfb-eb49-48bc-a5c7-707dbd3d1201+21a32f39-6b50-4b0a-9307-bf34576d5b77'"}
statements = docdb_query(collectionID = 'statements', query=query)
statements = [{'label':d['label'], 'id':d['id']} for d in statements if d['isGibberish'] == False and d['isZeroValue']==False]
# statements = pd.DataFrame(statements)
# with open(cached_embeddings_filepath, 'rb') as fp:
#     cached_embeddings = pickle.load(fp)
# emb = np.zeros((len(statements),512))
# for i, s in enumerate(statements.label.apply(str)):
# 	emb[i,:] = cached_embeddings[s]
# emb = pd.DataFrame(emb)
# emb.index = statements.label
# statements.to_csv('../../analysis/phantom_statements_all.csv',index=False)
# emb.to_csv('../../analysis/phantom_embeddings_all.csv')
ids = pd.DataFrame(statements).id.tolist()



#drop = ['Buy','I do not plan to purchase any new devices in the next 6 to 12 months', 'I not just the salesperson','Size picture brand','Use','Hokohoko jkk I have to','Hokohoko jkk I have how','Need would use it to justify paying for it','The price and whether I will be able to maneuver with that computer','They are various','Very like']
#statements = [s for s in statements if s['label'] not in drop]
#drop2 = ['If I were to purchase a new device, I would probably purchase either an LG product','Stop attacks without me having to do anything','Stop virus without me having to do anything','I dont know of the top of my head','I always make sure to read lots','The ultimate in softboard fun','Online','The price and whether I will be able to maneuver with that computer']
#statements = [s for s in statements if s['label'] not in drop2]


query = {'query': 'Select c.id as idRespondent, c.data["21a32f39-6b50-4b0a-9307-bf34576d5b77"].ideated from c where c.idStudy = "599a1cfb-eb49-48bc-a5c7-707dbd3d1201" and c.definition.complete=true'}
respondents = docdb_query(collectionID = 'respondents', query=query)
resp_to_keep = []
for i, r in enumerate(respondents):
	if 'ideated' in r:
		for j, idea in enumerate(r['ideated']):
			if 'idStatement' in idea:
				if idea['idStatement'] in ids:
					resp_to_keep.append(r['idRespondent'])
resp_to_keep = list(set(resp_to_keep))

query = {'query': "Select c.idStatement, c.idRespondent from c where c.idStudyStudyObject = '599a1cfb-eb49-48bc-a5c7-707dbd3d1201+21a32f39-6b50-4b0a-9307-bf34576d5b77'"}
elab = docdb_query(collectionID = 'statement-elaborations', query=query)
elab = [e for e in elab if e['idRespondent'] in resp_to_keep]
elab = pd.DataFrame(elab)['idStatement'].value_counts().to_dict()
elab = {k: np.asscalar(v) for k, v in elab.items()}


#elabs = pd.DataFrame(form['statement_elaborations'])['idStatement'].value_counts()
#elabs = pd.DataFrame({'id':elabs.index, 'elab_count':elabs.values})

#for i in range(5,80):
i= 90
form = {'statements':statements[0:i], 'statement_elaborations':elab}
form['ideaPoolSizeLimit'] = 28
form['numRespondentsCompleted'] = i
form['expectedNumRespondents'] = 130
form['use_v2'] = False
# with open('test/request.json','w') as fp:
# 	json.dump(form,fp)

data = json.dumps(form)
results = requests.post(url, data=data, headers = headers)
results_dict = json.loads(results.content.decode('utf-8'))
with open('test/response.json','w') as fp:
	json.dump(results_dict,fp)
results_df = pd.DataFrame(results_dict)

statements_df = pd.DataFrame(statements[0:i])
statements_df = statements_df.merge(results_df,on='id')
#elab_df = pd.DataFrame(form['statement_elaborations'])['idStatement'].value_counts()
#elab_df = pd.DataFrame({'id':elab_df.index, 'elab_count':elab_df.values})
#statements_df = statements_df.merge(elab_df,how='left',on='id')
#statements_df['elab_count'] = statements_df['elab_count'].fillna(0)


print(statements_df[['label','isDuplicate','elab_count']][(statements_df['isGibberish']==False) & (statements_df['isZeroValue']==False)].sort_values('elab_count',ascending=False))
print(statements_df['label'][(statements_df['isGibberish']==False) & (statements_df['isZeroValue']==False)].iloc[0:28].tolist())
print(statements_df['label'][statements_df['isNotRepresentative']==False])

form['use_v2'] = True
data = json.dumps(form)
results = requests.post(url, data=data, headers = headers)
results_dict = json.loads(results.content.decode('utf-8'))
results_df = pd.DataFrame(results_dict)
statements_df2 = pd.DataFrame(statements[0:i])
statements_df2 = statements_df2.merge(results_df,on='id')
statements_df2['label'][statements_df2['isDuplicate']==False]
#elab_df = pd.DataFrame(form['statement_elaborations'])['idStatement'].value_counts()
#elab_df = pd.DataFrame({'id':elab_df.index, 'elab_count':elab_df.values})
#statements_df2 = statements_df2.merge(elab_df,how='left',on='id')
#statements_df2['elab_count'] = statements_df2['elab_count'].fillna(0)

statements_df['label'][statements_df['isDuplicate']==False].to_csv('../analysis/IRI_validataion.csv')

import tensorflow as tf 
import tensorflow_hub as hub
import pandas as pd 
from scipy.cluster.hierarchy import linkage

old_and_new = pd.DataFrame({'old_method':statements_df['label'][(statements_df['isGibberish']==False) & (statements_df['isZeroValue']==False)].iloc[0:28].tolist(), 'new_method':statements_df['label'][statements_df['isDuplicate']==False].tolist(), 'freddie':statements_df2['label'][statements_df2['isDuplicate']==False].tolist()})

cached_embeddings_filepath = 'data/cached_embeddings.p'
with open(cached_embeddings_filepath, 'rb') as fp:
	cached_embeddings = pickle.load(fp)
old_emb = np.zeros((len(old_and_new),512))
for i, s in enumerate(old_and_new.old_method.apply(str)):
	old_emb[i,:] = cached_embeddings[s]

new_emb1 = np.zeros((len(old_and_new),512))
for i, s in enumerate(old_and_new.new_method.apply(str)):
	new_emb1[i,:] = cached_embeddings[s]

new_emb2 = np.zeros((len(old_and_new),512))
for i, s in enumerate(old_and_new.freddie.apply(str)):
	new_emb2[i,:] = cached_embeddings[s]

Z = linkage(old_emb, method='complete',metric='cityblock')
n = len(Z) + 1
cache = dict()
for k in range(len(Z)):
  c1, c2 = int(Z[k][0]), int(Z[k][1])
  c1 = [c1] if c1 < n else cache.pop(c1)
  c2 = [c2] if c2 < n else cache.pop(c2)
  cache[n+k] = c1 + c2
cache[2*len(Z)]
old_and_new['old_method'] = np.array(old_and_new.old_method)[cache[2*len(Z)]]

Z = linkage(new_emb1, method='complete',metric='cityblock')
n = len(Z) + 1
cache = dict()
for k in range(len(Z)):
  c1, c2 = int(Z[k][0]), int(Z[k][1])
  c1 = [c1] if c1 < n else cache.pop(c1)
  c2 = [c2] if c2 < n else cache.pop(c2)
  cache[n+k] = c1 + c2
cache[2*len(Z)]
old_and_new['new_method'] = np.array(old_and_new.new_method)[cache[2*len(Z)]]

Z = linkage(new_emb2, method='complete',metric='cityblock')
n = len(Z) + 1
cache = dict()
for k in range(len(Z)):
  c1, c2 = int(Z[k][0]), int(Z[k][1])
  c1 = [c1] if c1 < n else cache.pop(c1)
  c2 = [c2] if c2 < n else cache.pop(c2)
  cache[n+k] = c1 + c2
cache[2*len(Z)]
old_and_new['freddie'] = np.array(old_and_new.freddie)[cache[2*len(Z)]]

maros = pd.read_csv('../../analysis/maros_statements.csv')


m_emb = np.zeros((len(old_and_new),512))
for i, s in enumerate(maros.label.apply(str)):
	m_emb[i,:] = cached_embeddings[s]

Z = linkage(m_emb, method='complete',metric='cityblock')
n = len(Z) + 1
cache = dict()
for k in range(len(Z)):
  c1, c2 = int(Z[k][0]), int(Z[k][1])
  c1 = [c1] if c1 < n else cache.pop(c1)
  c2 = [c2] if c2 < n else cache.pop(c2)
  cache[n+k] = c1 + c2
cache[2*len(Z)]
old_and_new['maros'] = np.array(maros.label)[cache[2*len(Z)]]

old_and_new = old_and_new[['maros','old_method','new_method','freddie']]
old_and_new.to_csv('../../analysis/test_IRI_study.csv',index=False)

all_emb = np.zeros((len(statements_df),512))
for i, s in enumerate(statements_df['label'].apply(str)):
	all_emb[i,:] = cached_embeddings[s]

Z = linkage(all_emb, method='complete',metric='cityblock')
n = len(Z) + 1
cache = dict()
for k in range(len(Z)):
  c1, c2 = int(Z[k][0]), int(Z[k][1])
  c1 = [c1] if c1 < n else cache.pop(c1)
  c2 = [c2] if c2 < n else cache.pop(c2)
  cache[n+k] = c1 + c2
cache[2*len(Z)]
statements_df = statements_df.reindex(cache[2*len(Z)])
statements_df.sort_index()[['label','isGibberish','isZeroValue']].to_csv('../../analysis/test_all_statements.csv',index=False)



#print(len(statements_df['label'][statements_df['isDuplicate']==False]))

# cached_embeddings_filepath = 'data/cached_embeddings.p'
# with open(cached_embeddings_filepath, 'rb') as fp:
# 	cached_embeddings = pickle.load(fp)

# statements = list(cached_embeddings.keys())
# columns = list(range(512))
# df = pd.DataFrame(index=statements, columns=columns)
# df = df.fillna(0) 
# for s in statements:
# 	df.loc[s,:] = cached_embeddings[s]
# df.to_csv('../../analysis/embeddings.csv')
# statements_df.to_csv('../../analysis/statements.csv')