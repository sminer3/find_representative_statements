#import sys
#sys.stdout = open('log.txt', 'w')
#from filter_function import filter_statements
from filter_function2 import filter_statements2
import numpy as np
import pickle
import falcon
import pandas as pd
import json

#Start tensorflow session using the universal sentence encoder
import tensorflow as tf
import tensorflow_hub as hub
module_folder = "encoder" 
g = tf.Graph()
with g.as_default():
  text_input = tf.placeholder(dtype=tf.string, shape=[None])
  embed = hub.Module(module_folder)
  get_embedding = embed(text_input)
  init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
g.finalize()
session = tf.Session(graph=g)
session.run(init_op)

#Filepath to cache the embeddings
cached_embeddings_filepath = 'data/cached_embeddings.p'

class processStatements(object):
    def on_post(self, req, resp):
        #Get request and check if all needed fields are in the requests
        form = json.loads(req.stream.read().decode('utf-8'))
        checks = 'statements' in form and 'statement_elaborations' in form 
        study_checks = all(field in form for field in ['ideaPoolSizeLimit', 'numRespondentsCompleted', 'expectedNumRespondents'])
        form['use_v2'] = True

        #Check if the statement and elaboration fields contain all the needed fields
        other_checks = False
        if checks:
            statement_check = all([all(k in obj.keys() for k in ('id','label')) for obj in form['statements']])
            elab_check = all([isinstance(obj, (int, np.integer)) for obj in form['statement_elaborations'].values()])
            if statement_check and elab_check:
                other_checks = True
        
        if checks and study_checks and other_checks:
            try:
                #Below no longer needed
                #Subset the statements that are not gibberish / zero value
                #all_statements = pd.DataFrame(form['statements'])
                #statements = all_statements[(all_statements['isGibberish']==False) & (all_statements['isZeroValue']==False)]

                statements = pd.DataFrame(form['statements'])

                #Below no longer needed
                #Count the elaborations and merge the counts with the statements table
                #elabs = pd.DataFrame(form['statement_elaborations'])['idStatement'].value_counts()
                #elabs = pd.DataFrame({'id':elabs.index, 'elab_count':elabs.values})
                
                elabs = pd.Series(form['statement_elaborations'], name = 'elab_count')
                elabs.index.name = 'id'
                elabs = elabs.reset_index()
                statements = statements.merge(elabs,how='left',on='id')
                statements['elab_count'] = statements['elab_count'].fillna(0)

                #Get cached embeddings and check if we are missing any embeddings 
                with open(cached_embeddings_filepath, 'rb') as fp:
                    cached_embeddings = pickle.load(fp)
                missing = list(set(statements.label)-set(cached_embeddings.keys()))

                #If we are missing any embeddings, get universal sentence embeddings and save the added results
                if len(missing)>0:
                    embeddings = session.run(get_embedding, feed_dict={text_input: missing})
                    for i, sentence in enumerate(missing):
                        cached_embeddings[sentence] = embeddings[i]
                    with open(cached_embeddings_filepath, 'wb') as fp:
                        pickle.dump(cached_embeddings, fp)

                #Get embedding matrix for the valid statements from the request
                s_emb = np.zeros((len(statements),512))
                for i, s in enumerate(statements.label.apply(str)):
                    s_emb[i,:] = cached_embeddings[s]

                #Run the filter function
                study_data = [form['ideaPoolSizeLimit'], form['numRespondentsCompleted'], form['expectedNumRespondents']]
                results = filter_statements2(statements=statements, s_emb=s_emb, study_data=study_data, sd_threshold=1.1, cached_embeddings_filepath=cached_embeddings_filepath)

                #Below was formerly used for testing 
                # if form['use_v2']==True:
                    #     print('used newer')
                #     results = filter_statements2(statements=statements, s_emb=s_emb, study_data=study_data, sd_threshold=1.1, cached_embeddings_filepath=cached_embeddings_filepath)
                # else:
                #     results = filter_statements(statements=statements, s_emb=s_emb, study_data=study_data, sd_threshold=1.1, cached_embeddings_filepath=cached_embeddings_filepath)
                #     print('used older')
                
                #Below no longer needed
                #Add the other statements for the response
                #other_statements = all_statements[(all_statements['isGibberish']) | (all_statements['isZeroValue'])]
                #other_statements['isDuplicate'] = True
                #other_statements = other_statements[['id','isDuplicate']]
                #results = pd.concat((results,other_statements))

                #Convert table to json and send the response
                results = results.to_dict('records')
                resp.body = json.dumps(results)
                resp.status = falcon.HTTP_200
            except Exception as e:
                print(e)
                resp.body = json.dumps({'Error': 'An internal server error has occurred'})
                resp.status = falcon.HTTP_500
        else:
            if checks == False:
                resp.body = json.dumps({'Error': 'statements or statement_elaborations not found in the request', 'Request': form})
            elif study_checks == False:
                resp.body = json.dumps({'Error': 'ideaPoolSizeLimit, numRespondentsCompleted, or expectedNumRespondents not found in the request', 'Request': form})
            elif other_checks == False:
                resp.body = json.dumps({'Error': 'id and label not found in at least one of the statements array or at least one of the elaboration counts is not an integer', 'Request': form})
            else:
                resp.body = json.dumps({'Error': 'Input checking is broken. Function should be checked.', 'Request': form})
            resp.status = falcon.HTTP_400


class HeartBeat(object):
    def on_get(self, req, resp):
        doc = {
            "resp":"It's aliiive"
        }

        resp.body = json.dumps(doc, ensure_ascii=False)
        resp.status = falcon.HTTP_200

# instantiate a callable WSGI app
app = falcon.API()

# long-lived resource class instance
representativeStatementsFilter = processStatements()
heartBeat = HeartBeat()

# handle all requests to the URL path
app.req_options.auto_parse_form_urlencoded = False
app.add_route('/representativeStatementsFilter', representativeStatementsFilter)
app.add_route('/heartBeat', heartBeat)
