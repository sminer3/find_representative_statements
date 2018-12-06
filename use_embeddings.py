import tensorflow as tf
import tensorflow_hub as hub
module_url = "https://tfhub.dev/google/universal-sentence-encoder/2" 
g = tf.Graph()
with g.as_default():
  text_input = tf.placeholder(dtype=tf.string, shape=[None])
  embed = hub.Module(module_url)
  get_embedding = embed(text_input)
  init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
g.finalize() # not mandatory, but a nice practice, especially in multithreaded environment
session = tf.Session(graph=g)
session.run(init_op)

def get_use_embeddings(statements,cached_entries,session):
	print('3')
	missing = list(set(statements)-set(cached_entries.keys()))
	if len(missing)>0:
		#Input a list of sentences and get a np array of sentence embeddings
		print('missing:',missing)
		with tf.Session() as session:
			session.run([tf.global_variables_initializer(), tf.tables_initializer()])
			print('4')
			print(session.run(embed(missing)))
			embeddings = session.run(embed(missing))
			print(embeddings)
		print('5')
		for i in range(len(missing)):
			cached_entries[missing[i]] = embeddings[i]
		print('6')
		return cached_entries
	else:
		return cached_entries
