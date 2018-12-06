import pydocumentdb.document_client as document_client

def docdb_query(account='gs3-prod',databaseID='gs3-prod',collectionID='statements',query = { 'query': 'SELECT top 2 * FROM server s' }):
	#Function that returns a dictionary with documentdb data given a collection and query
	url = "https://gs3-prod.documents.azure.com:443/"
	key = "uUQNDoZoDLq663RS1laK3aKQkglkwEX8HIc95OKkBvkUUl8uofg3dXXVEQYJ1Rw17eGRXSajUCWNFtt90NJf0w=="
	client = document_client.DocumentClient(url, {'masterKey': key})
	collection_link = 'dbs/' + databaseID + '/colls/' + collectionID
	options = {} 
	options['enableCrossPartitionQuery'] = True
	options['maxItemCount'] = 10000
	result_iterable = client.QueryDocuments(collection_link, query, options)
	results = list(result_iterable)
	return results