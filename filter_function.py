#from use_embeddings import get_use_embeddings
import pickle
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform

def filter_statements(statements,s_emb,study_data,sd_threshold,cached_embeddings_filepath):
	# Inputs:
		# statements: data frame that has the id, label, and elab_count
		# s_emb: array of embeddings n_statement x 512 (or dim of embedding)
		# study_data: a list of the idea pool size limit, the number of respondents who have completed, and the number that we expect to complete the question
		# sd_threshold: helps us decide how many statements we should keep
		# cached_embedddings_filepath: The file that has the saved embeddings

	#Load embeddings, get new embeddings and save results	
	# with open(cached_embeddings_filepath, 'rb') as fp:
	# 	cached_embeddings = pickle.load(fp)
	# print('Loaded embeddings')
	# cached_embeddings = get_use_embeddings(list(statements.label),cached_embeddings)
	# print('Found embeddings')
	# with open(cached_embeddings_filepath, 'wb') as fp:
	# 	pickle.dump(cached_embeddings, fp)

	#Caculate all pairwise distances
	d = pdist(s_emb)

	# Create lists for the statements and number of elaborations. Get embeddings for study statements
	# s = list(statements.label.apply(str))
	num = list(statements.elab_count)
	# s_emb = np.zeros((len(s),512))
	# for i in range(len(s)):
	# 	s_emb[i,:] = cached_embeddings[s[i]]


	#Define threshold at which we will cutoff statements and save the smallest distance
	cutoff = np.mean(d) - sd_threshold * np.std(d)
	min_dist = np.min(d)

	#Extra variables to make sure the function doesn't include/exclude too many ideas:
	#Follows formula similar to idea pool limitation that aproaches the idea pool limit as we get halfway through a study
	#Also checks if we are halfway through the study. If halfway we will return the most dissimlar ideas up to the idea pool limit.
	current_ips = np.round((1 + np.sqrt(1 + study_data[1]*1056/25)) / 2) 
	target_num_ideas = min(max(current_ips, 20), study_data[0])
	halfway = study_data[1] / study_data[2] > .5
	margin = 1

	#Variables needed for loop
	indices = np.triu_indices(len(s_emb),1)
	remove = []
	nStatements = len(s_emb)
	#Loop logic:
	#If less than half of the respondents have gone through the open end, we want to continue removing statements until:
		#1. The minimum distance is below the threshold and we are also below the upper margin of the target number of ideas
		#2. We have less than the lower margin of the target number of ideas
	#If more than half of the respondents have answered the open end, we want to remove ideas until we reach the idea pool limit.
	while (halfway==False and (min_dist < cutoff or (nStatements > target_num_ideas + margin)) and (nStatements > target_num_ideas - margin))  or  (halfway == True and nStatements > study_data[0]):
		#Algorithm
		#1. Find which 2 statements have the minimum distance and store their indices
		#2. If 2 statements have different number of elaborations, drop the statement with the least number of elaborations
		#3. If the statements have the same number of elaborations, remove the statement that has the next smallest distance to the other statements
		#4. Track which statements have been removed and set the distance between the removed statements and all other statements to the maximum distance (so we don't evaluate those pairs)
		#5. Calculate the new minimum distance for the next iteration of the loop
		index = np.where(d == min_dist)[0][0]
		x1 = int(indices[0][index])
		x2 = int(indices[1][index])
		#print('statements:',statements.label[x1],'vs.',statements.label[x2])
		indices1 = np.where(((indices[0]==x1) & (indices[1]!=x2)) | ((indices[1]==x1) & (indices[0]!=x2)))[0]
		indices2 = np.where(((indices[0]==x2) & (indices[1]!=x1)) | ((indices[1]==x2) & (indices[0]!=x1)))[0]
		if num[x1] > num[x2] and x2 not in remove:
			remove.append(x2)
			d[indices2]=max(d)
		#	print(statements.label[x2])
		if num[x1] < num[x2] and x1 not in remove:
			remove.append(x1)
			d[indices1]=max(d)
		#	print(statements.label[x1])
		if num[x1] == num[x2]:
			d1 = np.min(d[indices1])
			d2 = np.min(d[indices2])
			if d1 >= d2 and x2 not in remove:
				remove.append(x2)
				d[indices2]=max(d)
		#		print(statements.label[x2])
			if d1 < d2 and x1 not in remove:
				remove.append(x1)
				d[indices1]=max(d)
				#print(statements.label[x1])
		d[index]=max(d)
		min_dist = np.min(d)
		nStatements = nStatements - 1
	
	print('Removed',len(remove),'ideas')
	print('')
	#print('Number of statements:', nStatements)
	mask = np.zeros(len(statements),dtype=bool) #np.ones_like(a,dtype=bool)
	mask[remove] = True
	statements['isDuplicate'] = mask
	return statements[['id','isDuplicate']]