#from use_embeddings import get_use_embeddings
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform

def filter_statements2(statements,s_emb,study_data,sd_threshold,cached_embeddings_filepath):
	# Inputs:
		# statements: data frame that has the id, label, and elab_count
		# s_emb: array of embeddings n_statement x 512 (or dim of embedding)
		# study_data: a list of the idea pool size limit, the number of respondents who have completed, and the number that we expect to complete the question
		# sd_threshold: helps us decide how many statements we should keep
		# cached_embedddings_filepath: The file that has the saved embeddings

	#Caculate all pairwise distances
	d = pdist(s_emb)
	d_square = squareform(d)
	means = np.mean(d_square,axis=1)
	num_keep = len(statements) - int(np.round(len(statements) / 5)) #Currently eliminates statements with highest 20% average distance
	drop = np.argsort(means)[num_keep:]
	keep = np.argsort(means)[:num_keep]
	dropped_statements = statements.ix[list(drop)]
	statements = statements.ix[list(keep)]
	s_emb = s_emb[keep,:]
	d = pdist(s_emb)

	#Number of elaborations per statement
	num = list(statements.elab_count)

	#Define threshold at which we will cutoff statements and save the smallest distance
	cutoff = np.mean(d) - sd_threshold * np.std(d)
	min_dist = np.min(d)

	#Extra variables to make sure the function doesn't include/exclude too many ideas:
	#Follows formula similar to idea pool limitation that aproaches the idea pool limit as we get halfway through a study
	#Also checks if we are halfway through the study. If halfway we will return the most dissimlar ideas up to the idea pool limit.
	current_ips = np.round((1 + np.sqrt(1 + study_data[1]*1056/25)) / 2) 
	current_ips = np.round(2 * study_data[0] * study_data[1] / study_data[2]) # 2 * study IPS * % who have completed study
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
		
		#Get statements with the minimum distance
		index = np.where(d == min_dist)[0][0]
		x1 = int(indices[0][index])
		x2 = int(indices[1][index])
		indices1 = np.where(((indices[0]==x1) & (indices[1]!=x2)) | ((indices[1]==x1) & (indices[0]!=x2)))[0]
		indices2 = np.where(((indices[0]==x2) & (indices[1]!=x1)) | ((indices[1]==x2) & (indices[0]!=x1)))[0]

		#First statement has more elaborations
		if num[x1] > num[x2] and x2 not in remove:
			remove.append(x2)
			d[indices2]=max(d)
		#Second statement has more elaborations
		if num[x1] < num[x2] and x1 not in remove:
			remove.append(x1)
			d[indices1]=max(d)

		#Both Statements have the same number of elaborations
		if num[x1] == num[x2]:
			d1 = np.min(d[indices1])
			d2 = np.min(d[indices2])
			if d1 >= d2 and x2 not in remove:
				remove.append(x2)
				d[indices2]=max(d)
			if d1 < d2 and x1 not in remove:
				remove.append(x1)
				d[indices1]=max(d)

		#Update info (set distance between the 2 statements as the max so that the 2 statements aren't compared again)
		d[index]=max(d)
		min_dist = np.min(d)
		nStatements = nStatements - 1
	
	#Return results
	mask = np.zeros(len(statements),dtype=bool) 
	mask[remove] = True
	statements['isNotRepresentative'] = mask
	dropped_statements['isNotRepresentative'] = True
	statements = pd.concat((statements, dropped_statements))
	return statements[['id','isNotRepresentative']]