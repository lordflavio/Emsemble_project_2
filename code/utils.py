###############################################################################
from functools import partial
from math import sqrt
from copy import deepcopy
import operator, sys

import json
import pandas as pd
import numpy as np
from scipy.io import arff

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
import GA

from prefit_voting_classifier import PrefitVotingClassifier


def load_experiment_configuration():
	STRATEGY_PERCENTAGE = 0.5
	N_JOBS = -1
	PRUNNING_CLUSTERS = 10

	config = {
	"num_folds": 10,
	"pool_size": 100,
	"kdn": 5,
	"strategy_percentage": STRATEGY_PERCENTAGE,
	"validation_hardnesses": _create_validation_hardnesses(threshold = 0.5),
	"base_classifier": partial(Perceptron, max_iter = 20, tol = 0.001,
		                       penalty = None, n_jobs = N_JOBS),
	"generation_strategy": partial(BaggingClassifier, 
		                           max_samples = STRATEGY_PERCENTAGE,
		                           n_jobs = -1),
	"pruning_strategies": _create_pruning_strategies(PRUNNING_CLUSTERS,
		                                             N_JOBS),
	"diversity_measures": _create_diversity_measures()
	}

	return config

def _create_validation_hardnesses(threshold):
	return [("None", partial(operator.gt, 2)), 
	        ("Hard", partial(operator.lt, threshold)), 
	        ("Easy", partial(operator.gt, threshold))]

def _create_diversity_measures():
	return [partial(_disagreement_measure), partial(_double_fault_measure)]

def _create_pruning_strategies(num_clusters, n_jobs):
	# return [("Best First", partial(_best_first_pruning)),
	#         ("K Best Means", partial(_k_best_means_pruning, k=num_clusters,
	#         	                     n_jobs = n_jobs)),
	# 		("GASEN", partial(_best_gasen_pruning))]
	return [("GASEN", partial(_best_gasen_pruning))]

def calculate_pool_diversity(measure_fn, pool, instances, gold_labels, pool_size):
	if pool_size <= 1:
		return 0

	error_vectors = [_get_error_vector(estim, instances, gold_labels) for estim \
	                 in pool.estimators]

	summed_diversity = 0
	for i in xrange(pool_size-1):
		for j in xrange(i+1, pool_size):
			matrix = _create_agreement_matrix(error_vectors[i], error_vectors[j])
			summed_diversity += measure_fn(matrix)

	return _average_pairs_diversity(summed_diversity, pool_size)

def _average_pairs_diversity(summed_diversity, pool_size):
	return (2*summed_diversity)/(pool_size*(pool_size-1))

def _get_error_vector(clf, instances, gold_labels):
	predicted = clf.predict(instances)
	return [predicted[i]==gold_labels[i] for i in xrange(len(gold_labels))]

def _create_agreement_matrix(di_vector, dj_vector):
	d00 = _get_agreement_matrix_position(False, False, di_vector, dj_vector)
	d01 = _get_agreement_matrix_position(False, True, di_vector, dj_vector)
	d10 = _get_agreement_matrix_position(True, False, di_vector, dj_vector)
	d11 = _get_agreement_matrix_position(True, True, di_vector, dj_vector)
	return [[d00, d01], [d10, d11]]

def _get_agreement_matrix_position(err_i, err_j, vec_i, vec_j):
	xrg = xrange(len(vec_i))
	agreement_vector = [vec_i[p] == err_i and vec_j[p] == err_j for p in xrg]
	filtered = filter(lambda b: b == True, agreement_vector)
	return len(filtered)

def _disagreement_measure(agreement_matrix):
	num = agreement_matrix[0][1] + agreement_matrix[1][0]
	den = sum(agreement_matrix[0]) + sum(agreement_matrix[1])
	return float(num)/den

def _double_fault_measure(agreement_matrix):
	num = agreement_matrix[0][0]
	den = sum(agreement_matrix[0]) + sum(agreement_matrix[1])
	return float(num)/den

def _find_k_neighbours(distances, k):
	
	matrix_neighbours = []
	for i in xrange(len(distances)):
		
		cur_neighbours = set()
		while len(cur_neighbours) < k:
			min_ix = np.argmin(distances[i])
			distances[i, min_ix] = sys.float_info.max

			if min_ix != i:
				cur_neighbours.add(min_ix)

		matrix_neighbours.append(list(cur_neighbours))

	return matrix_neighbours

def _calculate_kdn_hardness(instances, gold_labels, k):
	distances = euclidean_distances(instances, instances)
	neighbours = _find_k_neighbours(distances, k)

	hards = []
	for i in xrange(len(neighbours)):
		fixed_label = gold_labels[i]
		k_labels = gold_labels[neighbours[i]]
		dn = sum(map(lambda label: label != fixed_label, k_labels))
		hards.append(float(dn)/k)

	return hards

def select_validation_set(instances, labels, operator, k):
	hards = _calculate_kdn_hardness(instances, labels, k)
	filtered_triples = _filter_based_hardness(instances, labels, hards, operator)
	validation_instances = [t[0] for t in filtered_triples]
	validation_labels = [t[1] for t in filtered_triples]
	return np.array(validation_instances), validation_labels

def _filter_based_hardness(instances, labels, hards, op):
	triples = [(instances[i], labels[i], hards[i]) for i in xrange(len(hards))]
	return filter(lambda t: op(t[2]), triples)

def _order_clfs(pool_clf, validation_instances, validation_labels):
	clfs = pool_clf.estimators_
	clfs_feats = pool_clf.estimators_features_
	predictions = [clf.predict(validation_instances) for clf in clfs]
	errors = [_error_score(validation_labels, predicted_labels) for predicted_labels in predictions]
	triples = [(clfs[i], clfs_feats[i], errors[i]) for i in xrange(len(errors))]
	return sorted(triples, key=lambda t: t[2])

def _find_k_clusters(pool_clf, k, n_jobs):
	clfs = pool_clf.estimators_
	clfs_feats = pool_clf.estimators_features_

	pool_weights = [clf.coef_[0] for clf in clfs]
	k_means = KMeans(n_clusters = k, n_jobs = n_jobs)
	clusters_labels = k_means.fit_predict(pool_weights)

	clusters = {cluster_label: [] for cluster_label in clusters_labels}
	for i in xrange(len(clfs)):
		cluster = clusters_labels[i]
		clusters[cluster].append((clfs[i], clfs_feats[i]))

	return clusters

def _find_best_per_cluster(clusters, validation_instances, validation_labels):
	best_k_clf = []
	best_k_feats = []

	for cluster, clfs_tuples in clusters.iteritems():
		cur_best_clf = None
		cur_best_feats = None
		cur_best_error = 100

		for clf_tuple in clfs_tuples:
			clf = clf_tuple[0]
			predicted = clf.predict(validation_instances)
			error = _error_score(validation_labels, predicted)

			if error < cur_best_error:
				cur_best_error = error
				cur_best_clf = clf
				cur_best_feats = clf_tuple[1]

		best_k_clf.append(cur_best_clf)
		best_k_feats.append(cur_best_feats)

	return _get_voting_clf(best_k_clf, best_k_feats)

def _k_best_means_pruning(pool_clf, validation_instances, validation_labels, k, n_jobs):
	clusters = _find_k_clusters(pool_clf, k, n_jobs)
	return _find_best_per_cluster(clusters, validation_instances, validation_labels)

def _find_best_first(triples, validation_instances, validation_labels):
	best_ensemble_error = 100
	best_ensemble = None

	cur_clfs = []
	cur_feats = []
	for triple in triples:
		clf, clf_feat, error = triple
		cur_clfs.append(clf)
		cur_feats.append(clf_feat)
		ensemble = _get_voting_clf(cur_clfs, cur_feats)
		predicted = ensemble.predict(validation_instances)
		error = _error_score(validation_labels, predicted)

		if error < best_ensemble_error:
			best_ensemble_error = error
			best_ensemble = ensemble

	return best_ensemble

def _best_first_pruning(pool_clf, validation_instances, validation_labels):
	ordered_triples = _order_clfs(pool_clf, validation_instances, 
		                          validation_labels)

	return _find_best_first(ordered_triples, validation_instances, 
		                    validation_labels)

def weighted_ensemble(weights, clfs, validation_instances):
    # Assigning empty array to store 2D array of model predictio
	predictions = np.empty(shape=(validation_instances.shape[0], weights.shape[0]))
    # Loop through all model
	for i in range(len(clfs)):
		# predictions.append(clfs[i].predict(validation_instances))
		predictions[:,i] = clfs[i].predict(validation_instances)
	
	# voto majoritario 
	predictions_v = []
	for (j, p) in enumerate(predictions):
		pred_temp = np.bincount(p.astype(int), weights)
		predictions_v.append(np.argmax(pred_temp))

	predictions_v = np.array(predictions_v)
	return predictions_v

def ensemble_fitness(weights, clfs, validation_instances, validation_labels, metrica):
	import numpy as np
	import sklearn
	fitness = []
	for i in range(len(weights)):
		predictions_voted = weighted_ensemble(weights[i-1], clfs, validation_instances)
         
        # Setting output fitness value (erro)
		if metrica == "acc":
			ensembleFit = 1 - accuracy_score(predictions_voted, validation_labels)
		else:
			print("nao escolheu metrica")
		fitness.append(ensembleFit)

    #Returning fitness value to minimise
	return fitness

def _gasen(pool_clf, validation_instances, validation_labels,**kwargs):
	UPPER_BOUND = 1
	LOWER_BOUND = 0
	clfs = pool_clf.estimators_
	clfs_feats = pool_clf.estimators_features_

	#Create objective function
	objective_function = lambda w: ensemble_fitness(w, clfs, validation_instances, validation_labels, "acc")

	#Set Genetic Algorithm parameters
	sol_per_pop = 100 # testando 100 
	num_parents_mating = 50 # metade cruzando

	# Defining population size
	pop_size = (sol_per_pop, len(clfs))
	# Creating the initial population
	new_population = np.random.uniform(low=0, high=1, size=pop_size)

	num_geracoes = 100
	for generation in range(num_geracoes):
		print("Geracao: ", generation)
		# Mensurando o fitness (aptidao) de cada cromossomo da pop
		fitness = GA.cal_pop_fitness(objective_function, new_population)

		# Selecting the best parents in the population for mating
		parents = GA.select_mating_pool(new_population, fitness, num_parents_mating)

		# Generating next generation using crossover
		offspring_crossover = GA.crossover(parents, offspring_size=(pop_size[0]-parents.shape[0], len(clfs)))

		# Adding some variations to the offspring using mutation
		offspring_mutation = GA.mutation(offspring_crossover)

		# Creating the new population based on the parents and offspring
		new_population[0:parents.shape[0], :] = parents
		new_population[parents.shape[0]:, :] = offspring_mutation

	# Get the best solution after all generations
	fitness = GA.cal_pop_fitness(objective_function, new_population)
	# Return the index of that solution and corresponding best fitness
	best_match_idx = np.where(fitness == np.min(fitness))
	# print(best_match_idx)
	# print(fitness)

	# retorna os pesos
	return (new_population[best_match_idx], clfs, clfs_feats)

def _find_best_gasen(pool_clf, clf_feats, gasen_weights, validation_instances, validation_labels):
	return _get_voting_clf(pool_clf, clf_feats, gasen_weights)
	
def _best_gasen_pruning(pool_clf, validation_instances, validation_labels):
	gasen_weights, gasen_clfs, gasen_clfs_feats = _gasen(pool_clf, validation_instances, validation_labels)
	print('gasen_weights shape:', gasen_weights.shape)
	# entrar apenas a primeira linha pq esta retornando toda a pop
	return _find_best_gasen(gasen_clfs, gasen_clfs_feats, gasen_weights[0], validation_instances, validation_labels)
 

def _get_voting_clf(base_clfs, clfs_feats, weights=None):
	pool_size = len(base_clfs)
	clfs_tuples = [(str(i), base_clfs[i]) for i in xrange(pool_size)]
	if weights is None:
		return PrefitVotingClassifier(clfs_tuples, clfs_feats, voting = 'hard', weights=None)
	else:
		return PrefitVotingClassifier(clfs_tuples, clfs_feats, voting ='hard', weights=weights)

def get_voting_pool_size(voting_pool):
	return len(voting_pool.estimators)

def load_datasets_filenames():
    # return ['pc1', 'kc2']
	return ['jm1']
	# return ['jm1', 'kc1']
	# filenames = ["cm1", "jm1"]
	# return filenames

def load_dataset(set_filename):
	SET_PATH = "../data/"
	FILETYPE = ".arff"
	full_filepath = SET_PATH + set_filename + FILETYPE

	data, _ = arff.loadarff(full_filepath)

	dataframe = pd.DataFrame(data)
	dataframe.dropna(inplace=True)

	# gold_labels = pd.DataFrame(dataframe["defects"])
	# instances = dataframe.drop(columns = "defects")

	# if set_filename == "kc2":
	# 	gold_labels = pd.DataFrame(dataframe["problems"])
	# 	gold_labels["problems"] = gold_labels["problems"].apply(lambda x: x.decode())
	# 	instances = dataframe.drop(columns = "problems")
	gold_labels = pd.DataFrame(dataframe["defects"])
	gold_labels["defects"] = gold_labels['defects'].apply(lambda x: x.decode())
	instances = dataframe.drop(columns = "defects")
		
	return instances, gold_labels

def save_predictions(data):
	with open('../predictions/all_predictions.json', 'w') as outfile:
		json.dump(data, outfile)

def load_predictions_data():
	with open('../predictions/all_predictions.json', 'r') as outfile:
		return json.load(outfile)

def _error_score(gold_labels, predicted_labels):
	return 1 - accuracy_score(gold_labels, predicted_labels)

def _g1_score(gold_labels, predicted_labels, average):
	precision = precision_score(gold_labels, predicted_labels, average=average)
	recall = recall_score(gold_labels, predicted_labels, average=average)
	return sqrt(precision*recall)

def _calculate_metrics(gold_labels, data):

	predicted_labels = data[0]
	final_pool_size = data[1]
	disagreement = data[2]
	double_fault = data[3]

	metrics = {}
	# print('tipo gold_labels: ', type(gold_labels))
	# print('gold_labels[0:5]: ', gold_labels[0:5])
	# print('tipo predicted_labels: ', type(predicted_labels))
	# print('predicted_labels[0:5]: ', predicted_labels[0:5])

	metrics["auc_roc"] = roc_auc_score(gold_labels, predicted_labels, average='macro')
	metrics["g1"] = _g1_score(gold_labels, predicted_labels, average='macro')
	metrics["f1"] = f1_score(gold_labels, predicted_labels, average='macro')
	metrics["acc"] = accuracy_score(gold_labels, predicted_labels)
	metrics["pool"] = final_pool_size
	metrics["disagr"] = disagreement
	metrics["2xfault"] = double_fault

	return metrics

def _check_create_dict(given_dict, new_key):
	if new_key not in given_dict.keys():
		given_dict[new_key] = {}

def generate_metrics(predictions_dict):
	metrics = {}

	for set_name, set_dict in predictions_dict.iteritems():
		metrics[set_name] = {}

		for fold, fold_dict in set_dict.iteritems():

			gold_labels = fold_dict["gold_labels"]
			del fold_dict["gold_labels"]

			for hardness_type, filter_dict in fold_dict.iteritems():
				_check_create_dict(metrics[set_name], hardness_type)

				for strategy, data_arr in filter_dict.iteritems():

					metrics_str = metrics[set_name][hardness_type]

					fold_metrics = _calculate_metrics(gold_labels, data_arr)

					if strategy not in metrics_str.keys():
					    metrics_str[strategy] = [fold_metrics]
					else:
						metrics_str[strategy].append(fold_metrics)

	return metrics

def _summarize_metrics_folds(metrics_folds):
	summary = {}
	metric_names = metrics_folds[0].keys()

	for metric_name in metric_names:
		scores = [metrics_folds[i][metric_name] for i in xrange(len(metrics_folds))]
		summary[metric_name] = [np.mean(scores), np.std(scores)]

	return summary

def summarize_metrics_folds(metrics_dict):

	summary = deepcopy(metrics_dict)

	for set_name, set_dict in metrics_dict.iteritems():
		for hardness_type, filter_dict in set_dict.iteritems():
			for strategy, metrics_folds in filter_dict.iteritems():
				cur_summary = _summarize_metrics_folds(metrics_folds)
				summary[set_name][hardness_type][strategy] = cur_summary

	return summary

def pandanize_summary(summary):

	df = pd.DataFrame(columns = ['set', 'hardness', 'strategy',
	                  'mean_auc_roc', 'std_auc_roc', 'mean_acc', 'std_acc',
	                  'mean_f1', 'std_f1', 'mean_g1', 'std_g1',
	                  'mean_pool', 'std_pool', 'mean_2xfault',
	                  'std_2xfault', 'mean_disagr', 'std_disagr'])

	for set_name, set_dict in summary.iteritems():
		for hardness_type, filter_dict in set_dict.iteritems():
			for strategy, summary_folds in filter_dict.iteritems():
				df_folds = pd.DataFrame(_unfilled_row(3, 14),
					                    columns = df.columns)
				_fill_dataframe_folds(df_folds, summary_folds, set_name,
					                  hardness_type, strategy)
				df = df.append(df_folds)

	return df.reset_index(drop = True)

def _unfilled_row(str_columns, float_columns):
	row = [" " for i in xrange(str_columns)]
	row.extend([0.0 for j in xrange(float_columns)])
	return [row]

def _fill_dataframe_folds(df, summary, set_name, hardness, strategy):
	df.at[0, "set"] = set_name
	df.at[0, "hardness"] = hardness
	df.at[0, "strategy"] = strategy
	return _fill_dataframe_metrics(df, summary)

def _fill_dataframe_metrics(df, summary):
	for key, metrics in summary.iteritems():
		df.at[0, "mean_" + key] = metrics[0]
		df.at[0, "std_" + key] = metrics[1]
	return df

def save_pandas_summary(df):
	pd.to_pickle(df, '../metrics/metrics_summary.pkl')

def read_pandas_summary():
	return pd.read_pickle('../metrics/metrics_summary.pkl')

def separate_pandas_summary(df, separate_sets):
	dfs = []

	if separate_sets == True:
		sets = df["set"].unique()
		for set_name in sets:
			dfs.append(df.loc[df["set"]==set_name])
	else:
		dfs.append(df)

	return dfs

def write_comparison(dfs, focus_columns, filename):

	with open('../comparisons/'+ filename + '.txt', "w") as outfile:
		for df_set in dfs:
			if len(dfs) == 1:
				outfile.write("\n\nDATASET: Mixed\n")
			else:
				outfile.write("\n\nDATASET: " + df_set.iat[0,0] + "\n")
			outfile.write("Mean of metrics\n")
			outfile.write(df_set.groupby(by=focus_columns).mean().to_string())
			outfile.write("\n\nStd of metrics\n")
			outfile.write(df_set.groupby(by=focus_columns).std().to_string())
			outfile.write("\n")
			outfile.write("-------------------------------------------------")

def bool_str(s):

    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')

    return s == 'True'