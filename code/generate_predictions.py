###############################################################################
import numpy as np
import random as rn

#DO NOT CHANGE THIS
np.random.seed(1478)
rn.seed(2264)
###################

from utils import load_datasets_filenames, load_experiment_configuration
from utils import load_dataset, save_predictions
from utils import select_validation_set
from utils import get_voting_pool_size, calculate_pool_diversity

from sklearn.model_selection import StratifiedKFold


if __name__ == "__main__":

	print("Step 1 - Loading configurations")

	datasets_filenames = load_datasets_filenames()
	config = load_experiment_configuration()
	predictions = {}
	exp = 1

	print("Step 2 - Starting experiment")

	for dataset_filename in datasets_filenames:
		print('Dataset: ', dataset_filename)
		instances, gold_labels = load_dataset(dataset_filename)
		skfold = StratifiedKFold(n_splits = config["num_folds"],
			                     shuffle = True)

		gold_labels = (gold_labels["defects"] == 'true').astype(int)
		# if dataset_filename == 'kc2':
		# 	gold_labels = (gold_labels["problems"] == 'yes').astype(int)
		# elif dataset_filename == 'pc1':
		# 	gold_labels = (gold_labels["defects"] == 'true').astype(int)
        
		predictions[dataset_filename] = {}

		# separacaoo dos dados em folds
		for fold, division in enumerate(skfold.split(X=instances, y=gold_labels), 1):
			train_idxs = division[0]
			test_idxs = division[1]
			train_instances = instances.iloc[train_idxs].values
			train_gold_labels = gold_labels.iloc[train_idxs].values.ravel()
			test_instances = instances.iloc[test_idxs].values
			test_gold_labels = gold_labels.iloc[test_idxs].values.ravel()

			predictions[dataset_filename][fold] = {}
			predictions[dataset_filename][fold]["gold_labels"] = test_gold_labels.tolist()

			# separacao dos dados em dificuldade das instancias com kDN
			for hardness_type, filter_func in config["validation_hardnesses"]:

				validation_instances, validation_gold_labels = select_validation_set(
					  train_instances, train_gold_labels, filter_func, config["kdn"])

				predictions[dataset_filename][fold][hardness_type] = {}
				subpredictions = predictions[dataset_filename][fold][hardness_type]

				base_clf = config["base_classifier"]()
				clf_pool = config["generation_strategy"](base_clf, config["pool_size"])
				clf_pool.fit(train_instances, train_gold_labels)
			
				# Aplicar estrategia de poda (no caso especifico foram duas)
				for strategy_name, pruning_strategy in config["pruning_strategies"]:

					pruned_pool = pruning_strategy(clf_pool, validation_instances,
						                            validation_gold_labels)

					pool_rem_size = get_voting_pool_size(pruned_pool)

					cur_predictions = pruned_pool.predict(test_instances)
					data_arr = [cur_predictions.astype(int).tolist(), pool_rem_size]
					
					# aplica as metricas de diversidade
					for measure in config["diversity_measures"]:
						measure_value = calculate_pool_diversity(measure,
							                                 pruned_pool,
							                        validation_instances,
							                      validation_gold_labels,
							                               pool_rem_size)

						data_arr.append(measure_value)

					subpredictions[strategy_name] = data_arr

					print("Experiment " + str(exp))
					exp+=1

	print("Step 2 - Finished experiment")

	print("Step 3 - Storing predictions")
	save_predictions(predictions)