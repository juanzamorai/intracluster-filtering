import numpy as np
from sklearn.mixture import GaussianMixture as GMM
from collections import Counter
from sklearn.decomposition import PCA
import tensorflow as tf
import warnings

class DataSelector:
    def __init__(self, X_tr, y_tr, epochs_to_start_filter, update_period_in_epochs, filter_percentile=0.3, random_state=None, train_with_outliers=False):
        self.X_tr = X_tr
        self.y_tr = y_tr.numpy() if isinstance(y_tr, tf.Tensor) else y_tr
        self.out_clases_number = y_tr.shape[1]
        self.epochs_to_start_filter = epochs_to_start_filter
        self.update_period_in_epochs = update_period_in_epochs
        self.filtered_index = list(range(X_tr.shape[0]))
        self.filter_percentile = filter_percentile
        self.random_state = random_state
        self.train_with_outliers = train_with_outliers
        self.removed_data_indices = []
        self.original_indices = np.arange(X_tr.shape[0])  # Original indices
        self.all_removed_indices = []  # List to store all removed indexes
        self.previous_X_tr = X_tr  # Save a copy of the original data
        self.previous_y_tr = y_tr  # Save a copy of the original labels
        
    def check_filter_update_criteria(self, epoch):
        return (epoch >= self.epochs_to_start_filter and 
                (epoch - self.epochs_to_start_filter) % self.update_period_in_epochs == 0)

    # Function that applies PCA to the inspector_layer_out layer and returns the transformed data and the number of components
    def apply_pca(self, inspector_layer_out, explained_variance=None, n_components=None):
        if explained_variance is not None:
            pca = PCA(n_components=explained_variance)
        elif n_components is not None:
            pca = PCA(n_components=n_components)
        else:
            raise ValueError("You must provide either explained_variance or n_components")

        transformed_out = pca.fit_transform(inspector_layer_out)
        n_components = transformed_out.shape[1]
        
        # Make sure the number of components is not less than 2
        if n_components < 2:
            n_components = 2
            pca = PCA(n_components=n_components)
            transformed_out = pca.fit_transform(inspector_layer_out)
        
        if explained_variance is not None:
            print(f"PCA done: retained {explained_variance*100}% of the variance with {n_components} components")
        else:
            print(f"PCA done with {n_components} components")
        
        return transformed_out, n_components
       
    def get_train_data(self, epoch, model, outs_posibilities, explained_variance=None, n_components=None):
        if self.check_filter_update_criteria(epoch):
            inspector_layer_out = model.inspector_out(self.X_tr).numpy()
            inspector_layer_out, n_components = self.apply_pca(inspector_layer_out, explained_variance, n_components)
            print(f"PCA done with {n_components} components")

            gmm = GMM(n_components=self.y_tr.shape[1], random_state=self.random_state).fit(inspector_layer_out)
            clusterized_outs_proba = gmm.predict_proba(inspector_layer_out)
            clusterized_outs = clusterized_outs_proba.argmax(axis=1)

            class_gmm_to_real_class = {}
            percentage_of_pertenence = {}
            for class_it in outs_posibilities:
                mask = self.y_tr.argmax(axis=1) == class_it
                #most_common = Counter(clusterized_outs[mask]).most_common(1)
                #class_gmm_to_real_class[most_common[0][0]] = class_it
                #percentage_of_pertenence[class_it] = most_common[0][1] / mask.sum()
                if not mask.any():
                    print(f"Warning: no data for class {class_it}, filtering will not be done")
                    return self.previous_X_tr, self.previous_y_tr, self.original_indices, self.all_removed_indices
                most_common = Counter(clusterized_outs[mask]).most_common(1)
                
                if most_common:  # Check if most_common is not empty
                    class_gmm_to_real_class[most_common[0][0]] = class_it
                    percentage_of_pertenence[class_it] = most_common[0][1] / mask.sum()
                else:
                    # Handle the case where there is no most common element
                    print(f"Warning: no common elements found for class {class_it}")
                    return self.previous_X_tr, self.previous_y_tr, self.original_indices, self.all_removed_indices
            
            
            size_set_train = self.X_tr.shape[0]
            print(f"Size of the training set: {size_set_train}")

            if set(class_gmm_to_real_class.values()) != set(outs_posibilities):
                print("Warning: there are classes without a cluster associated")
                print("Warning: the filtering was not done")
                return self.previous_X_tr, self.previous_y_tr, self.original_indices, self.all_removed_indices

            print("All classes have just one cluster associated")

            clusterized_outs_proba = clusterized_outs_proba[:, list(class_gmm_to_real_class.keys())]
            prob_correct_class_cluster = clusterized_outs_proba[np.arange(len(clusterized_outs_proba)), self.y_tr.argmax(axis=1)]
            
            filtered_indices_per_class = []
            for class_it in outs_posibilities:
                class_mask = self.y_tr.argmax(axis=1) == class_it
                class_probs = np.round(prob_correct_class_cluster[class_mask], 2)  # Round to 2 decimals
                threshold = np.percentile(class_probs, self.filter_percentile * 100)
                threshold = round(threshold, 2)  # Round to 2 decimals

                # Find the probabilities less than the threshold and their indices
                filtered_probs_below_threshold = class_probs[class_probs < threshold]
                
                # Print the results
                print(f'Number of probabilities below the threshold {threshold} for the actual class {class_it}: ', len(filtered_probs_below_threshold))

                # Update filtered_indices_per_class with the indices corresponding to the probabilities greater than or equal to the threshold
                indices_above_threshold = np.where(class_mask)[0][class_probs >= threshold]
                filtered_indices_per_class.append(indices_above_threshold)

            # Concatenate all filtered indexes
            self.filtered_index = np.concatenate(filtered_indices_per_class)
            original_indices = np.arange(self.X_tr.shape[0], dtype=int)
            removed_data_indices = list(set(original_indices).difference(set(self.filtered_index)))
            removed_original_indices = self.original_indices[removed_data_indices]  # Guardar los índices originales

            self.removed_data_indices = removed_original_indices.tolist()  # Convertir a lista
            self.all_removed_indices.extend(removed_original_indices.tolist())  # Agregar índices removidos a la lista general

            print(f"Remove data: {removed_original_indices}")

            # Check if no outliers have been identified
            if len(self.filtered_index) == 0:
                print("No outliers identified, using previous filtered dataset")
                return self.previous_X_tr, self.previous_y_tr, self.original_indices, self.all_removed_indices
            
            # Update training data with filtered data
            filtered_X_tr = tf.gather(self.X_tr, self.filtered_index)
            filtered_y_tr = tf.gather(self.y_tr, self.filtered_index).numpy()  # Convertir a NumPy array
            filtered_original_indices = self.original_indices[self.filtered_index]  # Actualizar el mapeo de índices originales

            size_set_post = filtered_X_tr.shape[0]

            print("Data has been filtered")
            print(f"Size of data removed: {size_set_train - size_set_post}")

            if self.train_with_outliers:
                # Use the original indexes when working with outliers
                removed_data = tf.gather(self.X_tr, np.array(removed_data_indices, dtype=int))
                removed_labels = tf.gather(self.y_tr, np.array(removed_data_indices, dtype=int))
                removed_indices = tf.gather(self.original_indices, np.array(removed_data_indices, dtype=int))
                
                num_removed = len(removed_data_indices)
                
                # Check if the number of data removed is zero
                if num_removed == 0:
                    print("No data to remove for outliers, using previous filtered dataset")
                    return self.previous_X_tr, self.previous_y_tr, self.original_indices, self.all_removed_indices

                # Select an equal number of indexes at random, excluding those already removed
                all_indices = set(range(len(self.X_tr)))
                excluded_indices = set(removed_data_indices)
                available_indices = list(all_indices - excluded_indices)
                random_indices = np.random.choice(available_indices, num_removed, replace=False)
    
                # Recovers random data and labels
                random_data = tf.gather(self.X_tr, random_indices)
                random_labels = tf.gather(self.y_tr, random_indices)
                random_original_indices = np.array(self.original_indices)[random_indices]

                # Create new sets by combining the removed data and random data
                filtered_X_tr = np.concatenate((removed_data, random_data), axis=0)
                filtered_y_tr = np.concatenate((removed_labels, random_labels), axis=0)
                filtered_original_indices = np.concatenate((removed_indices, random_original_indices), axis=0)
                print(f"Training with outliers: added {num_removed} removed data points and {num_removed} random points")

            self.X_tr = filtered_X_tr
            self.y_tr = filtered_y_tr
            self.original_indices = filtered_original_indices

            # Save the current filtered data set
            self.previous_X_tr = self.X_tr
            self.previous_y_tr = self.y_tr

        return self.return_filtered_data()
    
    def return_filtered_data(self):
        return self.X_tr, self.y_tr, self.original_indices, self.all_removed_indices
