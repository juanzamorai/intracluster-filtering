import numpy as np
from sklearn.mixture import GaussianMixture as GMM
from collections import Counter
from sklearn.decomposition import PCA
import tensorflow as tf

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
        self.original_indices = np.arange(X_tr.shape[0])
        self.all_removed_indices = []  # Lista para almacenar todos los índices removidos
        self.pca_applied = False

    def check_filter_update_criteria(self, epoch):
        return (epoch >= self.epochs_to_start_filter and 
                (epoch - self.epochs_to_start_filter) % self.update_period_in_epochs == 0)

    def apply_pca(self, inspector_layer_out, n_clusters, explained_variance=0.95):
        pca = PCA(n_components=explained_variance)
        transformed_out = pca.fit_transform(inspector_layer_out)
        n_components = transformed_out.shape[1]
        print(f"Initial PCA components: {n_components} for {explained_variance*100}% variance explained")

        while n_components <= inspector_layer_out.shape[1]:
            gmm = GMM(n_components=n_clusters, random_state=self.random_state)
            gmm.fit(transformed_out)
            if len(np.unique(gmm.predict(transformed_out))) == n_clusters:
                return transformed_out, n_components
            n_components += 1
            pca = PCA(n_components=n_components)
            transformed_out = pca.fit_transform(inspector_layer_out)
            print(f"Retrying with PCA components: {n_components}")

        raise ValueError("Could not find enough components to create the required number of clusters.")

    def get_train_data(self, epoch, model, outs_posibilities):
        if self.check_filter_update_criteria(epoch):
            inspector_layer_out = model.inspector_out(self.X_tr).numpy()
            inspector_layer_out, n_components = self.apply_pca(inspector_layer_out, len(outs_posibilities))
            print(f"PCA DONE with {n_components} components")

            gmm = GMM(n_components=self.y_tr.shape[1], random_state=self.random_state).fit(inspector_layer_out)
            clusterized_outs_proba = gmm.predict_proba(inspector_layer_out)
            clusterized_outs = clusterized_outs_proba.argmax(axis=1)

            class_gmm_to_real_class = {}
            percentage_of_pertenence = {}
            for class_it in outs_posibilities:
                mask = self.y_tr.argmax(axis=1) == class_it
                most_common = Counter(clusterized_outs[mask]).most_common(1)
                class_gmm_to_real_class[most_common[0][0]] = class_it
                percentage_of_pertenence[class_it] = most_common[0][1] / mask.sum()
            
            size_set_train = self.X_tr.shape[0]
            print(f"Size of the training set: {size_set_train}")

            if set(class_gmm_to_real_class.values()) != set(outs_posibilities):
                print("Warning: there are classes without a cluster associated")
                print("Warning: the filtering was not done")
                return self.return_filtered_data()

            print(f"GMM-associated clusters to real class: {class_gmm_to_real_class}")
            print("All clases have just one cluster associated")

            clusterized_outs_proba = clusterized_outs_proba[:, list(class_gmm_to_real_class.keys())]
            prob_correct_class_cluster = clusterized_outs_proba[np.arange(len(clusterized_outs_proba)), self.y_tr.argmax(axis=1)]
            
            self.filtered_index = np.array([], dtype=int)
            for class_it in outs_posibilities:
                class_mask = self.y_tr.argmax(axis=1) == class_it
                class_probs = prob_correct_class_cluster[class_mask]
                threshold = np.percentile(class_probs, self.filter_percentile * 100)
                self.filtered_index = np.concatenate((self.filtered_index, np.where(class_mask & (prob_correct_class_cluster >= threshold))[0]))

            original_indices = np.arange(self.X_tr.shape[0], dtype=int)
            removed_data_indices = list(set(original_indices).difference(set(self.filtered_index)))
            self.removed_data_indices = removed_data_indices  # Store removed indices for later use
            self.all_removed_indices.extend(removed_data_indices)  # Agregar índices removidos a la lista general

            print(f"Remove data: {removed_data_indices}")
            
            if not self.train_with_outliers:
                # Actualizar los datos de entrenamiento con los datos filtrados
                self.X_tr = tf.gather(self.X_tr, self.filtered_index)
                self.y_tr = tf.gather(self.y_tr, self.filtered_index).numpy()  # Convertir a NumPy array
                self.original_indices = tf.gather(self.original_indices, self.filtered_index).numpy()

                size_set_post = self.X_tr.shape[0]

                print("Data has been filtered")
                print(f"Size of data removed: {size_set_train - size_set_post}")

            if self.train_with_outliers and epoch >= self.epochs_to_start_filter:
                removed_data = tf.gather(self.X_tr, np.array(self.removed_data_indices, dtype=int))
                removed_labels = tf.gather(self.y_tr, np.array(self.removed_data_indices, dtype=int))
                removed_indices = tf.gather(self.original_indices, np.array(self.removed_data_indices, dtype=int))
                
                num_removed = len(self.removed_data_indices)
    
                # Selecciona un número igual de índices al azar, excluyendo los ya removidos.
                all_indices = set(range(len(self.X_tr)))
                excluded_indices = set(self.removed_data_indices)
                available_indices = list(all_indices - excluded_indices)
                random_indices = np.random.choice(available_indices, num_removed, replace=False)
        
                # Recupera los datos y etiquetas aleatorios.
                random_data = tf.gather(self.X_tr, random_indices)
                random_labels = tf.gather(self.y_tr, random_indices)
                random_original_indices = tf.gather(self.original_indices, random_indices)
    
                # Crea nuevos conjuntos combinando los datos removidos y los datos aleatorios.
                self.X_tr = np.concatenate((removed_data, random_data), axis=0)
                self.y_tr = np.concatenate((removed_labels, random_labels), axis=0)
                self.original_indices = np.concatenate((removed_indices, random_original_indices), axis=0)
                print(f"Training with outliers: added {num_removed} removed data points and {num_removed} random points")

        return self.return_filtered_data()

    def return_filtered_data(self, return_original_indices=False):
        if return_original_indices:
            return self.X_tr, self.y_tr, self.original_indices, self.all_removed_indices
        else:
            return self.X_tr, self.y_tr, self.all_removed_indices
