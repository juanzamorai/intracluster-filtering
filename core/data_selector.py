import numpy as np
from sklearn.mixture import GaussianMixture as GMM
from collections import Counter
from sklearn.decomposition import PCA

class DataSelector:
    def __init__(self, X_tr, y_tr, epochs_to_start_filter, update_period_in_epochs,n_components_pca, filter_threshold=0.3):
        self.X_tr = X_tr
        self.y_tr = y_tr
        self.out_clases_number = y_tr.shape[1]
        self.epochs_to_start_filter = epochs_to_start_filter
        self.update_period_in_epochs = update_period_in_epochs
        self.filtered_index = list(range(X_tr.shape[0]))
        self.filter_threshold = filter_threshold
        self.pca_applied = False
        self.pca = PCA(n_components=n_components_pca)  # Define PCA con el número de componentes dado por el usuario


    def check_filter_update_criteria(self, epoch):
        return (epoch >= self.epochs_to_start_filter and 
                (epoch - self.epochs_to_start_filter) % self.update_period_in_epochs == 0)

    def get_train_data(self, epoch, model, outs_posibilities):
        if self.check_filter_update_criteria(epoch):
            inspector_layer_out = model.inspector_out(self.X_tr)
            if not self.pca_applied and epoch >= 0.7 * self.update_period_in_epochs:
                
                inspector_layer_out = self.pca.fit_transform(inspector_layer_out)
                print("PCA HECHO")
                self.pca_applied = True
                
            
            percentage_of_pertenence, clusterized_outs_proba, prob_correct_class_cluster = \
                from_inspector_results_to_gaussian_mixture_probabilities(inspector_layer_out, self.y_tr, outs_posibilities)
            
            if np.array(list(percentage_of_pertenence.values())).min() < 0.7:
                print("Warning: pertenence of a class for a cluster is lower than 70% then no filtering will be done.")
                return self.return_filtered_data()
            
            self.filtered_index = np.where(prob_correct_class_cluster > self.filter_threshold)[0]
            
            # Actualizar los datos de entrenamiento con los datos filtrados
            self.X_tr = self.X_tr[self.filtered_index]
            self.y_tr = self.y_tr[self.filtered_index]
        
        return self.return_filtered_data()

    def return_filtered_data(self):
        return self.X_tr, self.y_tr

    def get_removed_data(self):
        original_indices = np.arange(self.X_tr.shape[0])
        return list(set(original_indices).difference(set(self.filtered_index)))

def from_inspector_results_to_gaussian_mixture_probabilities(inspector_layer_out, y_tr, outs_posibilities):
    gmm = GMM(n_components=y_tr.shape[1]).fit(inspector_layer_out)
    clusterized_outs_proba = gmm.predict_proba(inspector_layer_out)
    clusterized_outs = clusterized_outs_proba.argmax(axis=1)
    class_gmm_to_real_class = {}
    percentage_of_pertenence = {}
    for class_it in outs_posibilities:
        mask = y_tr.argmax(axis=1) == class_it
        most_common = Counter(clusterized_outs[mask]).most_common(1)
        class_gmm_to_real_class[most_common[0][0]] = class_it
        percentage_of_pertenence[class_it] = most_common[0][1] / mask.sum()

    if set(class_gmm_to_real_class.values()) == set(outs_posibilities):
        print(class_gmm_to_real_class)
        assert True, "All clases have just one cluster associated"

    clusterized_outs_proba = clusterized_outs_proba[:, list(class_gmm_to_real_class.keys())]
    prob_correct_class_cluster = clusterized_outs_proba[
        np.arange(len(clusterized_outs_proba)), y_tr.argmax(axis=1)]
    
    return percentage_of_pertenence, clusterized_outs_proba, prob_correct_class_cluster
