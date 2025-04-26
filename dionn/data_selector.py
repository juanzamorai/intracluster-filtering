import numpy as np
from collections import Counter
from sklearn.mixture import GaussianMixture as GMM
from sklearn.decomposition import PCA
import tensorflow as tf
from scipy import linalg, special
import warnings

class StudentMixture:
    """Modelo de mezcla de distribuciones t de Student."""
    def __init__(self, n_components, covariance_type='full', tol=1e-3, reg_covar=1e-6, max_iter=100, random_state=None):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.random_state = np.random.RandomState(random_state)
        self.converged_ = False

    def _initialize_parameters(self, X):
        """Inicializa los parámetros del modelo de mezcla."""
        n_samples, n_features = X.shape
        self.weights_ = np.full(self.n_components, 1 / self.n_components)
        self.means_ = X[self.random_state.choice(n_samples, self.n_components, replace=False)]
        self.covariances_ = np.array([np.cov(X.T) + self.reg_covar * np.eye(n_features) for _ in range(self.n_components)])
        self.degrees_of_freedom_ = np.full(self.n_components, 10.0)

    def _estimate_log_prob(self, X):
        """Estima las probabilidades logarítmicas para cada componente."""
        n_samples, n_features = X.shape
        log_prob = np.empty((n_samples, self.n_components))

        for k in range(self.n_components):
            nu = self.degrees_of_freedom_[k]
            if nu <= 0:
                raise ValueError(f"Grados de libertad no válidos: {nu}. Deben ser mayores que 0.")
            diff = X - self.means_[k]
            precision = linalg.inv(self.covariances_[k])
            quad_form = np.sum(diff @ precision * diff, axis=1)
            log_det_cov = np.log(max(linalg.det(self.covariances_[k]), 1e-10))  # Evitar log(0) o negativos

            log_prob[:, k] = (
                special.gammaln((nu + n_features) / 2)
                - special.gammaln(nu / 2)
                - 0.5 * (n_features * np.log(nu * np.pi) + log_det_cov)
                - 0.5 * (nu + n_features) * np.log(1 + np.maximum(quad_form / nu, 1e-10))
            )
        return log_prob

    def _e_step(self, X):
        """Paso E."""
        log_prob = self._estimate_log_prob(X)
        weighted_log_prob = log_prob + np.log(self.weights_)
        log_prob_norm = special.logsumexp(weighted_log_prob, axis=1)
        log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
        return np.mean(log_prob_norm), log_resp

    def _m_step(self, X, log_resp):
        """Paso M."""
        resp = np.exp(log_resp)
        nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
        self.weights_ = nk / nk.sum()
        self.means_ = np.dot(resp.T, X) / nk[:, np.newaxis]
        self.covariances_ = self._estimate_covariances(X, resp, nk)
        self.degrees_of_freedom_ = self._update_degrees_of_freedom(X, resp, nk)

    def _estimate_covariances(self, X, resp, nk):
        """Estima las matrices de covarianza para cada componente."""
        n_samples, n_features = X.shape
        covariances = np.empty((self.n_components, n_features, n_features))
        for k in range(self.n_components):
            diff = X - self.means_[k]
            weighted_diff = resp[:, k][:, np.newaxis] * diff
            covariances[k] = np.dot(weighted_diff.T, diff) / nk[k] + self.reg_covar * np.eye(n_features)
        return covariances

    def _update_degrees_of_freedom(self, X, resp, nk):
        """Actualiza los grados de libertad para cada componente."""
        n_samples, n_features = X.shape
        new_dof = np.empty(self.n_components)
        for k in range(self.n_components):
            diff = X - self.means_[k]
            quad_form = np.sum(diff @ linalg.inv(self.covariances_[k]) * diff, axis=1)
            weighted_quad_form = np.dot(resp[:, k], quad_form)
            new_dof[k] = max(2 * (n_features + nk[k]) / (nk[k] - weighted_quad_form / (self.degrees_of_freedom_[k] + 2)), 1.0)
        return new_dof

    def fit(self, X):
        """Estimación de parámetros con el algoritmo EM."""
        self._initialize_parameters(X)
        for n_iter in range(self.max_iter):
            prev_weights = self.weights_.copy()
            log_prob_norm, log_resp = self._e_step(X)
            self._m_step(X, log_resp)
            if np.allclose(self.weights_, prev_weights, atol=self.tol):
                self.converged_ = True
                print(f"Convergencia alcanzada en la iteración {n_iter}.")
                break

    def predict_proba(self, X):
        """Calcula las probabilidades posteriores para cada componente."""
        _, log_resp = self._e_step(X)
        return np.exp(log_resp)

class DataSelector:
    def __init__(self, X_tr, y_tr, epochs_to_start_filter, update_period_in_epochs, filter_percentile=0.3, 
                 random_state=None, train_with_outliers=False, filter_model="gmm"):
        """
        Parámetros:
          - X_tr: datos de entrenamiento.
          - y_tr: etiquetas de entrenamiento (acepta tf.Tensor o numpy.array).
          - epochs_to_start_filter: epoch a partir del cual empezar el filtrado.
          - update_period_in_epochs: frecuencia (en epochs) de actualización del filtrado.
          - filter_percentile: percentil para definir el umbral de filtrado.
          - random_state: semilla para reproducibilidad.
          - train_with_outliers: flag para incluir outliers durante el entrenamiento.
          - filter_model: método de clustering a utilizar ('gmm' o 'smm'). Por defecto es 'gmm'.
        """
        self.X_tr = X_tr
        self.y_tr = y_tr.numpy() if isinstance(y_tr, tf.Tensor) else y_tr
        self.out_clases_number = self.y_tr.shape[1]
        self.epochs_to_start_filter = epochs_to_start_filter
        self.update_period_in_epochs = update_period_in_epochs
        self.filtered_index = list(range(X_tr.shape[0]))
        self.filter_percentile = filter_percentile
        self.random_state = random_state
        self.train_with_outliers = train_with_outliers
        self.removed_data_indices = []
        self.original_indices = np.arange(X_tr.shape[0])  # Índices originales
        self.all_removed_indices = []  # Lista de todos los índices removidos
        self.previous_X_tr = X_tr  # Copia de datos inicial
        self.previous_y_tr = y_tr
        self.inspector_layer_out = []  # Para almacenar la salida del inspector
        self.filter_model = filter_model.lower()  # 'gmm' o 'smm'

    def check_filter_update_criteria(self, epoch):
        """Determina si se debe actualizar el filtrado en este epoch."""
        return (epoch >= self.epochs_to_start_filter and 
                (epoch - self.epochs_to_start_filter) % self.update_period_in_epochs == 0)

    def apply_pca(self, inspector_layer_out, explained_variance=None, n_components=None):
        """Aplica PCA para reducir la dimensionalidad de la salida del inspector."""
        if explained_variance is not None:
            pca = PCA(n_components=explained_variance)
        elif n_components is not None:
            pca = PCA(n_components=n_components)
        else:
            raise ValueError("Debes proporcionar explained_variance o n_components.")
        transformed_out = pca.fit_transform(inspector_layer_out)
        n_components = transformed_out.shape[1]
        if n_components < 2:
            n_components = 2
            pca = PCA(n_components=n_components)
            transformed_out = pca.fit_transform(inspector_layer_out)
        if explained_variance is not None:
            print(f"PCA realizado: se retuvo el {explained_variance*100}% de la varianza con {n_components} componentes.")
        else:
            print(f"PCA realizado con {n_components} componentes.")
        return transformed_out, n_components

    def get_train_data(self, epoch, model, outs_posibilities, explained_variance=None, n_components=None):
        """
        Realiza el filtrado de datos usando el método de clustering seleccionado (GMM o SMM)
        y devuelve los datos filtrados junto con información adicional.
        """
        if self.check_filter_update_criteria(epoch):
            # Obtener la salida del "inspector" y reducir la dimensionalidad
            inspector_layer_out = model.inspector_out(self.X_tr).numpy()
            inspector_layer_out, n_components = self.apply_pca(inspector_layer_out, explained_variance, n_components)

            # Selección del método de clustering
            if self.filter_model == "gmm":
                print("Usando Gaussian Mixture Model (GMM) para el clustering...")
                gmm = GMM(n_components=self.y_tr.shape[1], random_state=self.random_state)
                gmm.fit(inspector_layer_out)
                clusterized_outs_proba = gmm.predict_proba(inspector_layer_out)
                clusterized_outs = clusterized_outs_proba.argmax(axis=1)
            elif self.filter_model == "smm":
                print("Usando Student Mixture Model (SMM) para el clustering...")
                smm = StudentMixture(
                    n_components=self.y_tr.shape[1],
                    random_state=self.random_state,
                    covariance_type="full",
                    max_iter=100,
                    tol=1e-3
                )
                smm.fit(inspector_layer_out)
                clusterized_outs_proba = smm.predict_proba(inspector_layer_out)
                clusterized_outs = clusterized_outs_proba.argmax(axis=1)
            else:
                raise ValueError("Método de filtrado inválido. Usa 'gmm' o 'smm'.")

            # Mapeo de clusters a clases reales
            class_cluster_to_real = {}
            percentage_of_pertenence = {}
            original_classes = list(outs_posibilities)
            for class_it in original_classes:
                mask = self.y_tr.argmax(axis=1) == class_it
                cluster_counts = Counter(clusterized_outs[mask])
                for cluster, current_count in cluster_counts.most_common():
                    current_percentage = current_count / mask.sum()
                    if cluster in class_cluster_to_real:
                        prev_class = class_cluster_to_real[cluster]
                        if current_percentage > percentage_of_pertenence[prev_class]:
                            # Actualiza el mapeo
                            class_cluster_to_real[cluster] = class_it
                            percentage_of_pertenence[class_it] = current_percentage
                            
                    else:
                        class_cluster_to_real[cluster] = class_it
                        percentage_of_pertenence[class_it] = current_percentage
                    break

            size_set_train = self.X_tr.shape[0]
            print(f"Tamaño del set de entrenamiento: {size_set_train}")

            if set(class_cluster_to_real.values()) != set(original_classes):
                print("Warning: existen clases sin un cluster asociado")
                print("Warning: no se realizó el filtrado")
                return self.previous_X_tr, self.previous_y_tr, self.original_indices, self.all_removed_indices, self.inspector_layer_out

            print("Cada clase tiene un único cluster asociado.")
            sorted_keys = [k for k, v in sorted(class_cluster_to_real.items(), key=lambda item: item[1])]
            clusterized_outs_proba = clusterized_outs_proba[:, sorted_keys]
            prob_correct_class_cluster = clusterized_outs_proba[np.arange(len(clusterized_outs_proba)), self.y_tr.argmax(axis=1)]

            # Filtrado por clases según umbral
            filtered_indices_per_class = []
            for class_it in original_classes:
                class_mask = self.y_tr.argmax(axis=1) == class_it
                class_probs = np.round(prob_correct_class_cluster[class_mask], 2)
                threshold = np.percentile(class_probs, self.filter_percentile * 100)
                if self.train_with_outliers:
                    threshold = round(threshold, 2)
                else:
                    threshold = round(min(threshold, 1 / self.out_clases_number), 2)
                print(f'Número de probabilidades por debajo del umbral {threshold} para la clase {class_it}: ',
                      len(class_probs[class_probs < threshold]))
                indices_above_threshold = np.where(class_mask)[0][class_probs >= threshold]
                filtered_indices_per_class.append(indices_above_threshold)

            self.filtered_index = np.concatenate(filtered_indices_per_class)
            original_indices = np.arange(self.X_tr.shape[0], dtype=int)
            removed_data_indices = list(set(original_indices).difference(set(self.filtered_index)))
            removed_original_indices = self.original_indices[removed_data_indices]

            self.removed_data_indices = removed_original_indices.tolist()
            self.all_removed_indices.extend(removed_original_indices.tolist())
            print(f"Datos removidos: {removed_original_indices}")

            if len(self.filtered_index) == 0:
                print("No se identificaron outliers, se utiliza el dataset filtrado previamente")
                return self.previous_X_tr, self.previous_y_tr, self.original_indices, self.all_removed_indices, self.inspector_layer_out

            filtered_X_tr = tf.gather(self.X_tr, self.filtered_index)
            filtered_y_tr = tf.gather(self.y_tr, self.filtered_index).numpy()
            filtered_original_indices = self.original_indices[self.filtered_index]
            size_set_post = filtered_X_tr.shape[0]
            print("El dataset ha sido filtrado.")
            print(f"Tamaño de datos removidos: {size_set_train - size_set_post}")

            if self.train_with_outliers:
                # Manejo de outliers: se mezclan datos removidos con puntos aleatorios
                removed_data = tf.gather(self.X_tr, np.array(removed_data_indices, dtype=int))
                removed_labels = tf.gather(self.y_tr, np.array(removed_data_indices, dtype=int))
                removed_indices = tf.gather(self.original_indices, np.array(removed_data_indices, dtype=int))
                num_removed = 3 * len(removed_data_indices)
                if num_removed == 0:
                    print("No hay datos para remover como outliers, se utiliza el dataset filtrado previo")
                    return self.previous_X_tr, self.previous_y_tr, self.original_indices, self.all_removed_indices, self.inspector_layer_out
                all_indices = set(range(len(self.X_tr)))
                excluded_indices = set(removed_data_indices)
                available_indices = list(all_indices - excluded_indices)
                random_indices = np.random.choice(available_indices, num_removed, replace=False)
                random_data = tf.gather(self.X_tr, random_indices)
                random_labels = tf.gather(self.y_tr, random_indices)
                random_original_indices = np.array(self.original_indices)[random_indices]
                filtered_X_tr = np.concatenate((removed_data, random_data), axis=0)
                filtered_y_tr = np.concatenate((removed_labels, random_labels), axis=0)
                filtered_original_indices = np.concatenate((removed_indices, random_original_indices), axis=0)
                print(f"Entrenamiento con outliers: se agregaron {num_removed} puntos removidos y {num_removed} puntos aleatorios.")

            # Actualización de variables para el siguiente epoch
            self.X_tr = filtered_X_tr
            self.y_tr = filtered_y_tr
            self.original_indices = filtered_original_indices
            self.previous_X_tr = self.X_tr
            self.previous_y_tr = self.y_tr
            self.inspector_layer_out = inspector_layer_out

        return self.return_filtered_data()

    def return_filtered_data(self):
        """Devuelve los datos filtrados y la información asociada."""
        return self.X_tr, self.y_tr, self.original_indices, self.all_removed_indices, self.inspector_layer_out


