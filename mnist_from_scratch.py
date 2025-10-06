"""
MNIST Classifier from Scratch
Implementación manual de una red neuronal multicapa sin usar bibliotecas de alto nivel.
Solo se utiliza NumPy para operaciones matriciales.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time


class NeuralNetwork:
    """
    Red Neuronal Multicapa implementada desde cero.
    Arquitectura: 784 -> 128 -> 64 -> 10
    """
    
    def __init__(self, layer_sizes, learning_rate=0.01, random_seed=42):
        """
        Inicializa la red neuronal.
        
        Args:
            layer_sizes: Lista con el número de neuronas en cada capa [784, 128, 64, 10]
            learning_rate: Tasa de aprendizaje para gradient descent
            random_seed: Semilla para reproducibilidad
        """
        np.random.seed(random_seed)
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.num_layers = len(layer_sizes)
        
        # Inicializar pesos y sesgos
        self.weights = {}
        self.biases = {}
        
        # Inicialización He para ReLU
        for l in range(1, self.num_layers):
            self.weights[l] = np.random.randn(
                layer_sizes[l], 
                layer_sizes[l-1]
            ) * np.sqrt(2.0 / layer_sizes[l-1])
            self.biases[l] = np.zeros((layer_sizes[l], 1))
        
        # Para almacenar valores durante forward/backward pass
        self.cache = {}
        
        # Historial de entrenamiento
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_loss_history = []
        self.val_acc_history = []
    
    def relu(self, Z):
        """Función de activación ReLU."""
        return np.maximum(0, Z)
    
    def relu_derivative(self, Z):
        """Derivada de ReLU."""
        return (Z > 0).astype(float)
    
    def softmax(self, Z):
        """
        Función Softmax para la capa de salida.
        Usa trick de estabilidad numérica restando el máximo.
        """
        exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
    
    def forward_pass(self, X):
        """
        Propagación hacia adelante.
        
        Args:
            X: Matriz de entrada (n_features, m_samples)
            
        Returns:
            A_L: Activaciones de la capa de salida (probabilidades)
        """
        self.cache['A0'] = X
        A = X
        
        # Capas ocultas con ReLU
        for l in range(1, self.num_layers - 1):
            Z = self.weights[l] @ A + self.biases[l]
            A = self.relu(Z)
            self.cache[f'Z{l}'] = Z
            self.cache[f'A{l}'] = A
        
        # Capa de salida con Softmax
        L = self.num_layers - 1
        Z_L = self.weights[L] @ A + self.biases[L]
        A_L = self.softmax(Z_L)
        self.cache[f'Z{L}'] = Z_L
        self.cache[f'A{L}'] = A_L
        
        return A_L
    
    def compute_loss(self, Y_pred, Y_true):
        """
        Calcula Cross-Entropy Loss.
        
        Args:
            Y_pred: Predicciones (n_classes, m_samples)
            Y_true: Etiquetas one-hot (n_classes, m_samples)
            
        Returns:
            loss: Valor del error promedio
        """
        m = Y_true.shape[1]
        # Añadir epsilon para evitar log(0)
        epsilon = 1e-8
        loss = -np.sum(Y_true * np.log(Y_pred + epsilon)) / m
        return loss
    
    def backward_pass(self, Y_true):
        """
        Propagación hacia atrás (Backpropagation).
        
        Args:
            Y_true: Etiquetas one-hot (n_classes, m_samples)
            
        Returns:
            gradients: Diccionario con dW y db para cada capa
        """
        m = Y_true.shape[1]
        L = self.num_layers - 1
        gradients = {}
        
        # Capa de salida: Softmax + Cross-Entropy
        dZ = self.cache[f'A{L}'] - Y_true
        gradients[f'dW{L}'] = (1/m) * (dZ @ self.cache[f'A{L-1}'].T)
        gradients[f'db{L}'] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        
        # Capas ocultas (hacia atrás)
        for l in range(L-1, 0, -1):
            dA = self.weights[l+1].T @ dZ
            dZ = dA * self.relu_derivative(self.cache[f'Z{l}'])
            
            gradients[f'dW{l}'] = (1/m) * (dZ @ self.cache[f'A{l-1}'].T)
            gradients[f'db{l}'] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        
        return gradients
    
    def update_parameters(self, gradients):
        """
        Actualiza pesos y sesgos usando Gradient Descent.
        
        Args:
            gradients: Diccionario con dW y db para cada capa
        """
        for l in range(1, self.num_layers):
            self.weights[l] -= self.learning_rate * gradients[f'dW{l}']
            self.biases[l] -= self.learning_rate * gradients[f'db{l}']
    
    def train_step(self, X_batch, Y_batch):
        """
        Ejecuta un paso de entrenamiento (forward + backward + update).
        
        Args:
            X_batch: Batch de datos (n_features, batch_size)
            Y_batch: Batch de etiquetas one-hot (n_classes, batch_size)
            
        Returns:
            loss: Error del batch
        """
        # Forward pass
        Y_pred = self.forward_pass(X_batch)
        
        # Compute loss
        loss = self.compute_loss(Y_pred, Y_batch)
        
        # Backward pass
        gradients = self.backward_pass(Y_batch)
        
        # Update parameters
        self.update_parameters(gradients)
        
        return loss
    
    def predict(self, X):
        """
        Realiza predicciones sobre un conjunto de datos.
        
        Args:
            X: Matriz de entrada (n_features, m_samples)
            
        Returns:
            predictions: Clases predichas (m_samples,)
        """
        Y_pred = self.forward_pass(X)
        predictions = np.argmax(Y_pred, axis=0)
        return predictions
    
    def compute_accuracy(self, X, Y):
        """
        Calcula accuracy sobre un conjunto de datos.
        
        Args:
            X: Datos (n_features, m_samples)
            Y: Etiquetas one-hot (n_classes, m_samples)
            
        Returns:
            accuracy: Porcentaje de aciertos
        """
        predictions = self.predict(X)
        true_labels = np.argmax(Y, axis=0)
        accuracy = np.mean(predictions == true_labels) * 100
        return accuracy
    
    def train(self, X_train, Y_train, X_val, Y_val, epochs=30, batch_size=128, verbose=True):
        """
        Entrena la red neuronal.
        
        Args:
            X_train: Datos de entrenamiento (n_features, m_train)
            Y_train: Etiquetas de entrenamiento one-hot (n_classes, m_train)
            X_val: Datos de validación (n_features, m_val)
            Y_val: Etiquetas de validación one-hot (n_classes, m_val)
            epochs: Número de épocas
            batch_size: Tamaño de mini-batch
            verbose: Si imprimir progreso
        """
        m = X_train.shape[1]
        num_batches = m // batch_size
        
        print(f"Iniciando entrenamiento...")
        print(f"Arquitectura: {' -> '.join(map(str, self.layer_sizes))}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Batch size: {batch_size}")
        print(f"Épocas: {epochs}")
        print(f"Samples entrenamiento: {m}")
        print(f"Batches por época: {num_batches}\n")
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Mezclar datos al inicio de cada época
            permutation = np.random.permutation(m)
            X_shuffled = X_train[:, permutation]
            Y_shuffled = Y_train[:, permutation]
            
            # Mini-batch gradient descent
            epoch_loss = 0
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                
                X_batch = X_shuffled[:, start_idx:end_idx]
                Y_batch = Y_shuffled[:, start_idx:end_idx]
                
                batch_loss = self.train_step(X_batch, Y_batch)
                epoch_loss += batch_loss
            
            # Promediar loss de la época
            epoch_loss /= num_batches
            
            # Calcular métricas
            train_acc = self.compute_accuracy(X_train, Y_train)
            val_loss = self.compute_loss(self.forward_pass(X_val), Y_val)
            val_acc = self.compute_accuracy(X_val, Y_val)
            
            # Guardar historial
            self.train_loss_history.append(epoch_loss)
            self.train_acc_history.append(train_acc)
            self.val_loss_history.append(val_loss)
            self.val_acc_history.append(val_acc)
            
            elapsed = time.time() - start_time
            
            if verbose:
                print(f"Época {epoch+1}/{epochs} - "
                      f"Loss: {epoch_loss:.4f} - "
                      f"Train Acc: {train_acc:.2f}% - "
                      f"Val Loss: {val_loss:.4f} - "
                      f"Val Acc: {val_acc:.2f}% - "
                      f"Tiempo: {elapsed:.2f}s")
        
        print("\n¡Entrenamiento completado!")
    
    def plot_training_history(self):
        """Grafica el historial de entrenamiento."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss
        axes[0].plot(self.train_loss_history, label='Train Loss', linewidth=2)
        axes[0].plot(self.val_loss_history, label='Validation Loss', linewidth=2)
        axes[0].set_xlabel('Época', fontsize=12)
        axes[0].set_ylabel('Loss (Cross-Entropy)', fontsize=12)
        axes[0].set_title('Evolución del Error Durante Entrenamiento', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[1].plot(self.train_acc_history, label='Train Accuracy', linewidth=2)
        axes[1].plot(self.val_acc_history, label='Validation Accuracy', linewidth=2)
        axes[1].set_xlabel('Época', fontsize=12)
        axes[1].set_ylabel('Accuracy (%)', fontsize=12)
        axes[1].set_title('Evolución de la Precisión Durante Entrenamiento', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def load_and_preprocess_mnist():
    """
    Carga y preprocesa el dataset MNIST.
    
    Returns:
        X_train, X_val, X_test: Datos normalizados (n_features, m_samples)
        Y_train, Y_val, Y_test: Etiquetas one-hot (n_classes, m_samples)
    """
    print("Cargando dataset MNIST...")
    
    # Cargar MNIST desde scikit-learn
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    X = mnist.data.astype('float32').values  # Convert DataFrame to NumPy array
    y = mnist.target.astype('int').values  # Convert Series to NumPy array
    
    print(f"Dataset cargado: {X.shape[0]} imágenes de {X.shape[1]} píxeles")
    
    # Dividir en train, validation y test
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=10000, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=10000, random_state=42
    )
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Normalizar píxeles a rango [0, 1]
    X_train = X_train / 255.0
    X_val = X_val / 255.0
    X_test = X_test / 255.0
    
    # Convertir a formato (n_features, m_samples) - transponer
    X_train = X_train.T
    X_val = X_val.T
    X_test = X_test.T
    
    # Convertir etiquetas a one-hot encoding
    def to_one_hot(y, num_classes=10):
        m = y.shape[0]
        one_hot = np.zeros((num_classes, m))
        one_hot[y, np.arange(m)] = 1
        return one_hot
    
    Y_train = to_one_hot(y_train)
    Y_val = to_one_hot(y_val)
    Y_test = to_one_hot(y_test)
    
    print("\nPreprocesamiento completado.")
    print(f"Forma de X_train: {X_train.shape}")
    print(f"Forma de Y_train: {Y_train.shape}\n")
    
    return X_train, X_val, X_test, Y_train, Y_val, Y_test


def plot_confusion_matrix(y_true, y_pred):
    """Crea y grafica una matriz de confusión."""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicción', fontsize=12)
    plt.ylabel('Etiqueta Real', fontsize=12)
    plt.title('Matriz de Confusión - MNIST desde Cero', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_sample_predictions(X_test, Y_test, model, num_samples=10):
    """Muestra algunas predicciones de ejemplo."""
    # Convertir de vuelta a formato para visualización
    X_samples = X_test[:, :num_samples].T  # (num_samples, 784)
    y_true = np.argmax(Y_test[:, :num_samples], axis=0)
    y_pred = model.predict(X_test[:, :num_samples])
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    for i in range(num_samples):
        axes[i].imshow(X_samples[i].reshape(28, 28), cmap='gray')
        axes[i].set_title(f'Real: {y_true[i]}, Pred: {y_pred[i]}',
                         color='green' if y_true[i] == y_pred[i] else 'red',
                         fontweight='bold')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def main():
    """Función principal para entrenar y evaluar el modelo."""
    
    # Cargar y preprocesar datos
    X_train, X_val, X_test, Y_train, Y_val, Y_test = load_and_preprocess_mnist()
    
    # Definir arquitectura de la red
    layer_sizes = [784, 128, 64, 10]
    
    # Crear modelo
    model = NeuralNetwork(
        layer_sizes=layer_sizes,
        learning_rate=0.1,
        random_seed=42
    )
    
    # Entrenar modelo
    model.train(
        X_train, Y_train,
        X_val, Y_val,
        epochs=30,
        batch_size=128,
        verbose=True
    )
    
    # Graficar historial de entrenamiento
    model.plot_training_history()
    
    # Evaluar en conjunto de test
    print("\n" + "="*50)
    print("EVALUACIÓN EN CONJUNTO DE TEST")
    print("="*50)
    
    test_acc = model.compute_accuracy(X_test, Y_test)
    test_loss = model.compute_loss(model.forward_pass(X_test), Y_test)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    # Matriz de confusión
    y_test_labels = np.argmax(Y_test, axis=0)
    y_test_pred = model.predict(X_test)
    plot_confusion_matrix(y_test_labels, y_test_pred)
    
    # Mostrar predicciones de ejemplo
    plot_sample_predictions(X_test, Y_test, model, num_samples=10)
    
    print("\n" + "="*50)
    print("RESUMEN FINAL")
    print("="*50)
    print(f"Arquitectura: {' -> '.join(map(str, layer_sizes))}")
    print(f"Mejor Validation Accuracy: {max(model.val_acc_history):.2f}%")
    print(f"Test Accuracy Final: {test_acc:.2f}%")
    print(f"Parámetros totales: {sum(w.size for w in model.weights.values()) + sum(b.size for b in model.biases.values())}")
    
    return model


if __name__ == "__main__":
    model = main()
