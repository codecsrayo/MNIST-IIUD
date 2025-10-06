"""
MNIST Classifier usando scikit-learn
Implementación usando MLPClassifier de sklearn para comparar con la implementación desde cero.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import time


def load_mnist_data():
    """Carga y preprocesa el dataset MNIST."""
    print("Cargando dataset MNIST...")
    
    # Cargar MNIST
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
    
    print("\nPreprocesamiento completado.\n")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_sklearn_model(X_train, y_train, X_val, y_val):
    """Entrena un clasificador MLPClassifier de scikit-learn."""
    
    print("="*60)
    print("ENTRENAMIENTO CON SCIKIT-LEARN MLPClassifier")
    print("="*60)
    
    # Crear modelo con arquitectura similar a la implementación desde cero
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),  # 2 capas ocultas: 128 y 64 neuronas
        activation='relu',              # Activación ReLU
        solver='sgd',                   # Stochastic Gradient Descent
        batch_size=128,                 # Tamaño de batch
        learning_rate_init=0.1,         # Learning rate inicial
        learning_rate='constant',       # Learning rate constante
        max_iter=30,                    # 30 épocas
        random_state=42,
        verbose=True,                   # Mostrar progreso
        early_stopping=False,           # Sin early stopping para comparación justa
        n_iter_no_change=30,
        tol=0
    )
    
    print(f"\nArquitectura: 784 -> 128 -> 64 -> 10")
    print(f"Activación: ReLU")
    print(f"Optimizador: SGD")
    print(f"Learning rate: 0.1")
    print(f"Batch size: 128")
    print(f"Épocas: 30\n")
    
    # Entrenar modelo
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"\nEntrenamiento completado en {training_time:.2f} segundos")
    
    # Evaluar en train y validation
    train_acc = accuracy_score(y_train, model.predict(X_train)) * 100
    val_acc = accuracy_score(y_val, model.predict(X_val)) * 100
    
    print(f"\nTrain Accuracy: {train_acc:.2f}%")
    print(f"Validation Accuracy: {val_acc:.2f}%")
    
    return model


def evaluate_model(model, X_test, y_test):
    """Evalúa el modelo en el conjunto de test."""
    
    print("\n" + "="*60)
    print("EVALUACIÓN EN CONJUNTO DE TEST")
    print("="*60)
    
    # Predicciones
    y_pred = model.predict(X_test)
    
    # Accuracy
    test_acc = accuracy_score(y_test, y_pred) * 100
    print(f"\nTest Accuracy: {test_acc:.2f}%")
    
    # Reporte de clasificación
    print("\nReporte de Clasificación:\n")
    print(classification_report(y_test, y_pred, target_names=[str(i) for i in range(10)]))
    
    return y_pred, test_acc


def plot_confusion_matrix(y_true, y_pred, filename='confusion_matrix_sklearn.png'):
    """Crea y grafica una matriz de confusión."""
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicción', fontsize=12)
    plt.ylabel('Etiqueta Real', fontsize=12)
    plt.title('Matriz de Confusión - MNIST con scikit-learn', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'/home/ubuntu/2025/III/ML/{filename}', dpi=150, bbox_inches='tight')
    print(f"\nMatriz de confusión guardada en: {filename}")
    plt.show()


def plot_sample_predictions(X_test, y_test, model, num_samples=10):
    """Muestra algunas predicciones de ejemplo."""
    
    # Seleccionar muestras aleatorias
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    X_samples = X_test[indices]
    y_true = y_test[indices]
    y_pred = model.predict(X_samples)
    
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


def plot_loss_curve(model):
    """Grafica la curva de pérdida durante el entrenamiento."""
    
    if hasattr(model, 'loss_curve_'):
        plt.figure(figsize=(10, 6))
        plt.plot(model.loss_curve_, linewidth=2, color='green')
        plt.xlabel('Iteración', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Evolución del Error Durante Entrenamiento (scikit-learn)', 
                 fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    else:
        print("El modelo no tiene historial de pérdidas disponible.")


def analyze_errors(X_test, y_test, y_pred):
    """Analiza los errores de clasificación."""
    
    # Encontrar índices de predicciones incorrectas
    error_indices = np.where(y_test != y_pred)[0]
    
    print(f"\n{'='*60}")
    print("ANÁLISIS DE ERRORES")
    print(f"{'='*60}")
    print(f"Total de errores: {len(error_indices)} de {len(y_test)} ({len(error_indices)/len(y_test)*100:.2f}%)")
    
    # Mostrar algunos errores
    if len(error_indices) > 0:
        num_errors_to_show = min(10, len(error_indices))
        error_samples = np.random.choice(error_indices, num_errors_to_show, replace=False)
        
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.ravel()
        
        for i, idx in enumerate(error_samples):
            axes[i].imshow(X_test[idx].reshape(28, 28), cmap='gray')
            axes[i].set_title(f'Real: {y_test[idx]}, Pred: {y_pred[idx]}',
                             color='red', fontweight='bold')
            axes[i].axis('off')
        
        plt.suptitle('Ejemplos de Clasificaciones Incorrectas', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()


def compare_architectures(X_train, y_train, X_test, y_test):
    """Compara diferentes arquitecturas de red."""
    
    print("\n" + "="*60)
    print("COMPARACIÓN DE DIFERENTES ARQUITECTURAS")
    print("="*60)
    
    architectures = {
        'Simple (64)': (64,),
        'Mediana (128, 64)': (128, 64),
        'Compleja (256, 128, 64)': (256, 128, 64),
        'Muy compleja (512, 256, 128)': (512, 256, 128)
    }
    
    results = {}
    
    for name, hidden_layers in architectures.items():
        print(f"\nEntrenando arquitectura: {name}")
        print(f"Capas: 784 -> {' -> '.join(map(str, hidden_layers))} -> 10")
        
        model = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation='relu',
            solver='adam',
            batch_size=128,
            learning_rate_init=0.001,
            max_iter=20,
            random_state=42,
            verbose=False
        )
        
        start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start
        
        test_acc = accuracy_score(y_test, model.predict(X_test)) * 100
        
        results[name] = {
            'accuracy': test_acc,
            'time': train_time,
            'params': sum(w.size for w in model.coefs_) + sum(b.size for b in model.intercepts_)
        }
        
        print(f"Test Accuracy: {test_acc:.2f}% - Tiempo: {train_time:.2f}s - Parámetros: {results[name]['params']:,}")
    
    # Graficar comparación
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in names]
    times = [results[name]['time'] for name in names]
    
    axes[0].barh(names, accuracies, color='skyblue')
    axes[0].set_xlabel('Test Accuracy (%)', fontsize=12)
    axes[0].set_title('Comparación de Precisión', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='x')
    
    axes[1].barh(names, times, color='lightcoral')
    axes[1].set_xlabel('Tiempo de Entrenamiento (s)', fontsize=12)
    axes[1].set_title('Comparación de Tiempo', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.show()
    
    return results


def main():
    """Función principal."""
    
    # Cargar datos
    X_train, X_val, X_test, y_train, y_val, y_test = load_mnist_data()
    
    # Entrenar modelo
    model = train_sklearn_model(X_train, y_train, X_val, y_val)
    
    # Graficar curva de pérdida
    plot_loss_curve(model)
    
    # Evaluar modelo
    y_pred, test_acc = evaluate_model(model, X_test, y_test)
    
    # Matriz de confusión
    plot_confusion_matrix(y_test, y_pred)
    
    # Predicciones de ejemplo
    plot_sample_predictions(X_test, y_test, model)
    
    # Análisis de errores
    analyze_errors(X_test, y_test, y_pred)
    
    # Comparar diferentes arquitecturas
    comparison_results = compare_architectures(X_train, y_train, X_test, y_test)
    
    # Resumen final
    print("\n" + "="*60)
    print("RESUMEN FINAL - SCIKIT-LEARN")
    print("="*60)
    print(f"Arquitectura: 784 -> 128 -> 64 -> 10")
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    total_params = sum(w.size for w in model.coefs_) + sum(b.size for b in model.intercepts_)
    print(f"Parámetros totales: {total_params:,}")
    print(f"\nMejor arquitectura alternativa:")
    best_arch = max(comparison_results.items(), key=lambda x: x[1]['accuracy'])
    print(f"  {best_arch[0]}: {best_arch[1]['accuracy']:.2f}% accuracy")
    
    return model


if __name__ == "__main__":
    model = main()
