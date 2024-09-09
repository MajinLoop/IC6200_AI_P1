import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# Cargar el dataset
df = pd.read_csv('../data/alzheimer.csv')
df = df[df['Group'] != 'Converted']

# Prepara los datos
X = df.drop('Group', axis=1)  # 'Group' es la columna objetivo
y = df['Group']

# Convierte variables categóricas a variables dummy
X = pd.get_dummies(X, drop_first=True)

# Eliminar filas con valores faltantes
X_clean = X.dropna()
y_clean = y[X_clean.index]

# Divide el dataset en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)

# Escala las características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

best_k = None
best_accuracy = 0

for k in range(1,40,2):
    # Inicializa el clasificador KNN
    model = KNeighborsClassifier(n_neighbors=k)
    
    # Entrena el modelo
    model.fit(X_train, y_train)
    
    # Realiza predicciones
    y_pred = model.predict(X_test)
    
    # Imprime el informe de clasificación
    print(f'Informe de clasificación para k={k}:')
    print(classification_report(y_test, y_pred))
    
    # Calcula la precisión
    accuracy = model.score(X_test, y_test)
    print(f'Precisión para k={k}: {accuracy:.4f}\n')
    
    # Guarda el mejor valor de k
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k

print(f'El mejor valor de k es {best_k} con una precisión de {best_accuracy:.4f}')
