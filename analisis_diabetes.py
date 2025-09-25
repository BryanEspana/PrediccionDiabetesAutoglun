import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# AutoGluon
from autogluon.tabular import TabularPredictor

# Carga del Conjunto de Datos
print("=== CARGA DEL CONJUNTO DE DATOS ===")
print("Cargando el archivo diabetes.csv...")

# Crear DataFrame con el contenido del archivo diabetes.csv
df = pd.read_csv('diabetes.csv')

print(f"✓ DataFrame creado exitosamente")
print(f"✓ Dimensiones del dataset: {df.shape[0]} filas x {df.shape[1]} columnas")
print()

# Revisión y descripción de las variables
print("=== DESCRIPCIÓN DE LAS VARIABLES ===")
print("Variables del conjunto de datos:")
print()

# Mostrar información básica de cada columna
for i, col in enumerate(df.columns, 1):
    print(f"{i}. {col}")
    if col == 'Pregnancies':
        print("   - Número de veces que ha estado embarazada")
    elif col == 'Glucose':
        print("   - Concentración de glucosa")
    elif col == 'BloodPressure':
        print("   - Presión de sangre diastólica (mm Hg)")
    elif col == 'SkinThickness':
        print("   - Grosor del pliegue cutáneo del Tríceps (mm)")
    elif col == 'Insulin':
        print("   - Suero de insulina 2-Horas (mu U/ml)")
    elif col == 'BMI':
        print("   - Índice de masa corporal (peso en Kg/(estatura en mts)²)")
    elif col == 'DiabetesPedigreeFunction':
        print("   - Función de pedigree de diabetes")
    elif col == 'Age':
        print("   - Edad en años")
    elif col == 'Outcome':
        print("   - Diabetes ó no diabetes (0 ó 1)")
    print()

# Explicación del contexto
print("=== CONTEXTO DEL CONJUNTO DE DATOS ===")
print("Este conjunto de datos contiene información médica de pacientes y se utiliza para")
print("predecir si un paciente tiene diabetes o no.")
print()
print("La columna 'Outcome' es la variable objetivo que representa:")
print("• 1 = El paciente SÍ tiene diabetes")
print("• 0 = El paciente NO tiene diabetes")
print()
print("Las demás variables son características médicas que se utilizan como predictores")
print("para determinar la probabilidad de que un paciente desarrolle diabetes.")
print()

# =============================================================================
# ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# =============================================================================

print("=" * 60)
print("ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
print("=" * 60)

# 1. Estadísticas descriptivas básicas
print("\n=== 1. ESTADÍSTICAS DESCRIPTIVAS BÁSICAS ===")
print("Estadísticas descriptivas del conjunto de datos:")
print()
print(df.describe())
print()

# 2. Verificación de valores nulos
print("=== 2. VERIFICACIÓN DE VALORES NULOS ===")
print("Valores nulos por columna:")
print(df.isnull().sum())
print()

# 3. Verificación de valores atípicos (usando IQR)
print("=== 3. DETECCIÓN DE VALORES ATÍPICOS ===")
print("Valores atípicos detectados usando el método IQR:")
print()

outliers_info = {}
for col in df.columns[:-1]:  # Excluir la columna Outcome
    if df[col].dtype in ['int64', 'float64']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outliers_count = len(outliers)
        outliers_info[col] = outliers_count
        print(f"{col}: {outliers_count} valores atípicos")

print()

# 4. Análisis del balance de clases
print("=== 4. ANÁLISIS DEL BALANCE DE CLASES ===")
print("Distribución de la variable 'Outcome':")
outcome_counts = df['Outcome'].value_counts()
print(outcome_counts)
print()
print("Porcentajes:")
outcome_percentages = df['Outcome'].value_counts(normalize=True) * 100
for outcome, percentage in outcome_percentages.items():
    status = "SÍ tiene diabetes" if outcome == 1 else "NO tiene diabetes"
    print(f"  {status}: {percentage:.1f}%")
print()

# 5. Matriz de correlación
print("=== 5. MATRIZ DE CORRELACIÓN ===")
print("Calculando correlaciones entre variables numéricas...")
correlation_matrix = df.corr()
print("\nMatriz de correlación:")
print(correlation_matrix.round(3))
print()

# 6. Visualizaciones
print("=== 6. GENERANDO VISUALIZACIONES ===")
print("Creando gráficos de distribución y análisis...")

# Configurar el estilo de los gráficos
plt.style.use('default')
fig = plt.figure(figsize=(20, 15))

# 6.1. Histogramas de todas las variables numéricas
print("Generando histogramas...")
for i, col in enumerate(df.columns[:-1], 1):  # Excluir Outcome
    plt.subplot(3, 3, i)
    plt.hist(df[col], bins=30, alpha=0.7, edgecolor='black')
    plt.title(f'Distribución de {col}')
    plt.xlabel(col)
    plt.ylabel('Frecuencia')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('histogramas_diabetes.png', dpi=300, bbox_inches='tight')
plt.show()

# 6.2. Boxplots para detectar outliers
print("Generando boxplots...")
fig, axes = plt.subplots(3, 3, figsize=(20, 15))
axes = axes.ravel()

for i, col in enumerate(df.columns[:-1]):  # Excluir Outcome
    axes[i].boxplot(df[col])
    axes[i].set_title(f'Boxplot de {col}')
    axes[i].set_ylabel('Valores')
    axes[i].grid(True, alpha=0.3)

# Ocultar el último subplot si no se usa
if len(df.columns) - 1 < len(axes):
    axes[-1].set_visible(False)

plt.tight_layout()
plt.savefig('boxplots_diabetes.png', dpi=300, bbox_inches='tight')
plt.show()

# 6.3. Matriz de correlación heatmap
print("Generando matriz de correlación...")
plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, 
            mask=mask,
            annot=True, 
            cmap='coolwarm', 
            center=0,
            square=True,
            fmt='.2f',
            cbar_kws={"shrink": .8})
plt.title('Matriz de Correlación - Variables del Dataset de Diabetes')
plt.tight_layout()
plt.savefig('matriz_correlacion_diabetes.png', dpi=300, bbox_inches='tight')
plt.show()

# 6.4. Gráfico de barras para el balance de clases
print("Generando gráfico de balance de clases...")
plt.figure(figsize=(8, 6))
outcome_labels = ['No Diabetes', 'Diabetes']
colors = ['lightblue', 'lightcoral']
bars = plt.bar(outcome_labels, outcome_counts.values, color=colors, edgecolor='black')
plt.title('Balance de Clases - Variable Outcome')
plt.ylabel('Número de Pacientes')
plt.grid(True, alpha=0.3)

# Agregar valores en las barras
for bar, count in zip(bars, outcome_counts.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
             f'{count}\n({count/len(df)*100:.1f}%)', 
             ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('balance_clases_diabetes.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n=== RESUMEN DEL EDA ===")
print(f"✓ Dataset cargado: {df.shape[0]} filas x {df.shape[1]} columnas")
print(f"✓ Valores nulos: {df.isnull().sum().sum()} (ninguno encontrado)")
print(f"✓ Valores atípicos detectados: {sum(outliers_info.values())} en total")
print(f"✓ Balance de clases: {outcome_counts[0]} sin diabetes vs {outcome_counts[1]} con diabetes")
print(f"✓ Archivos de gráficos guardados: histogramas_diabetes.png, boxplots_diabetes.png, matriz_correlacion_diabetes.png, balance_clases_diabetes.png")
print("\n¡Análisis Exploratorio de Datos completado!")

# =============================================================================
# ENTRENAMIENTO CON AUTOGLUON
# =============================================================================

print("=" * 60)
print("ENTRENAMIENTO CON AUTOGLUON")
print("=" * 60)

# 1. Separación de datos en entrenamiento y prueba
print("\n=== 1. SEPARACIÓN DE DATOS ===")
print("Separando datos en conjuntos de entrenamiento y prueba...")

# Preparar datos para AutoGluon
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# División 80% entrenamiento, 20% prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Crear DataFrames completos para AutoGluon
train_data = X_train.copy()
train_data['Outcome'] = y_train

test_data = X_test.copy()
test_data['Outcome'] = y_test

print(f"✓ Datos de entrenamiento: {train_data.shape[0]} muestras")
print(f"✓ Datos de prueba: {test_data.shape[0]} muestras")
print(f"✓ Balance de clases en entrenamiento: {y_train.value_counts().to_dict()}")
print(f"✓ Balance de clases en prueba: {y_test.value_counts().to_dict()}")

# 2. Entrenamiento con AutoGluon
print("\n=== 2. ENTRENAMIENTO CON AUTOGLUON ===")
print("Configurando AutoGluon con preset 'best_quality'...")

# Configurar predictor
predictor = TabularPredictor(
    label='Outcome',
    problem_type='binary',
    eval_metric='accuracy'
)

print("Iniciando entrenamiento con múltiples modelos...")
print("⏱️  Tiempo límite: 300 segundos (5 minutos)")

# Entrenar con AutoGluon
predictor.fit(
    train_data=train_data,
    presets='best_quality',
    time_limit=300,  # 5 minutos
    verbosity=2
)

print("✓ Entrenamiento completado!")

# 3. Evaluación de modelos
print("\n=== 3. EVALUACIÓN DE MODELOS ===")
print("Evaluando rendimiento en datos de prueba...")

# Obtener predicciones
y_pred_autogluon = predictor.predict(test_data.drop('Outcome', axis=1))
accuracy_autogluon = accuracy_score(y_test, y_pred_autogluon)

print(f"✓ Precisión de AutoGluon: {accuracy_autogluon:.4f} ({accuracy_autogluon*100:.2f}%)")

# 4. Comparación con modelo base (Regresión Logística)
print("\n=== 4. COMPARACIÓN CON MODELO BASE ===")
print("Entrenando modelo base: Regresión Logística...")

# Estandarizar datos para regresión logística
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenar regresión logística
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)

# Predecir con regresión logística
y_pred_lr = lr_model.predict(X_test_scaled)
accuracy_lr = accuracy_score(y_test, y_pred_lr)

print(f"✓ Precisión de Regresión Logística: {accuracy_lr:.4f} ({accuracy_lr*100:.2f}%)")

# Comparar resultados
print(f"\n📊 COMPARACIÓN DE RESULTADOS:")
print(f"   AutoGluon (mejor modelo):     {accuracy_autogluon:.4f} ({accuracy_autogluon*100:.2f}%)")
print(f"   Regresión Logística (base):   {accuracy_lr:.4f} ({accuracy_lr*100:.2f}%)")
print(f"   Mejora:                       {accuracy_autogluon - accuracy_lr:.4f} ({(accuracy_autogluon - accuracy_lr)*100:.2f}%)")

# 5. Importancia de características
print("\n=== 5. IMPORTANCIA DE CARACTERÍSTICAS ===")
print("Analizando importancia de características...")

# Obtener importancia de características
feature_importance = predictor.feature_importance(test_data)
print("\nImportancia de características (AutoGluon):")
print(feature_importance)

# Visualizar importancia de características
plt.figure(figsize=(10, 6))
feature_importance.plot(kind='barh')
plt.title('Importancia de Características - AutoGluon')
plt.xlabel('Importancia')
plt.tight_layout()
plt.savefig('importancia_caracteristicas_autogluon.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. Resumen de modelos entrenados
print("\n=== 6. RESUMEN DE MODELOS ENTRENADOS ===")
print("Modelos entrenados por AutoGluon:")
leaderboard = predictor.leaderboard(test_data, silent=True)
print(leaderboard[['model', 'score_val', 'score_test']].head(10))

# 7. Reporte de clasificación detallado
print("\n=== 7. REPORTE DE CLASIFICACIÓN DETALLADO ===")
print("Reporte para el mejor modelo de AutoGluon:")
print(classification_report(y_test, y_pred_autogluon, target_names=['No Diabetes', 'Diabetes']))

print("\nReporte para Regresión Logística:")
print(classification_report(y_test, y_pred_lr, target_names=['No Diabetes', 'Diabetes']))

print("=" * 60)
print("ENTRENAMIENTO CON AUTOGLUON COMPLETADO")
print("=" * 60)
