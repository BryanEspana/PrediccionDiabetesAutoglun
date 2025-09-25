import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

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
