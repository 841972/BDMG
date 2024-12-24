import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from rapidfuzz import fuzz
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Cargar los datasets
path_baseA = 'BaseA_cleaned.csv'
path_baseB= 'BaseB_cleaned.csv'

base_a = pd.read_csv(path_baseA)
base_b = pd.read_csv(path_baseB)

base_a["Block"] = base_a["Name"].str[0].str.lower()
base_b["Block"] = base_b["Name"].str[0].str.lower()

#print(base_a.head())

# Inicializar una lista para almacenar los pares bloqueados
blocked_pairs = []

# Realizar el bloque por cada valor único en el atributo de bloqueo
for block in base_a["Block"].unique():
    # Filtrar los registros por bloque
    block_a = base_a[base_a["Block"] == block]
    block_b = base_b[base_b["Block"] == block]
    
    # Crear pares dentro del bloque
    for idx_a, record_a in block_a.iterrows():
        for idx_b, record_b in block_b.iterrows():
            blocked_pairs.append((idx_a, idx_b))

# Convertir los pares bloqueados en un DataFrame
pairs_df = pd.DataFrame(blocked_pairs, columns=["Index_A", "Index_B"])
#print(pairs_df.head())

# Crear características para los pares bloqueados
def compute_features(row):
    record_a = base_a.loc[row["Index_A"]]
    record_b = base_b.loc[row["Index_B"]]

    # Similitudes
    name_similarity = fuzz.WRatio(record_a["Name"], record_b["Name"])
    email_similarity = fuzz.WRatio(record_a["Email"], record_b["Email"])

    id_match = int(record_a["ID"] == record_b["ID"])
    phone_match = int(record_a["Phone"] == record_b["Phone"])

    #print(record_a["Transaction_Amount"], record_b["Transaction_Amount"])
    try:
        sus1 = float(record_a["Transaction_Amount"])
        sus2 = float(record_b["Transaction_Amount"])
    except ValueError:
        sus1 = 0.0
        sus2 = 0.0
    #print(sus1)
    #print(sus2)
    total_spent_diff = abs(sus1 - sus2)

    return pd.Series([name_similarity, email_similarity, phone_match, total_spent_diff])

# Aplicar características a los pares
pairs_df[["Name_Similarity", "Email_Similarity", "Phone_Match", "Total_Spent_Diff"]] = pairs_df.apply(compute_features, axis=1)

pairs_df["Label"] = [1 if base_a.loc[row["Index_A"], "ID"] == base_b.loc[row["Index_B"], "ID"] else 0 for _, row in pairs_df.iterrows()]

# Dividir los datos en entrenamiento y prueba
X = pairs_df[["Name_Similarity", "Email_Similarity", "Phone_Match", "Total_Spent_Diff"]]
y = pairs_df["Label"]
#print(X.head())
#print(y.head())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

rf_model.fit(X_train, y_train)

# Hacer predicciones
y_pred = rf_model.predict(X_test)

# Evaluar el modelo
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}")



####GRAFICAR

# Obtener la importancia de las características
feature_importances = rf_model.feature_importances_
features = X.columns

# Crear un DataFrame para ordenarlas
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

print(importance_df)

#X2_train, X2_test, y2_train, y2_test = train_test_split(X, y, test_size=0.2, random_state=42)

#rf_model2 = RandomForestClassifier(
#    n_estimators=200,    
#    max_depth=10,       
#    min_samples_split=5, 
#    random_state=42
#)

#rf_model2 = RandomForestClassifier(n_estimators=100, random_state=42)

#rf_model2.fit(X2_train, y2_train)

# Hacer predicciones
#y_pred2 = rf_model2.predict(X2_test)

# Evaluar el modelo
#precision = precision_score(y2_test, y_pred2)
#recall = recall_score(y2_test, y_pred2)
#f1 = f1_score(y2_test, y_pred2)

#print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}")
