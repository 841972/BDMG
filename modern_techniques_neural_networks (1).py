import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score
from rapidfuzz import fuzz
#from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from datetime import datetime
from scikeras.wrappers import KerasClassifier

# Cargar los datasets
path_baseA = 'BaseA_cleaned.csv'
path_baseB= 'BaseB_cleaned.csv'

base_a = pd.read_csv(path_baseA)
base_b = pd.read_csv(path_baseB)

# Convertir DOB y Transaction_Date a formato datetime
base_a["DOB"] = pd.to_datetime(base_a["DOB"], errors='coerce')
base_b["DOB"] = pd.to_datetime(base_b["DOB"], errors='coerce')
base_a["Transaction_Date"] = pd.to_datetime(base_a["Transaction_Date"], errors='coerce')
base_b["Transaction_Date"] = pd.to_datetime(base_b["Transaction_Date"], errors='coerce')


# Llenar valores faltantes en columnas relevantes
base_a.fillna({'Name': '', 'Email': '', 'Address': '', 'Phone': '', 'DOB': pd.Timestamp.min, 'Transaction_Date': pd.Timestamp.min, 'TotalSpent': 0.0, 'Transaction_Amount': 0.0}, inplace=True)
base_b.fillna({'Name': '', 'Email': '', 'Address': '', 'Phone': '', 'DOB': pd.Timestamp.min, 'Transaction_Date': pd.Timestamp.min, 'TotalSpent': 0.0, 'Transaction_Amount': 0.0}, inplace=True)


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

# Calcular características
def compute_features(row):
    record_a = base_a.loc[row["Index_A"]]
    record_b = base_b.loc[row["Index_B"]]

    # Match exacto de ID
    #id_match = int(record_a["ID"] == record_b["ID"])

    # Match exacto de Phone
    phone_match = int(record_a["Phone"] == record_b["Phone"])

    # Similitudes de cadenas
    email_similarity = fuzz.WRatio(record_a["Email"], record_b["Email"])
    name_similarity = fuzz.WRatio(record_a["Name"], record_b["Name"])

    address_similarity = fuzz.WRatio(record_a["Address"], record_b["Address"])
    
    # Diferencia de fechas
    if pd.notnull(record_a["DOB"]) and pd.notnull(record_b["DOB"]):
        dob_diff = abs((record_a["DOB"].to_pydatetime() - record_b["DOB"].to_pydatetime()).days)
    else:
        dob_diff = 1000  # Valor predeterminado

    if pd.notnull(record_a["Transaction_Date"]) and pd.notnull(record_b["Transaction_Date"]):
        transaction_date_diff = abs((record_a["Transaction_Date"].to_pydatetime() - record_b["Transaction_Date"].to_pydatetime()).days)
    else:
        transaction_date_diff = 1000  # Valor predeterminado


    # Diferencia numérica en gasto total
    try:
        sus1 =float(record_a["TotalSpent"])
        sus2 = float(record_b["TotalSpent"])
    except:
        sus1 = 0.0
        sus2 = 0.0
    total_spent_diff = abs(sus1 - sus2)

    # Matches exactos
    transaction_amount_match = int(record_a["Transaction_Amount"] == record_b["Transaction_Amount"])

    return pd.Series([name_similarity, email_similarity, address_similarity, dob_diff, transaction_date_diff, total_spent_diff, phone_match, transaction_amount_match])

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Evaluar precisión, recall y F1-score
def evaluate_results(matches, true_matches):

    y_true = [
        1 if (pair[0], pair[1]) in true_matches else 0
        for pair in matches['index_pair'] # Iterating over the index pairs
    ]
    
    y_pred = [1] * len(matches)

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return precision, recall, f1

# def evaluate_results(num_matches, num_true_matches):
#     # Recall
#     recall = num_true_matches / num_matches
#     if recall > 1:
#         recall = 1
#     # Precision
#     precision = num_matches / num_true_matches
#     # F1-Score
#     f1 = 2 * (precision * recall) / (precision + recall)

#     return precision, recall, f1

# Crear el DataFrame de pares bloqueados
pairs_df = pd.DataFrame(blocked_pairs, columns=["Index_A", "Index_B"])

# Calcular características para los pares bloqueados
pairs_df[["Name_Similarity", "Email_Similarity", "Address_Similarity", "DOB_Diff", "Transaction_Date_Diff", "Total_Spent_Diff", "Phone_Match", "Transaction_Amount_Match"]] = pairs_df.apply(compute_features, axis=1)

# Normalizar características numéricas
scaler = MinMaxScaler()
pairs_df[["DOB_Diff", "Transaction_Date_Diff", "Total_Spent_Diff"]] = scaler.fit_transform(pairs_df[["DOB_Diff", "Transaction_Date_Diff", "Total_Spent_Diff"]])

# Codificar etiquetas
pairs_df["Label"] = [
    1 if base_a.loc[row["Index_A"], "ID"] == base_b.loc[row["Index_B"], "ID"]
    else (1 if base_a.loc[row["Index_A"], "Phone"] == base_b.loc[row["Index_B"], "Phone"] and
              fuzz.WRatio(base_a.loc[row["Index_A"], "Name"], base_b.loc[row["Index_B"], "Name"]) > 85
          else 0)
    for _, row in pairs_df.iterrows()
]


# Dividir los datos en entrenamiento y prueba
X = pairs_df.drop(columns=["Index_A", "Index_B", "Label"])  # Elimina columnas que no son características
y = pairs_df["Label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo
model = create_model()
keras_model = KerasClassifier(model=create_model, epochs=30, batch_size=32, verbose=0)

scores = cross_val_score(keras_model, X, y, cv=5, scoring='f1')
print(f"Cross-validated F1 scores: {scores}")
print(f"Mean F1 score: {scores.mean():.2f}")

history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)


# Evaluar el modelo
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}")

# Graficar la precisión
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Graficar la pérdida
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#print(X.head())
#print(y.head())