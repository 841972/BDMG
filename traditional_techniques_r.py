import pandas as pd
from itertools import product
from rapidfuzz import fuzz, process
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuración de umbral de similitud
SIMILARITY_THRESHOLD = 0.9
path_baseA = './BaseA_cleaned.csv'
path_baseB = './BaseB_cleaned.csv'

RELEVANT_ATTRIBUTES = ['ID', 'Email', 'Phone', 'Name']  

def load_data(file1, file2):
    # Load the datasets
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    return df1, df2

# compare words with Jaro-Winkler similarity
def jaro_similarity(str1, str2):
    # Asegurarse de que los valores sean cadenas y no sean None
    if str1 is None or str2 is None:
        return 0.0
    else:
        value = fuzz.WRatio(str(str1), str(str2)) / 100.0
    return value

# generate all pairs
def generate_pairs(df1, df2, relevant_attributes):
    if relevant_attributes is None:
        pairs = pd.DataFrame(list(product(df1.to_dict('records'), df2.to_dict('records'))), columns=['record1', 'record2'])
    else:
        df1_relevant = df1[relevant_attributes]
        df2_relevant = df2[relevant_attributes]
        pairs = pd.DataFrame(list(product(df1_relevant.to_dict('records'), df2_relevant.to_dict('records'))), columns=['record1', 'record2'])
    return pairs

# Aplicar similitudes a pares necesarios
def apply_similarity(pairs, relevant_attributes):
    pairs['is_match'] = 0
    for index, row in pairs.iterrows():
        record1 = row['record1']
        record2 = row['record2']
        
        # Calcular la similitud solo para los atributos relevantes
        similarities = []
        for attr in relevant_attributes:
            similarity = jaro_similarity(record1.get(attr), record2.get(attr))
            similarities.append(similarity)
        
        # Promedio de similitudes
        avg_similarity = sum(similarities) / len(similarities)
        
        if avg_similarity >= SIMILARITY_THRESHOLD:
            pairs.loc[index, 'is_match'] = 1

    return pairs

# Filtrar resultados con coincidencias altas
def filter_matches(pairs):
    matches = pairs[pairs['is_match'] == 1]
    return matches

# Evaluar precisión, recall y F1-score
def evaluate_results(num_matches, num_true_matches):
    # Recall
    recall = num_true_matches / num_matches if num_matches != 0 else 0
    if recall > 1:
        recall = 1
    # Precision
    precision = num_matches / num_true_matches if num_true_matches != 0 else 0
    # F1-Score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return precision, recall, f1

def main():
    df1, df2 = load_data(path_baseA, path_baseB)
    
    # Main String Similarity Workflow
    relevant_pairs = generate_pairs(df1, df2, RELEVANT_ATTRIBUTES)
    
    aplied_relevant = apply_similarity(relevant_pairs, RELEVANT_ATTRIBUTES)
    print('String Similarity Applied Successfully!')

    matches_relevant = filter_matches(aplied_relevant)
    matches_uniques_a = matches_relevant.drop_duplicates(subset=['record1'])

    print('Matches ALL:' + str(len(matches_relevant)))
    print('Matches Unique A:' + str(len(matches_uniques_a)))
    
    num_matches_all = len(matches_uniques_a)
    precision_a, recall_a, f1_a = evaluate_results(num_matches_all, 445)
    
    print(f"String All Similarity Results - Precision: {precision_a}, Recall: {recall_a}, F1-Score: {f1_a}")

if __name__ == '__main__':
    main()