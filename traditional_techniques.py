import pandas as pd
from itertools import product
from rapidfuzz import fuzz, process
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuración de umbral de similitud
SIMILARITY_THRESHOLD = 0.75

path_baseA = 'BaseA_cleaned.csv'
path_baseB= 'BaseB_cleaned.csv'


def load_data(file1, file2):
    # Load the datasets
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    return df1, df2

# compare words with Jaro-Winkler similarity
def jaro_similarity(str1, str2):
    # Asegurarse de que los valores sean cadenas y no sean None
    if str1 is None or str2 is None:
        #print('None values found!')
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
def apply_similarity(pairs):
    #print(pairs)
    pairs['is_match'] = 0
    for index, row in pairs.iterrows():
        record1 = row['record1']
        record2 = row['record2']
        ##print(f"Record1: {record1}, Record2: {record2}")
        #print(record1)
        similarity = jaro_similarity(record1, record2)
        #print(f"Similarity: {similarity}")
        if similarity >= SIMILARITY_THRESHOLD:
            #print(f"Match found! Similarity: {similarity}")
            pairs.loc[index, 'is_match'] = 1
            #next row
            continue

    return pairs

# Filtrar resultados con coincidencias altas
def filter_matches(pairs):
    matches = pairs[pairs['is_match'] == 1]
    return matches

# Evaluar precisión, recall y F1-score
def evaluate_results(num_matches, num_true_matches):
    # Recall
    recall = num_true_matches / num_matches
    if recall > 1:
        recall = 1
    # Precision
    precision = num_matches / num_true_matches
    # F1-Score
    f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1

def save_matching_records(matches, file_path):
    with open(file_path, 'w') as f:
        for _, row in matches.iterrows():
            record1 = row['record1']
            record2 = row['record2']
            f.write(f"Match Found:\nRecord1: {record1}\nRecord2: {record2}\n\n")

def main():
    df1, df2 = load_data(path_baseA, path_baseB)
    #print(df1.head())
    #print(df2.tail())
    
    # Main String Similarity Workflow
    all_pairs = generate_pairs(df1, df2, None)
    #print(all_pairs)
    #print(all_pairs.head())
    #print(relevant_pairs.tail())

    aplied_all = apply_similarity(all_pairs)  # Comparar por columna 'name'
    print('String Similarity Applied Successfully!')

    matches_all = filter_matches(aplied_all)
    matches_uniques_a = matches_all.drop_duplicates(subset=['record1'])

    output_file = 'string_similarity_results.txt'
    print('Matches ALL:' + str(len(matches_all)))
    print('Matches Unique A:' + str(len(matches_uniques_a)))
    save_matching_records(matches_uniques_a, output_file)

    
    
    num_matches_all = len(matches_uniques_a)
    precision_a, recall_a, f1_a = evaluate_results(num_matches_all, 445)

    
    print(f"String All Similarity Results - Precision: {precision_a}, Recall: {recall_a}, F1-Score: {f1_a}")

if __name__ == '__main__':
    main()