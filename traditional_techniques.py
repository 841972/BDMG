import pandas as pd
from itertools import product
from sklearn.metrics import precision_score, recall_score, f1_score
from rapidfuzz.distance import JaroWinkler
from globals import SIMILARITY_THRESHOLD, PATH_BASEA, PATH_BASEB, TRUE_MATCHES  


path_baseA = PATH_BASEA
path_baseB= PATH_BASEB


def load_data(file1, file2):
    # Load the datasets
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    return df1, df2

# Apply blocking to reduce the number of comparisons
def apply_blocking(df1, df2):
    # Initialize list to store blocked pairs
    df1['Block'] = df1['Name'].str[0]
    df2['Block'] = df2['Name'].str[0]

    blocked_pairs = []
    for block in df1['Block'].unique():
        df1_block = df1[df1['Block'] == block]
        df2_block = df2[df2['Block'] == block]

        for idx1, row1 in df1_block.iterrows():
            for idx2, row2 in df2_block.iterrows():
                blocked_pairs.append((idx1, idx2))

    return blocked_pairs


# compare words with Jaro-Winkler similarity
def jaro_similarity(str1, str2):
    # Asegurarse de que los valores sean cadenas y no sean None
    if str1 is None or str2 is None:
        #print('None values found!')
        return 0.0
    else: 
        value = JaroWinkler.similarity(str(str1), str(str2)) 
    return value


# Aplicar similitudes a pares necesarios
def apply_similarity(pairs, df1, df2):
    result = []

    # Iterating over all pairs
    for pair in pairs:
        index1, index2 = pair

        # Obtain the records
        record1 = df1.loc[index1]
        record1_str = record1.to_dict()
        record1_str = str(record1_str)

        record2 = df2.loc[index2]
        record2_str = record2.to_dict()
        record2_str = str(record2_str)

        # Calculate the similarity
        similarity = jaro_similarity(record1_str, record2_str)  
        if similarity >= SIMILARITY_THRESHOLD:
            result.append((pair, 1))  # Añadir el par y el valor 1 

    #print(result)
    result_df = pd.DataFrame(result, columns=['index_pair', 'is_match'])
    #print(result_df)
    return result_df

    return pairs


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

def save_matching_records(matches, file_path):
    with open(file_path, 'w') as f:
        for _, row in matches.iterrows():
            record1 = row['record1']
            record2 = row['record2']
            f.write(f"Match Found:\nRecord1: {record1}\nRecord2: {record2}\n\n")

def main():
    df1, df2 = load_data(path_baseA, path_baseB)

    
    # Main String Similarity Workflow
    blocked_pairs = apply_blocking(df1, df2)


    aplied_all = apply_similarity(blocked_pairs, df1, df2)  # Comparar por columna 'name'
    print('String Similarity Applied Successfully!')


    precision_a, recall_a, f1_a = evaluate_results(aplied_all, TRUE_MATCHES)

    
    print(f"String All Similarity Results - Precision: {precision_a}, Recall: {recall_a}, F1-Score: {f1_a}")

if __name__ == '__main__':
    main()