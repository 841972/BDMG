import pandas as pd
from itertools import product
from rapidfuzz import fuzz, process

# Configuración de umbral de similitud
SIMILARITY_THRESHOLD = 0.85
path_baseA = 'BaseA_cleaned.csv'
path_baseB= 'BaseB_cleaned.csv'

output_file = 'matching_records.txt'

def load_data(file1, file2):
    # Load the datasets
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    return df1, df2

def jaro_similarity(str1, str2):
    # Ensure both inputs are strings and not None
    if pd.isna(str1) or pd.isna(str2):
       # print('None values found!')
        return 0.0
    return fuzz.WRatio(str(str1), str(str2)) / 100.0

def generate_pairs(df1, df2, relevant_attributes):
    if relevant_attributes is None:
        pairs = pd.DataFrame(list(product(df1.to_dict('records'), df2.to_dict('records'))), columns=['record1', 'record2'])
    else:
        df1_relevant = df1[relevant_attributes]
        df2_relevant = df2[relevant_attributes]
        pairs = pd.DataFrame(list(product(df1_relevant.to_dict('records'), df2_relevant.to_dict('records'))), columns=['record1', 'record2'])
    return pairs

def apply_similarity(pairs, keys):
    pairs['is_match'] = 0
    pairs['match_reason'] = ''

    for index, row in pairs.iterrows():
        record1 = row['record1']
        record2 = row['record2']

        for key in keys:
            sim = jaro_similarity(record1.get(key, ''), record2.get(key, ''))
            if sim >= SIMILARITY_THRESHOLD:
                pairs.loc[index, 'is_match'] = 1
                pairs.loc[index, 'match_reason'] += f'{key}_match;'
                break

    return pairs

def filter_matches(pairs):
    return pairs[pairs['is_match'] == 1]

def evaluate_results(num_matches, num_true_matches):
    # Recall
    recall = num_true_matches / num_matches
    if recall > 1:
        recall = 1
    # Precision
    precision = num_matches / num_true_matches
    if precision > 1:
        precision = 1
    # F1-Score
    f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1

def save_matching_records(matches, file_path):
    with open(file_path, 'w') as f:
        for _, row in matches.iterrows():
            record1 = row['record1']
            record2 = row['record2']
            reason = row['match_reason']
            f.write(f"Match Found:\nRecord1: {record1}\nRecord2: {record2}\nReason: {reason}\n\n")

def main():
    df1, df2 = load_data(path_baseA, path_baseB)
    # print(df1.head())
    # print(df2.tail())
    relevant_keys = ['ID', 'Name', 'Email', 'Phone', 'Address']

    # Generate pairs
    all_pairs = generate_pairs(df1, df2, None)
    print('Pairs Generated Successfully!')
    #print(all_pairs.head())

    # Apply similarity checks
    matched_pairs = apply_similarity(all_pairs, relevant_keys)
    print('String Similarity Applied Successfully!')

    # Filter matches
    matches = filter_matches(matched_pairs)
    unique_matches = matches.drop_duplicates(subset=['record1'])

    print('Matches ALL:', len(matches))
    print('Matches Unique A:', len(unique_matches))

    # Save matching records to file
    save_matching_records(unique_matches, output_file)
    print(f'Matching records saved to {output_file}!')

    num_matches = len(unique_matches)
    precision, recall, f1_score = evaluate_results(num_matches, 445)
    print(f'Precision: {precision}, Recall: {recall}, F1-Score: {f1_score}')

if __name__ == '__main__':
    main()
