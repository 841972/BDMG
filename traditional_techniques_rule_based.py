import pandas as pd
from fuzzywuzzy import fuzz


def match_exact(field_a, field_b):
    return field_a == field_b

def match_fuzzy(field_a, field_b, threshold=85):
    similarity_score = fuzz.ratio(field_a, field_b)
    return similarity_score >= threshold

# match if field_a is exact match with field_b or field_c is exact match with field_d and field_e is fuzzy match with field_f
def compound_match(field_a, field_b, field_c, field_d, field_e, field_f):
    return match_exact(field_a, field_b) or (match_exact(field_c, field_d) and match_fuzzy(field_e, field_f))

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
                #if row1['Phone'][0] == row2['Phone'][0]:
                    blocked_pairs.append((idx1, idx2))

    #print(f"Number of blocked pairs: {len(blocked_pairs)}")
    return blocked_pairs

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

def rule_based_matching(blocked_pairs,df1, df2):
    # Initialize list to store matched pairs
    matched_pairs = []

    for pair in blocked_pairs:
        idx1, idx2 = pair
        row1 = df1.loc[idx1]
        row2 = df2.loc[idx2]

        if compound_match(row1['Phone'], row2['Phone'], row1['ID'], row2['ID'], row1['Name'], row2['Name']):
            matched_pairs.append((idx1, idx2))
    
    return matched_pairs


#def choose_rigth_pair(pair, unique_pairs):
#    return

def remove_duplicates(matched_pairs):
    seen = set()
    unique_pairs = []
    duplicates = {}
    
    for pair in matched_pairs:
        if pair[0] not in seen:
            unique_pairs.append(pair)
            seen.add(pair[0])
        #else:
            #duplicates[pair[0]] = pair[1]
            #print('Repetido:',pair[0])

    eliminates = len(matched_pairs) - len(unique_pairs)
    return unique_pairs, eliminates

def main():

    # load data
    df1 = pd.read_csv("./BaseA_cleaned.csv")
    df2 = pd.read_csv("./BaseB_cleaned.csv")

    # Apply blocking
    blocked_pairs = apply_blocking(df1, df2)

    # Apply rule-based matching
    matched_pairs = rule_based_matching(blocked_pairs, df1, df2)

    #drop duplicates on matched_pairs
    unique_pairs, eliminates  = remove_duplicates(matched_pairs)

    # Write matched pairs to a text file
    with open("matches_rule_based.txt", "w") as file:
        if unique_pairs:
            file.write("Matched Pairs:\n")
            for match in unique_pairs:
                idx1, idx2 = match
                file.write(f"Record from df1 (index {idx1} = {df1.loc[idx1]['Name']}, {df1.loc[idx1]['Phone']}, {df1.loc[idx1]['ID']}) matches with record from df2 (index {idx2} = {df2.loc[idx2]['Name']}, {df2.loc[idx2]['Phone']}, {df2.loc[idx2]['ID']})\n")
        else:
            file.write("No matches found.\n")
        print("No matches found.")

    print('Eliminados:',eliminates)
    print('Unique matches:', len(unique_pairs))
    print ('Total matches:', len(matched_pairs))

    num_matches_all = len(unique_pairs)
    precision_a, recall_a, f1_a = evaluate_results(num_matches_all, 445)

    
    print(f"Rule Based Results - Precision: {precision_a}, Recall: {recall_a}, F1-Score: {f1_a}")



if __name__ == "__main__":
    main()
