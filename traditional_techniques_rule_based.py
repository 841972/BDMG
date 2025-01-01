import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from rapidfuzz.distance import JaroWinkler
from globals import *

def match_exact(field_a, field_b):
    return field_a == field_b

def match_fuzzy(field_a, field_b, threshold=SIMILARITY_THRESHOLD):
    similarity_score = JaroWinkler.similarity(field_a, field_b)
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


def main():

    # load data
    df1 = pd.read_csv(PATH_BASEA)
    df2 = pd.read_csv(PATH_BASEB)

    # Apply blocking
    blocked_pairs = apply_blocking(df1, df2)

    # Apply rule-based matching
    matched_pairs = rule_based_matching(blocked_pairs, df1, df2)

    #drop duplicates on matched_pairs

    # Write matched pairs to a text file
    index_machted=[]
    with open("matches_rule_based.txt", "w") as file:
        if matched_pairs:
            file.write("Matched Pairs:\n")
            for match in matched_pairs:
                idx1, idx2 = match
                file.write(f"{match}\n")
                index_machted.append((match, 1))
        else:
            file.write("No matches found.\n")

    index_machted_df = pd.DataFrame(index_machted, columns=['index_pair', 'is_match'])
    precision_a, recall_a, f1_a = evaluate_results(index_machted_df, TRUE_MATCHES)

    
    print(f"Rule Based Results - Precision: {precision_a}, Recall: {recall_a}, F1-Score: {f1_a}")



if __name__ == "__main__":
    main()
