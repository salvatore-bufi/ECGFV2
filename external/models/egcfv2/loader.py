import pandas as pd
import ast

filename_list = ['decision_path_entropy_1.tsv', 'decision_path_entropy_5.tsv', 'decision_path_entropy_10.tsv',
                 'decision_path_gini_1.tsv', 'decision_path_gini_5.tsv', 'decision_path_gini_10.tsv']

filename_list2 = ['./external/models/egcfv2/data/decision_path_entropy_1.tsv',
                  './external/models/egcfv2/data/decision_path_entropy_5.tsv',
                  './external/models/egcfv2/data/decision_path_entropy_10.tsv',
                  './external/models/egcfv2/data/decision_path_gini_1.tsv',
                  './external/models/egcfv2/data/decision_path_gini_5.tsv',
                  './external/models/egcfv2/data/decision_path_gini_10.tsv']


def load_decision_path(filename=filename_list[0], default_path=True):
    """
    Load the tsv dataset - [ user | feature_path | item ]
    :param filename: name of the dataset
    :return: pandas dataset
    """
    if default_path:
        path = './external/models/egcfv2/data/' + filename
    else:
        path = filename
    dp = pd.read_csv(path, sep='\t')
    dp = dp.astype({"user": "int", "item": "int"})
    dp['feature_path'] = dp['feature_path'].apply(lambda x: ast.literal_eval(x))
    return dp


def load_decision_path_all(filename_list=filename_list, default_path=True):
    """
     :param filename: list of filenames paths es: ['file_name_1', 'file_file_name2']
    :return: dictionary of dataframes key=file_name, value = file_dataframe es: ['file_name_1': dg_1, 'file_name_2': df2]
    """
    decision_path_dict = dict()
    if default_path:
        for i in range(len(filename_list)):
            decision_path_dict[filename_list[i]] = load_decision_path(filename_list[i])
    else:
        for i in range(len(filename_list)):
            name = filename_list[i].split('/')[-1]
            decision_path_dict[name] = load_decision_path(filename_list[i], default_path=False)
    return decision_path_dict


