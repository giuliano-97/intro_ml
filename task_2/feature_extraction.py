## Libraries
# General purpose libraries
import pandas as pd
import numpy as np
# Import helpers
from helpers import *

## Execution parameters
# Sparsity threshold
avg_sparsity_threshold = 8.0
# Records per patient
n_records_per_patient = 12

## Functions
def getUniqueUnsortedPidsFromRecords(duplicatePidList):
    size = len(duplicatePidList)
    uniques = []
    i = 0
    while(i < size):
        uniques.append(duplicatePidList[i])
        i += n_records_per_patient
    return uniques

def getTrueOrderIndex(sortedList, unsortedList):
    trueOrderIndex = []
    for idx in sortedList:
        trueOrderIndex.append(unsortedList.index(idx))
    return trueOrderIndex

# Perform features extraction
def feature_extraction(input_file,output_files):
    data = pd.read_csv(input_file)
    unsortedPids = getUniqueUnsortedPidsFromRecords(data['pid'].values.tolist())

    # Group data by pid - each group contains all the data
    # with the same pid i.e. corresponding to the same patient
    grouped = data.groupby(['pid'])

    # Dict containing the average number of nan in a series
    # for each type of data
    nan_stats = {}

    # Compute NaN stats
    for col in data.columns[3:]:
        # Apply the same function to each group ---> obtain a pandas.Series containing
        # the time series corresponding to "col" for each patient, then compute nan count
        # for each series
        nan_count=\
            grouped[col].apply(lambda x: pd.Series(x.values)).unstack().isna().sum(axis=1)
        # Compute nan count summary stats
        nan_stats[col] = {'mean':nan_count.mean(), 'std':nan_count.std()}

    # Per column summary stats
    col_sumstats = {}
    for col in data.columns[3:]:
        stats_dict = {}
        # Compute column range
        stats_dict['min'] = data[col].min()
        stats_dict['max'] = data[col].max()
        col_sumstats[col] = stats_dict

    # Define aggregation functions dictionaries for the generation of:
    # Classfication features
    clf_aggregation_dict = {'Age': pd.NamedAgg(column='Age', aggfunc=(lambda x : x[0]))}
    # Regression features
    rgr_aggregation_dict = {'Age': pd.NamedAgg(column='Age', aggfunc=(lambda x : x[0]))}
    for col in data.columns[3:]:
        if nan_stats[col]['mean'] < avg_sparsity_threshold:
            # Classification features
            clf_aggregation_dict[col+'_mean'] = pd.NamedAgg(column=col, aggfunc=np.nanmean)
            clf_aggregation_dict[col+'_std'] = pd.NamedAgg(column=col, aggfunc=(nth_moment(2)))
            clf_aggregation_dict[col+'_median'] = pd.NamedAgg(column=col, aggfunc=np.nanmedian)
            clf_aggregation_dict[col+'_min'] = pd.NamedAgg(column=col, aggfunc=np.nanmin)
            clf_aggregation_dict[col+'_max'] = pd.NamedAgg(column=col, aggfunc=np.nanmax)
            clf_aggregation_dict[col+'_lfdiff'] = pd.NamedAgg(column=col, aggfunc=last_first_diff)
            clf_aggregation_dict[col+'_lfslope'] = pd.NamedAgg(column=col, aggfunc=last_first_slope)
            clf_aggregation_dict[col+'_tcount'] = pd.NamedAgg(column=col,aggfunc='count')
            # Regression features
            rgr_aggregation_dict[col+'_mean'] = pd.NamedAgg(column=col, aggfunc=np.nanmean)
            rgr_aggregation_dict[col+'_std'] = pd.NamedAgg(column=col, aggfunc=(nth_moment(2)))
            rgr_aggregation_dict[col+'_median'] = pd.NamedAgg(column=col, aggfunc=np.nanmedian)
            rgr_aggregation_dict[col+'_min'] = pd.NamedAgg(column=col, aggfunc=np.nanmin)
            rgr_aggregation_dict[col+'_max'] = pd.NamedAgg(column=col, aggfunc=np.nanmax)
            rgr_aggregation_dict[col+'_lfdiff'] = pd.NamedAgg(column=col, aggfunc=last_first_diff)
            rgr_aggregation_dict[col+'_lfslope'] = pd.NamedAgg(column=col, aggfunc=last_first_slope)
            rgr_aggregation_dict[col+'_tcount'] = pd.NamedAgg(column=col,aggfunc='count')
        else:
            # Count number measurements for sparse series
            clf_aggregation_dict[col+'_mean'] = pd.NamedAgg(column=col, aggfunc=np.nanmean)
            clf_aggregation_dict[col+'_tcount'] = pd.NamedAgg(column=col,aggfunc='count')
            rgr_aggregation_dict[col+'_mean'] = pd.NamedAgg(column=col, aggfunc=np.nanmean)
            rgr_aggregation_dict[col+'_tcount'] = pd.NamedAgg(column=col,aggfunc='count')

    # Build features by aggregating
    clf_features = grouped.agg(**clf_aggregation_dict)
    rgr_features = grouped.agg(**rgr_aggregation_dict)
    # Fill NaNs with column mean
    clf_features.fillna(clf_features.mean(), inplace=True)
    rgr_features.fillna(rgr_features.mean(), inplace=True)

    # Restore order and write classification features to file
    clf_features.reindex(unsortedPids).to_csv(output_files[0])
    # Write regression features to file
    rgr_features.reindex(unsortedPids).to_csv(output_files[1])

## Code
def main():
    feature_extraction(train_file,train_features_files)
    feature_extraction(test_file,test_features_files)

if __name__ == '__main__':
    main()
