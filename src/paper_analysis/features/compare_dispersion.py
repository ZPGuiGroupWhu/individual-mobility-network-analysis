"""Calculate dispersion summaries for processed feature tables."""

import pandas as pd
import numpy as np
import os

SPATIAL_COLUMNS = [
    'number_of_trips',
    'number_of_locations',
    'trip_length',
    'radius_of_gyration'
]

TOPO_COLUMNS = [
    'average_degree',
    'degree_parament1',
    'average_clustering_coefficient',
    'average_shortest_path_length',
    'degree_r',
    'visit_count_r'
]

def calculate_mad_sampled(values, max_samples=10000):
    """
    Estimate mean absolute difference, sampling when arrays are large.
    """
    n = len(values)
    if n <= max_samples:
        sample1 = values
        sample2 = values
    else:
        rng = np.random.default_rng()
        sample1 = rng.choice(values, size=max_samples, replace=False)
        sample2 = rng.choice(values, size=max_samples, replace=False)

    diff_matrix = sample1[:, np.newaxis] - sample2
    return np.mean(np.abs(diff_matrix))

def calculate_statistics(series: pd.Series, df=None, column_name=None):
    """
    Calculate summary statistics for one feature column, ignoring missing values.
    """
    if column_name == 'degree_parament1' and df is not None:
        mask = df['degree_distribution'] == 'Truncated Power Law'
        series = series[mask]

    clean_values = series.dropna().values

    if len(clean_values) == 0:
        return pd.Series({
            'Mean': np.nan,
            'Median': np.nan,
            'Standard Deviation': np.nan,
            'Interquartile Range': np.nan,
            'Coefficient of Variation': np.nan,
            'Quartile Coefficient of Dispersion': np.nan,
            'Mean Absolute Difference': np.nan,
            'Relative Mean Absolute Difference': np.nan,
            'Mean Absolute Deviation': np.nan,
            'Relative Mean Absolute Deviation': np.nan
        })

    mean = np.mean(clean_values)
    median = np.median(clean_values)
    std_dev = np.std(clean_values)

    mean_abs= np.mean(np.abs(clean_values))
    q1, q3 = np.percentile(clean_values, [25, 75])
    iqr = q3 - q1

    cv = std_dev / mean_abs if mean_abs != 0 else np.nan

    qcd = iqr / (np.abs(q1) + np.abs(q3)) if (np.abs(q1) + np.abs(q3)) != 0 else np.nan

    mad_diff = calculate_mad_sampled(clean_values)
    rmad_diff = mad_diff / mean_abs if mean_abs != 0 else np.nan

    mad_dev = np.mean(np.abs(clean_values - mean))
    rmad_dev = mad_dev / mean_abs if mean_abs != 0 else np.nan

    # dispersion index
    var=np.var(clean_values)
    minx=np.min(clean_values)
    tr=mean-minx
    disp_index=(var-tr)/(var+tr)

    return pd.Series({
        'Mean': mean,
        'Median': median,
        'Standard Deviation': std_dev,
        'Interquartile Range': iqr,
        'Coefficient of Variation': cv,
        'Quartile Coefficient of Dispersion': qcd,
        'Mean Absolute Difference': mad_diff,
        'Relative Mean Absolute Difference': rmad_diff,
        'Mean Absolute Deviation': mad_dev,
        'Relative Mean Absolute Deviation': rmad_dev,
        'Dispersion Index': disp_index
    })

def calculate_dataframe_statistics(df: pd.DataFrame, columns: list, feature_type: str, dataset_name: str):
    """
    Calculate summary statistics for selected columns in a DataFrame.
    """
    stats_dict = {}

    for column in columns:
        stats = calculate_statistics(df[column], df if feature_type == 'topological' else None, column)
        stats['feature_type'] = feature_type
        stats['dataset'] = dataset_name
        stats['feature_name'] = column
        stats_dict[column] = stats

    return pd.DataFrame.from_dict(stats_dict, orient='index')

def process_dataset(dataset_name: str, spatial_path, topo_path,
                    spatial_columns=None, topo_columns=None):
    """
    Process spatial and topological feature tables for one dataset.
    """

    spatial_columns = spatial_columns or SPATIAL_COLUMNS
    topo_columns = topo_columns or TOPO_COLUMNS

    try:
        spatial_df = pd.read_csv(spatial_path, usecols=spatial_columns)
        topo_df = pd.read_csv(topo_path, usecols=topo_columns + ['degree_distribution'])

        spatial_stats = calculate_dataframe_statistics(spatial_df, spatial_columns, 'spatial', dataset_name)
        topo_stats = calculate_dataframe_statistics(topo_df, topo_columns, 'topological', dataset_name)

        return spatial_stats, topo_stats
    except Exception as e:
        print(f"Error processing dataset {dataset_name}: {str(e)}")
        return None, None

def process_datasets(
        dataset_combinations=None,
        spatial_path_template=None,
        topo_path_template=None,
        output_path=None,
        spatial_columns=None,
        topo_columns=None,
):
    """
    Calculate dispersion summaries for selected dataset combinations.

    Parameters:
    -----------
    dataset_combinations : list of list
        Dataset groups to summarize. Each inner list is combined before
        calculating dispersion statistics.
    """
    spatial_columns = spatial_columns or SPATIAL_COLUMNS
    topo_columns = topo_columns or TOPO_COLUMNS
    np.random.seed(42)

    if dataset_combinations is None:
        dataset_combinations = [
            ['D1_YJMob100K'],
            ['D3_FourSuqare'],
            ['D1_YJMob100K', 'D3_FourSuqare']
        ]

    if spatial_path_template is None:
        spatial_path_template = os.path.join(
            'results', 'intermediate', '{dataset}', 'spatial_features', 'processed_spatial_stats.csv'
        )
    if topo_path_template is None:
        topo_path_template = os.path.join(
            'results', 'intermediate', '{dataset}', 'topological_features', 'processed_topological_stats.csv'
        )
    if output_path is None:
        output_path = os.path.join('results', 'tables', 'dispersion_combination.csv')

    all_datasets = list(set([dataset for sublist in dataset_combinations for dataset in sublist]))

    spatial_dfs = []
    topo_dfs = []

    for dataset in all_datasets:
        try:
            spatial_df = pd.read_csv(spatial_path_template.format(dataset=dataset),
                                  usecols=spatial_columns)
            topo_df = pd.read_csv(topo_path_template.format(dataset=dataset),
                                usecols=topo_columns + ['degree_distribution'])

            spatial_df['dataset'] = dataset
            topo_df['dataset'] = dataset

            spatial_dfs.append(spatial_df)
            topo_dfs.append(topo_df)
        except Exception as e:
            print(f"Error reading dataset {dataset}: {str(e)}")
            continue

    all_spatial_df = pd.concat(spatial_dfs, ignore_index=True)
    all_topo_df = pd.concat(topo_dfs, ignore_index=True)

    all_results = []

    for datasets in dataset_combinations:
        combination_name = '+'.join(d for d in datasets)

        spatial_mask = all_spatial_df['dataset'].isin(datasets)
        topo_mask = all_topo_df['dataset'].isin(datasets)

        combined_spatial_df = all_spatial_df[spatial_mask].copy()
        combined_topo_df = all_topo_df[topo_mask].copy()

        del combined_spatial_df['dataset']
        del combined_topo_df['dataset']

        spatial_stats = calculate_dataframe_statistics(
            combined_spatial_df, spatial_columns, 'spatial', combination_name)
        topo_stats = calculate_dataframe_statistics(
            combined_topo_df, topo_columns, 'topological', combination_name)

        all_results.extend([spatial_stats, topo_stats])

    if not all_results:
        print("No results were generated. Please check the datasets and file paths.")
        return

    final_results = pd.concat(all_results, ignore_index=True)

    column_order = [
        'dataset',
        'feature_type',
        'feature_name',
        'Mean',
        'Median',
        'Standard Deviation',
        'Interquartile Range',
        'Mean Absolute Difference',
        'Mean Absolute Deviation',
        'Coefficient of Variation',
        'Quartile Coefficient of Dispersion',
        'Relative Mean Absolute Difference',
        'Relative Mean Absolute Deviation',
        'Dispersion Index'
    ]

    final_results = final_results[column_order]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_results.to_csv(output_path, index=False)
    print(f"Statistics have been saved to {output_path}")


