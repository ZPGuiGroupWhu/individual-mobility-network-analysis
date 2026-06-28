import pandas as pd
import ast


def ensure_list_column(df, col_name, verbose=True):
    """
    Ensure that a DataFrame column contains list values.

    Args:
        df (pd.DataFrame): Input DataFrame.
        col_name (str): Column to convert.
        verbose (bool): Whether to print parse warnings.

    Returns:
        pd.DataFrame: The same DataFrame with ``col_name`` converted to lists.
    """

    def _convert(x):
        if isinstance(x, list):
            return x
        if pd.isna(x) or x is None:
            return []
        if isinstance(x, str):
            try:
                val = ast.literal_eval(x)
                if isinstance(val, list):
                    return val
                else:
                    if verbose:
                        print(f"[WARN] Non-list string in {col_name}: {x}")
                    return []
            except Exception:
                if verbose:
                    print(f"[WARN] Failed to parse string in {col_name}: {x}")
                return []
        return []

    df[col_name] = df[col_name].apply(_convert)
    return df


def ensure_dict_column(df, col_name, verbose=True):
    """
    Ensure that a DataFrame column contains dict values.

    Args:
        df (pd.DataFrame): Input DataFrame.
        col_name (str): Column to convert.
        verbose (bool): Whether to print parse warnings.

    Returns:
        pd.DataFrame: The same DataFrame with ``col_name`` converted to dicts.
    """

    def _convert(x):
        if isinstance(x, dict):
            return x
        if pd.isna(x) or x is None:
            return {}
        if isinstance(x, str):
            try:
                val = ast.literal_eval(x)
                if isinstance(val, dict):
                    return val
                else:
                    if verbose:
                        print(f"[WARN] Non-dict string in {col_name}: {x}")
                    return {}
            except Exception:
                if verbose:
                    print(f"[WARN] Failed to parse string in {col_name}: {x}")
                return {}
        return {}

    df[col_name] = df[col_name].apply(_convert)
    return df
