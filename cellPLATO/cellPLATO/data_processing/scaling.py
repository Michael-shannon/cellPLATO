from initialization.config import *

import numpy as np
import pandas as pd
from typing import Optional, List

from sklearn.preprocessing import MinMaxScaler, StandardScaler


def scale_features(
    df: pd.DataFrame,
    factors: List[str],
    method: str = SCALING_METHOD,
    average_time_windows: bool = AVERAGE_TIME_WINDOWS,
    factors_to_transform: Optional[List[str]] = None,
    factors_not_to_transform: Optional[List[str]] = None,
    verbose: bool = False,
):
    """
    Center/scale features according to a single, shared implementation
    used by both DR and clustering.

    Returns (X_scaled: np.ndarray, used_columns: list[str])
    """
    if average_time_windows:
        selected_columns = [f"{f}_tmean" for f in factors]
    else:
        selected_columns = list(factors)

    sub_df = df[selected_columns]
    X = sub_df.values

    if method in ['minmax']:
        X_scaled = MinMaxScaler().fit_transform(X)
        return X_scaled, selected_columns

    if method in ['standard', 'standardscaler']:
        X_scaled = StandardScaler().fit_transform(X)
        return X_scaled, selected_columns

    if method == 'log2minmax':
        negative_cols: list[str] = []
        positive_cols: list[str] = []
        for col in selected_columns:
            if np.min(df[col]) < 0:
                negative_cols.append(col)
            else:
                positive_cols.append(col)

        pos_df = df[positive_cols]
        neg_df = df[negative_cols]
        pos_x = pos_df.values if len(positive_cols) else np.empty((len(df), 0))
        neg_x = neg_df.values if len(negative_cols) else np.empty((len(df), 0))

        if neg_x.shape[1] == 0:
            neg_x_scaled = neg_x
        else:
            neg_x_scaled = MinMaxScaler().fit_transform(neg_x)

        if pos_x.shape[1] == 0:
            pos_x_scaled = pos_x
        else:
            pos_x_constant = pos_x + 1e-6
            pos_x_log = np.log2(pos_x_constant)
            pos_x_scaled = MinMaxScaler().fit_transform(pos_x_log)

        X_scaled = np.concatenate((pos_x_scaled, neg_x_scaled), axis=1)
        used_cols = positive_cols + negative_cols
        return X_scaled, used_cols

    if method == 'choice':
        # Determine columns to transform vs not
        if factors_to_transform is not None and factors_not_to_transform is not None:
            to_log = [f for f in selected_columns if f in factors_to_transform]
            not_log = [f for f in selected_columns if f in factors_not_to_transform]
            unassigned = [f for f in selected_columns if f not in to_log and f not in not_log]
            to_log.extend(unassigned)
        elif factors_not_to_transform is not None:
            not_log = [f for f in selected_columns if f in factors_not_to_transform]
            to_log = [f for f in selected_columns if f not in not_log]
        elif factors_to_transform is not None:
            to_log = [f for f in selected_columns if f in factors_to_transform]
            not_log = [f for f in selected_columns if f not in to_log]
        else:
            # Defaults mirror existing pipeline behavior
            if average_time_windows:
                default_not = ['arrest_coefficient_tmean', 'rip_L_tmean', 'rip_p_tmean', 'rip_K_tmean', 'eccentricity_tmean', 'orientation_tmean', 'directedness_tmean', 'turn_angle_tmean', 'dir_autocorr_tmean', 'glob_turn_deg_tmean']
            else:
                default_not = ['arrest_coefficient', 'rip_L', 'rip_p', 'rip_K', 'eccentricity', 'orientation', 'directedness', 'turn_angle', 'dir_autocorr', 'glob_turn_deg']
            not_log = [f for f in selected_columns if f in default_not]
            to_log = [f for f in selected_columns if f not in not_log]

        trans_df = df[to_log] if len(to_log) else pd.DataFrame(index=df.index)
        nontrans_df = df[not_log] if len(not_log) else pd.DataFrame(index=df.index)

        trans_x = trans_df.values if len(to_log) else np.empty((len(df), 0))
        nontrans_x = nontrans_df.values if len(not_log) else np.empty((len(df), 0))

        if trans_x.shape[1] > 0:
            trans_x_constant = trans_x + 1e-6
            trans_x_log = np.log2(trans_x_constant)
            trans_x_scaled = MinMaxScaler().fit_transform(trans_x_log)
        else:
            trans_x_scaled = trans_x

        if nontrans_x.shape[1] > 0:
            nontrans_x_scaled = MinMaxScaler().fit_transform(nontrans_x)
        else:
            nontrans_x_scaled = nontrans_x

        X_scaled = np.concatenate((trans_x_scaled, nontrans_x_scaled), axis=1)
        used_cols = to_log + not_log
        return X_scaled, used_cols

    if method == 'powertransformer':
        from sklearn.preprocessing import PowerTransformer
        pt = PowerTransformer(method='yeo-johnson')
        X_scaled = pt.fit_transform(X)
        return X_scaled, selected_columns

    if method in ['robust', 'robustscaler']:
        from sklearn.preprocessing import RobustScaler
        X_scaled = RobustScaler().fit_transform(X)
        return X_scaled, selected_columns

    if method in ['normalize', 'normalizer']:
        from sklearn.preprocessing import Normalizer
        X_scaled = Normalizer().fit_transform(X)
        return X_scaled, selected_columns

    if method in ['quantile', 'quantileuniform']:
        from sklearn.preprocessing import QuantileTransformer
        X_scaled = QuantileTransformer(output_distribution='uniform').fit_transform(X)
        return X_scaled, selected_columns

    if method in ['maxabs', 'maxabsscaler']:
        from sklearn.preprocessing import MaxAbsScaler
        X_scaled = MaxAbsScaler().fit_transform(X)
        return X_scaled, selected_columns

    if method in ['yeo-johnson', 'box-cox']:
        from sklearn.preprocessing import PowerTransformer
        pt = PowerTransformer(method='yeo-johnson' if method == 'yeo-johnson' else 'box-cox')
        X_scaled = pt.fit_transform(X)
        return X_scaled, selected_columns

    available = ['minmax', 'standard', 'standardscaler', 'log2minmax', 'choice', 'powertransformer', 'robust', 'normalize', 'quantile', 'maxabs', 'yeo-johnson', 'box-cox']
    raise ValueError(f"Unknown scaling method: '{method}'. Available methods: {available}")


