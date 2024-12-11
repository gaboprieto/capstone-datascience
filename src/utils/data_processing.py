import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def process_course_data(course_data):
    course_data = course_data.dropna()
    features = course_data.drop(
        [
            "user_id",
            "course_id",
            "course_name",
            "team_id",
            "team_name",
            "semester",
            "year",
            "analysis_dates",
            "checkpoint",
        ],
        axis=1,
    )

    print(features.head())
    target = course_data["team_id"]
    user_ids = course_data["user_id"]

    # Convert features to numeric, replacing any non-numeric values with NaN
    features = features.apply(pd.to_numeric, errors="coerce")

    # Drop any rows with NaN values
    valid_rows = features.notna().all(axis=1)
    features = features[valid_rows]
    target = target[valid_rows]
    user_ids = user_ids[valid_rows]

    # Drop constant columns
    features = features.loc[:, features.var() != 0]

    # Drop rows with all zeros
    non_zero_rows = (features != 0).any(axis=1)
    features = features[non_zero_rows]
    target = target[non_zero_rows]
    user_ids = user_ids[non_zero_rows]

    if features.empty or features.shape[1] == 0:
        return None, None, None

    scaler = MinMaxScaler()
    normalized_features = pd.DataFrame(
        scaler.fit_transform(features), columns=features.columns
    )

    # Add error checking
    print(f"Features shape: {normalized_features.shape}")
    print(f"Features data type: {normalized_features.dtypes.unique()}")
    print(f"Target shape: {target.shape}")
    print(f"User IDs shape: {user_ids.shape}")

    return normalized_features, target, user_ids


def get_data_from_csv(file_path):
    return pd.read_csv(file_path)


def reorder_similarity_matrix(similarity_matrix, labels):
    order = np.argsort(labels)
    return similarity_matrix[order, :][:, order]
