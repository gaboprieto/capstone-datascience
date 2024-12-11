import numpy as np
from sklearn.metrics import pairwise_distances, silhouette_score

from src.clustering.algorithms import apply_clustering
import pandas as pd

from src.visualization.plots import create_comparison_plot


def calculate_clustering_metrics(features, labels, similarity_matrix, original_labels):
    """Calculate metrics for clustering results."""
    print(f"Features shape in metrics: {features.shape}")

    # Print data types correctly based on input type
    if isinstance(features, pd.DataFrame):
        print(f"Features data types: {features.dtypes.unique()}")
        features_array = features.values
    else:
        print(f"Features data type: {features.dtype}")
        features_array = features

    print(f"Labels shape: {labels.shape}")
    print(f"Labels data type: {labels.dtype}")
    print(f"Similarity matrix shape: {similarity_matrix.shape}")

    # Convert labels to numpy array
    labels_array = np.array(labels)

    # Initialize metrics dictionaries
    global_metrics = {
        "avg_similarity": 0,
        "avg_distance_to_centroid": 0,
        "team_assignment_dissimilarity": 0,
    }
    team_metrics = {}

    # Calculate team assignment dissimilarity
    n_samples = len(labels)
    clustered_matrix = np.zeros((n_samples, n_samples))
    original_matrix = np.zeros((n_samples, n_samples))

    # Create binary matrices excluding noise points
    valid_mask = labels != -1
    valid_indices = np.where(valid_mask)[0]
    n_valid = len(valid_indices)

    if n_valid > 1:  # Only proceed if we have valid samples
        # Fill matrices (only upper triangle, excluding diagonal)
        for i in range(n_valid):
            for j in range(i + 1, n_valid):
                idx1, idx2 = valid_indices[i], valid_indices[j]
                # Same cluster in clustering result
                if labels[idx1] == labels[idx2]:
                    clustered_matrix[idx1, idx2] = 1
                    clustered_matrix[idx2, idx1] = 1  # Symmetric
                # Same cluster in original assignment
                if original_labels[idx1] == original_labels[idx2]:
                    original_matrix[idx1, idx2] = 1
                    original_matrix[idx2, idx1] = 1  # Symmetric

        # Calculate dissimilarity (only for valid pairs)
        n_pairs = (n_valid * (n_valid - 1)) // 2  # Number of unique pairs
        if n_pairs > 0:
            # Sum differences and divide by number of valid pairs
            global_metrics["team_assignment_dissimilarity"] = np.sum(
                np.abs(original_matrix - clustered_matrix)
            ) / (
                2 * n_pairs
            )  # Divide by 2 because matrix is symmetric
    else:
        global_metrics["team_assignment_dissimilarity"] = 0.0

    # Calculate per-team metrics
    unique_labels = np.unique(labels[labels != -1])
    total_students = 0

    for label in unique_labels:
        team_mask = labels == label
        team_size = np.sum(team_mask)

        if team_size > 0:
            # Get team features and calculate centroid
            team_features = features_array[team_mask]
            team_centroid = np.mean(team_features, axis=0)

            # Calculate average distance to centroid
            distances_to_centroid = np.mean(
                [np.linalg.norm(student - team_centroid) for student in team_features]
            )

            # Calculate average similarity within team
            team_similarities = similarity_matrix[team_mask][:, team_mask]
            avg_similarity = (
                (np.sum(team_similarities) - team_size) / (team_size * (team_size - 1))
                if team_size > 1
                else 0
            )

            # Store team metrics
            team_metrics[f"Team_{label}"] = {
                "size": team_size,
                "avg_similarity": avg_similarity,
                "avg_distance_to_centroid": distances_to_centroid,
            }

            # Update global metrics
            global_metrics["avg_similarity"] += avg_similarity * team_size
            global_metrics["avg_distance_to_centroid"] += (
                distances_to_centroid * team_size
            )
            total_students += team_size

    # Calculate global averages
    if total_students > 0:
        global_metrics["avg_similarity"] /= total_students
        global_metrics["avg_distance_to_centroid"] /= total_students

    return {"global_metrics": global_metrics, "team_metrics": team_metrics}


def calculate_team_coherence(labels, matrix):
    unique_labels = np.unique(labels)
    coherences = []
    for label in unique_labels:
        team_indices = np.where(labels == label)[0]
        team_matrix = matrix[np.ix_(team_indices, team_indices)]
        coherences.append(np.mean(team_matrix))
    return np.mean(coherences)


def calculate_similarity_matrix(features):
    distances = pairwise_distances(features)
    similarities = 1 - (distances / distances.max())
    return similarities


def compare_clustering_methods(
    features,
    original_labels,
    similarity_matrix,
    year,
    semester,
    course_name,
    period,
    visual_dir,
):
    methods = ["original", "kmeans", "hierarchical", "dbscan"]
    results = []
    all_team_metrics = {}
    returned_labels = {}

    n_clusters = len(set(original_labels))  # Get number of original clusters

    for method in methods:
        if method == "original":
            clustered_labels = original_labels
        else:
            # Pass n_clusters to all methods
            clustered_labels = apply_clustering(features, n_clusters, method)

        # Skip if clustering failed
        if clustered_labels is None or (
            method == "dbscan" and len(set(clustered_labels)) == 1
        ):
            print(f"{method} clustering failed for {semester} {year} {period}")
            continue

        # Store the labels
        returned_labels[method] = clustered_labels

        unique_labels = len(set(clustered_labels))
        if unique_labels > 1:
            _, clustered_labels_normalized = np.unique(
                clustered_labels, return_inverse=True
            )
            silhouette = silhouette_score(features, clustered_labels_normalized)
        else:
            silhouette = float("nan")

        # Calculate metrics
        metrics = calculate_clustering_metrics(
            features, clustered_labels, similarity_matrix, original_labels
        )

        # Store team metrics with method name
        team_metrics_df = pd.DataFrame.from_dict(
            metrics["team_metrics"], orient="index"
        )
        all_team_metrics[f"{method}_{semester}_{year}_{period}"] = team_metrics_df

        # Store results for comparison table
        results.append(
            {
                "method": method,
                "silhouette_score": silhouette,
                "unique_clusters": unique_labels,
                **metrics["global_metrics"],
            }
        )

        # if method != "original":
        #     create_comparison_plot(
        #         original_labels,
        #         clustered_labels,
        #         similarity_matrix,
        #         features,
        #         year,
        #         semester,
        #         period,
        #         f"{course_name}_{method}",
        #         visual_dir,
        #     )

    # Create comparison table
    comparison_table = pd.DataFrame(results).set_index("method")
    print(f"\nComparison Table for {course_name} - {semester} {year} - {period}:")
    print(comparison_table.round(4).to_string())

    return returned_labels, comparison_table, all_team_metrics


def calculate_clustering_metrics_with_mapping(
    features,
    target_labels,
    clustering_results,
    similarities,
    year,
    semester,
    course_name,
    period,
):
    # Skip if dataset is too small
    if len(features) < 10:
        print(
            f"Skipping {course_name} - {semester} {year} - {period}: dataset too small (n={len(features)})"
        )
        return None, None

    comparison_table = []
    all_team_metrics = {}

    # For each clustering method
    for method, labels in clustering_results.items():
        # For original method, use target_labels directly
        if method == "original":
            mapped_labels = target_labels
        else:
            # Map labels for other clustering methods
            n_samples = len(features)
            mapped_labels = np.full(n_samples, -1)
            min_length = min(len(mapped_labels), len(labels))
            mapped_labels[:min_length] = labels[:min_length]

        metrics = calculate_clustering_metrics(
            features, mapped_labels, similarities, target_labels
        )

        # Calculate silhouette score if we have enough samples and clusters
        n_clusters = len(np.unique(mapped_labels[mapped_labels != -1]))
        n_samples = len(features)

        if n_samples > n_clusters > 1:
            silhouette = silhouette_score(features, mapped_labels)
        else:
            silhouette = float("nan")

        comparison_table.append(
            {
                "method": method,
                "silhouette_score": silhouette,
                "unique_clusters": n_clusters,
                **metrics[
                    "global_metrics"
                ],  # This now includes team_assignment_dissimilarity instead of matrix_distance
            }
        )

        all_team_metrics.update(
            {
                f"{method}_{semester}_{year}_{period}": pd.DataFrame.from_dict(
                    metrics["team_metrics"], orient="index"
                )
            }
        )

    return pd.DataFrame(comparison_table).set_index("method"), all_team_metrics
