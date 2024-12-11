import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score


def apply_clustering(features, n_clusters, method="kmeans"):
    if method == "dbscan":
        # If n_clusters is None, use a default value based on data size
        target_clusters = (
            n_clusters if n_clusters is not None else max(2, len(features) // 5)
        )
        best_params, best_labels = optimize_dbscan_parameters(features, target_clusters)
        if best_params is not None:
            return best_labels
        else:
            # Return all points as noise if optimization fails
            return np.full(len(features), -1)

    # Rest of the methods remain the same
    if method == "kmeans":
        initial_centers = initialize_balanced_centers(features, n_clusters)
        clusterer = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            init=initial_centers,
            n_init=1,
        )
    elif method == "hierarchical":
        clusterer = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage="ward",
        )
    elif method == "spectral":
        n_samples = features.shape[0]
        n_neighbors = min(max(n_clusters * 2, 2), n_samples - 1)
        clusterer = SpectralClustering(
            n_clusters=n_clusters,
            random_state=42,
            affinity="nearest_neighbors",
            n_neighbors=n_neighbors,
            assign_labels="kmeans",
        )
    else:
        raise ValueError(f"Unknown clustering method: {method}")

    labels = clusterer.fit_predict(features)

    # Apply balancing post-processing for all methods except spectral and dbscan
    if method in ["kmeans", "hierarchical"]:
        labels = balance_clusters(features, labels, n_clusters)

    return labels


def initialize_balanced_centers(features, n_clusters):
    """Initialize cluster centers to encourage balanced clustering."""
    n_samples = len(features)
    target_size = n_samples // n_clusters

    # Use k-means++ for the first center
    centers = [features.iloc[np.random.randint(n_samples)].values]

    # Choose remaining centers to encourage balance
    distances = np.zeros((n_samples, n_clusters))

    for i in range(1, n_clusters):
        # Calculate distances to existing centers
        for j, center in enumerate(centers):
            distances[:, j] = np.linalg.norm(features - center, axis=1)

        # Get minimum distance to any existing center
        min_distances = distances[:, :i].min(axis=1)

        # Weight by distance and avoid selecting points too close to existing clusters
        weights = min_distances * (
            1 / (1 + np.sum(distances[:, :i] < np.median(min_distances), axis=1))
        )

        # Select next center
        next_center_idx = np.argmax(weights)
        centers.append(features.iloc[next_center_idx].values)

    return np.array(centers)


def balance_clusters(features, labels, n_clusters):
    """Post-process clustering to balance cluster sizes."""
    n_samples = len(features)
    target_size = n_samples // n_clusters

    # Calculate current cluster sizes
    unique_labels, counts = np.unique(labels, return_counts=True)
    cluster_sizes = dict(zip(unique_labels, counts))

    # Calculate cluster centers
    centers = np.array([features[labels == i].mean(axis=0) for i in range(n_clusters)])

    # Identify oversized and undersized clusters
    oversized = [l for l, c in cluster_sizes.items() if c > target_size + 1]
    undersized = [l for l, c in cluster_sizes.items() if c < target_size]

    while oversized and undersized:
        from_cluster = oversized[0]
        to_cluster = undersized[0]

        # Find points in oversized cluster
        mask = labels == from_cluster
        points_in_cluster = features[mask]

        # Calculate distances to both cluster centers
        dist_to_current = np.linalg.norm(
            points_in_cluster - centers[from_cluster], axis=1
        )
        dist_to_target = np.linalg.norm(points_in_cluster - centers[to_cluster], axis=1)

        # Find the point that has the smallest ratio of distances
        # (closest to target relative to current)
        ratios = dist_to_target / dist_to_current
        point_idx = np.where(mask)[0][np.argmin(ratios)]

        # Move point to new cluster
        labels[point_idx] = to_cluster

        # Update cluster sizes and lists
        cluster_sizes[from_cluster] -= 1
        cluster_sizes[to_cluster] += 1

        if cluster_sizes[from_cluster] <= target_size + 1:
            oversized.remove(from_cluster)
        if cluster_sizes[to_cluster] >= target_size:
            undersized.remove(to_cluster)

    return labels


def optimize_dbscan_parameters(features, target_n_clusters):
    """
    Optimize DBSCAN parameters aiming for a specific number of clusters and balanced teams.
    """
    best_score = -1
    best_params = None
    best_labels = None
    best_cluster_diff = float("inf")

    # Calculate the average distance to k nearest neighbors to help set eps range
    neigh = NearestNeighbors(n_neighbors=2)
    nbrs = neigh.fit(features)
    distances, _ = nbrs.kneighbors(features)
    avg_dist = np.mean(distances[:, 1])

    # Define parameter ranges with more granularity
    eps_range = np.concatenate(
        [
            np.linspace(avg_dist * 0.1, avg_dist, 15),  # More points in lower range
            np.linspace(avg_dist, avg_dist * 2, 10),  # Fewer points in upper range
        ]
    )

    # Adjust min_samples range based on data size and target clusters
    n_samples = len(features)
    min_expected_cluster_size = max(2, n_samples // (target_n_clusters * 2))
    min_samples_range = range(2, min(min_expected_cluster_size + 1, n_samples // 2))

    print(f"Optimizing DBSCAN parameters (target clusters: {target_n_clusters})")
    print(f"Number of samples: {n_samples}")
    print(f"Average distance: {avg_dist:.3f}")

    for eps in eps_range:
        for min_samples in min_samples_range:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(features)

            # Count actual clusters (excluding noise)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

            # Skip invalid configurations
            if n_clusters < 2:
                continue

            noise_ratio = np.sum(labels == -1) / len(labels)
            if noise_ratio > 0.2:  # Allow up to 20% noise points
                continue

            # Calculate cluster sizes
            unique_labels = set(labels[labels != -1])
            cluster_sizes = [np.sum(labels == label) for label in unique_labels]

            # Skip if any cluster is too small
            min_cluster_size = max(2, n_samples // (target_n_clusters * 3))
            if min(cluster_sizes) < min_cluster_size:
                continue

            # Calculate cluster size variance (lower is better)
            size_variance = np.var(cluster_sizes)

            # Calculate silhouette score, ignoring noise points
            non_noise_mask = labels != -1
            if np.sum(non_noise_mask) > 1:
                try:
                    score = silhouette_score(
                        features[non_noise_mask], labels[non_noise_mask]
                    )

                    # Calculate how far we are from target number of clusters
                    cluster_diff = abs(n_clusters - target_n_clusters)

                    # Modified scoring to better balance different aspects
                    cluster_penalty = 1 / (1 + cluster_diff)
                    size_penalty = 1 / (1 + size_variance)
                    noise_penalty = 1 - noise_ratio

                    combined_score = (
                        score * cluster_penalty * size_penalty * noise_penalty
                    )

                    if combined_score > best_score:
                        best_score = combined_score
                        best_params = {"eps": eps, "min_samples": min_samples}
                        best_labels = labels
                        best_cluster_diff = cluster_diff

                        print(
                            f"eps={eps:.3f}, min_samples={min_samples}: "
                            f"score={score:.3f}, clusters={n_clusters}, "
                            f"variance={size_variance:.3f}, noise={noise_ratio:.2%}"
                        )
                except:
                    continue

    if best_params is None:
        print("DBSCAN optimization failed to find valid parameters")
        # Try one more time with more relaxed constraints
        return optimize_dbscan_parameters_relaxed(features, target_n_clusters)

    print(f"\nBest parameters: {best_params}")
    print(f"Best combined score: {best_score:.3f}")

    # Analyze final clustering
    if best_labels is not None:
        noise_points = np.sum(best_labels == -1)
        print(
            f"Number of noise points: {noise_points} "
            f"({noise_points/len(best_labels)*100:.1f}%)"
        )

        unique_labels = set(best_labels[best_labels != -1])
        cluster_sizes = [np.sum(best_labels == label) for label in unique_labels]
        print("\nCluster sizes:", cluster_sizes)
        print(f"Size variance: {np.var(cluster_sizes):.3f}")

    return best_params, best_labels


def optimize_dbscan_parameters_relaxed(features, target_n_clusters):
    """
    Fallback optimization with relaxed constraints for difficult cases.
    """
    print("\nTrying relaxed constraints...")

    neigh = NearestNeighbors(n_neighbors=2)
    nbrs = neigh.fit(features)
    distances, _ = nbrs.kneighbors(features)
    avg_dist = np.mean(distances[:, 1])

    # Try a wider range of parameters
    eps_range = np.linspace(avg_dist * 0.05, avg_dist * 3, 30)
    min_samples_range = range(2, max(3, len(features) // 3))

    best_labels = None
    best_n_clusters = 0
    best_eps = None
    best_min_samples = None

    for eps in eps_range:
        for min_samples in min_samples_range:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(features)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters >= 2:  # Accept any valid clustering
                noise_ratio = np.sum(labels == -1) / len(labels)
                if noise_ratio <= 0.3:  # Allow up to 30% noise
                    best_labels = labels
                    best_n_clusters = n_clusters
                    best_eps = eps
                    best_min_samples = min_samples
                    break
        if best_labels is not None:
            break

    if best_labels is not None:
        return {"eps": best_eps, "min_samples": best_min_samples}, best_labels

    return None, None
