import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


def create_feature_evolution_plots(
    df, results, feature_plots_dir=None, period="end", all_period_data=None
):
    """
    Create evolution plots for each feature, comparing all algorithms together.
    """

    if feature_plots_dir is None:
        feature_plots_dir = "time_based_analysis/feature_plots"

    os.makedirs(feature_plots_dir, exist_ok=True)

    methods = ["original", "kmeans", "hierarchical", "dbscan"]
    colors = {
        "original": "#d62728",
        "kmeans": "#2ca02c",
        "hierarchical": "#ff7f0e",
        "dbscan": "#1f77b4",
    }

    # Get features
    exclude_columns = [
        "user_id",
        "course_id",
        "course_name",
        "team_id",
        "team_name",
        "semester",
        "year",
        "analysis_dates",
        "checkpoint",
        "semester_year",
    ]
    features = [col for col in df.columns if col not in exclude_columns]

    print(f"\nFeatures shape: {df.shape}")
    for period in results:  # First level: periods
        print(f"\nPeriod: {period}")
        for method in results[period]:  # Second level: methods
            labels = results[period][method]
            print(f"{method} labels shape: {len(labels)}")

    # Create plot for each feature
    for feature in features:
        plt.figure(figsize=(12, 7))
        ax = plt.gca()
        width = 0.2
        x = [0]
        means = []
        errors = []
        labels = []

        for method in methods:
            if method not in results:
                continue

            # Get cluster labels and ensure they match features length
            cluster_labels = results[method]
            if len(cluster_labels) != len(df):
                print(
                    f"Warning: {method} labels length ({len(cluster_labels)}) doesn't match features length ({len(df)})"
                )
                continue

            # Calculate statistics for each cluster
            cluster_means = []
            cluster_sizes = []

            for cluster in np.unique(cluster_labels):
                if cluster != -1:  # Exclude noise points
                    cluster_mask = cluster_labels == cluster
                    cluster_data = df.loc[cluster_mask, feature]
                    if len(cluster_data) > 0:
                        cluster_means.append(cluster_data.mean())
                        cluster_sizes.append(len(cluster_data))

            if cluster_means:
                method_mean = np.average(cluster_means, weights=cluster_sizes)
                weighted_var = np.average(
                    (np.array(cluster_means) - method_mean) ** 2, weights=cluster_sizes
                )
                method_error = np.sqrt(weighted_var / len(cluster_means))

                means.append(method_mean)
                errors.append(method_error)
                labels.append(method)

        # Plot bars
        for i, (method, mean_val, err) in enumerate(zip(labels, means, errors)):
            offset = width * (i - len(labels) / 2 + 0.5)
            bars = ax.bar(
                x[0] + offset,
                mean_val,
                width,
                label=method.capitalize(),
                color=colors[method],
                alpha=0.8,
            )

            ax.errorbar(
                x[0] + offset,
                mean_val,
                yerr=err,
                fmt="none",
                color="black",
                capsize=5,
                alpha=0.5,
            )

            ax.text(
                x[0] + offset,
                mean_val,
                f"{mean_val:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=45,
            )

        # Customize plot
        plt.title(
            f'Feature Comparison: {feature.replace("_", " ").title()}\n{period.capitalize()}',
            pad=20,
            fontsize=14,
        )
        plt.xlabel("Clustering Methods", fontsize=12)
        plt.ylabel(feature.replace("_", " ").title(), fontsize=12)
        ax.set_xticks([])
        ax.grid(True, axis="y", linestyle="--", alpha=0.7)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        if means:
            data_range = max(means) - min(means)
            padding = data_range * 0.2
            y_min = min(means) - padding - max(errors)
            y_max = max(means) + padding + max(errors)
            ax.set_ylim(y_min, y_max)
            ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5, alpha=0.5)

        plt.tight_layout()
        filename = f"feature_comparison_{feature}_{period}.png"
        plt.savefig(
            os.path.join(feature_plots_dir, filename), bbox_inches="tight", dpi=300
        )
        plt.close()


def create_feature_comparison_barcharts(
    features, returned_labels, period, output_dir, all_period_data=None
):
    """
    Create bar charts comparing feature distributions across different clustering methods and periods,
    averaged across all semesters.
    """
    os.makedirs(output_dir, exist_ok=True)

    colors = {
        "dbscan": "#1f77b4",  # blue
        "hierarchical": "#ff7f0e",  # orange
        "kmeans": "#2ca02c",  # green
        "original": "#d62728",  # red
    }

    method_order = ["dbscan", "hierarchical", "kmeans", "original"]
    periods = ["beginning", "middle", "end"]

    # Create a plot for each feature
    for feature_name in features.columns:
        plt.figure(figsize=(15, 6))

        n_methods = len(method_order)
        bar_width = 0.25
        indices = np.arange(n_methods)

        # Store data for plotting
        plot_data = {period: {"means": [], "stds": []} for period in periods}

        # Calculate averages across all semesters for each period and method
        for curr_period in periods:
            period_data_list = all_period_data[curr_period]
            if not period_data_list:
                continue

            # Initialize lists to store values for each method
            method_values = {method: [] for method in method_order}

            # Collect values across all semesters
            for curr_features, curr_labels in period_data_list:
                for method in method_order:
                    if method not in curr_labels:
                        continue

                    labels = curr_labels[method]
                    valid_mask = (
                        labels != -1
                        if method != "original"
                        else np.ones_like(labels, dtype=bool)
                    )

                    # Calculate mean value for this method in this semester using original values
                    if np.any(valid_mask):  # Only calculate if we have valid data
                        method_mean = curr_features[valid_mask][feature_name].mean()
                        method_values[method].append(method_mean)

            # Calculate average and std across semesters for each method
            for method in method_order:
                values = method_values[method]
                if values:
                    plot_data[curr_period]["means"].append(np.mean(values))
                    plot_data[curr_period]["stds"].append(np.std(values))
                else:
                    plot_data[curr_period]["means"].append(np.nan)
                    plot_data[curr_period]["stds"].append(np.nan)

        # Add debug prints
        print(f"\nFeature: {feature_name}")
        for curr_period in periods:
            print(f"\nPeriod: {curr_period}")
            print("Means:", plot_data[curr_period]["means"])
            print("Stds:", plot_data[curr_period]["stds"])

        # Plot bars for each period
        for p_idx, curr_period in enumerate(periods):
            period_positions = indices + (p_idx - 1) * bar_width
            means = plot_data[curr_period]["means"]
            stds = plot_data[curr_period]["stds"]

            bars = plt.bar(
                period_positions,
                means,
                bar_width,
                yerr=stds,
                label=curr_period.capitalize(),
                alpha=0.7,
                capsize=3,
            )

            # Add value labels
            for bar, value in zip(bars, means):
                if not np.isnan(value):
                    height = bar.get_height()
                    plt.text(
                        bar.get_x() + bar.get_width() / 2,
                        height,
                        f"{value:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

        plt.title(f"{feature_name} Average Across All Semesters", pad=20)
        plt.xlabel("Clustering Method")
        plt.ylabel(f"{feature_name} Value")

        plt.xticks(
            indices, [method.capitalize() for method in method_order], rotation=45
        )
        plt.grid(True, axis="y", linestyle="--", alpha=0.3)

        plt.legend(title="Period")
        plt.tight_layout()

        plt.savefig(
            os.path.join(output_dir, f"feature_evolution_{feature_name}.png"),
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()


def plot_feature_averages(feature_averages_path, output_dir):
    """
    Plot the feature averages across all semesters, grouped by period,
    comparing original and clustered assignments.
    """
    with open(feature_averages_path, "r") as f:
        feature_averages = json.load(f)

    # Extract features from first available semester and team
    first_semester = next(iter(feature_averages["original"]))
    first_team = next(iter(feature_averages["original"][first_semester]))
    features = list(feature_averages["original"][first_semester][first_team].keys())

    # Create plots for each feature
    for feature in features:
        plt.figure(figsize=(15, 8))

        # Collect all feature values for original and clustered
        original_values = []
        clustered_values = []

        for assignment_type in ["original", "clustered"]:
            for semester, semester_data in feature_averages[assignment_type].items():
                for team_id, team_data in semester_data.items():
                    if assignment_type == "original":
                        original_values.append(team_data[feature])
                    else:
                        clustered_values.append(team_data[feature])

        # Sort the values
        original_values.sort()
        clustered_values.sort()

        # Create x-axis points (percentiles)
        original_x = np.linspace(0, 100, len(original_values))
        clustered_x = np.linspace(0, 100, len(clustered_values))

        # Plot sorted averages
        plt.plot(
            original_x,
            original_values,
            label="Original Teams",
            color="lightcoral",
            linewidth=2,
        )
        plt.plot(
            clustered_x,
            clustered_values,
            label="Clustered Teams",
            color="skyblue",
            linewidth=2,
        )

        plt.title(f'Distribution of {feature.replace("_", " ").title()} (Sorted)')
        plt.xlabel("Percentile")
        plt.ylabel(feature.replace("_", " ").title())
        plt.legend()
        plt.grid(True, alpha=0.3)

        # # Add statistics annotations
        # plt.axhline(
        #     y=np.mean(original_values), color="skyblue", linestyle="--", alpha=0.5
        # )
        # plt.axhline(
        #     y=np.mean(clustered_values), color="lightcoral", linestyle="--", alpha=0.5
        # )

        # Add text box with statistics
        stats_text = (
            f"Original Teams Mean: {np.mean(original_values):.1f}\n"
            f"Clustered Teams Mean: {np.mean(clustered_values):.1f}\n"
            f"Original Teams Std: {np.std(original_values):.1f}\n"
            f"Clustered Teams Std: {np.std(clustered_values):.1f}"
        )
        plt.text(
            0.02,
            0.98,
            stats_text,
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        # Save plot
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"{feature}_distribution_sorted.png"),
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()

    print(f"Feature plots saved to {output_dir}")


def generate_line_graph(df, output_dir):
    """Generate line graphs showing:
    1. Number of teams and students over time for CEN4072
    2. Average number of students per team over time for CEN4072
    """
    # Create output directory if it doesn't exist
    line_dir = os.path.join(output_dir, "line_graphs")
    os.makedirs(line_dir, exist_ok=True)

    # Filter for CEN4072 and sort by semester and year
    df = df[df["course_name"] == "CEN4072"]
    df["year_sem"] = df["year"].astype(str) + " " + df["semester"]

    # Get number of teams per semester
    teams_by_term = (
        df[(df["checkpoint"] == "end")]
        .groupby("year_sem")["team_id"]
        .nunique()
        .reset_index()
    )

    # Get number of students per semester
    students_by_term = (
        df[(df["checkpoint"] == "end")]
        .groupby("year_sem")["user_id"]
        .nunique()
        .reset_index()
    )

    # Add diagnostic printing
    print("\nDiagnostic Information:")
    print("\nTeams by term:")
    print(teams_by_term)
    print("\nStudents by term:")
    print(students_by_term)

    # Calculate students per team with error checking
    students_per_team = []
    for term in teams_by_term["year_sem"]:
        n_teams = teams_by_term[teams_by_term["year_sem"] == term]["team_id"].iloc[0]
        n_students = students_by_term[students_by_term["year_sem"] == term][
            "user_id"
        ].iloc[0]

        if n_teams == 0:
            print(f"Warning: Zero teams found for {term}")
            students_per_team.append(None)
        else:
            ratio = n_students / n_teams
            students_per_team.append(ratio)
            print(f"{term}: {n_students} students / {n_teams} teams = {ratio:.2f}")

    # Second plot: Students per team over time
    fig2, ax2 = plt.subplots(figsize=(12, 6))

    # Plot only valid data points
    valid_indices = [i for i, x in enumerate(students_per_team) if x is not None]
    valid_ratios = [students_per_team[i] for i in valid_indices]
    valid_terms = [teams_by_term["year_sem"].iloc[i] for i in valid_indices]

    ax2.plot(
        range(len(valid_ratios)),
        valid_ratios,
        marker="o",
        linewidth=2,
        markersize=8,
        color="purple",
    )

    ax2.set_title("CEN4072: Average Students per Team Over Time", fontsize=14, pad=20)
    ax2.set_xlabel("Semester", fontsize=12)
    ax2.set_ylabel("Students per Team", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(len(valid_ratios)))
    ax2.set_xticklabels(valid_terms, rotation=45, ha="right")

    # Add min/max/mean lines
    mean_ratio = np.mean(valid_ratios)
    ax2.axhline(
        y=mean_ratio,
        color="gray",
        linestyle="--",
        alpha=0.5,
        label=f"Mean: {mean_ratio:.2f}",
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(line_dir, "cen4072_students_per_team.png"), bbox_inches="tight"
    )
    plt.close()

    return line_dir
