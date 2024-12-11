import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import numpy as np
import re

from src.utils.data_processing import get_data_from_csv, process_course_data


def create_comparison_plot(
    original_labels,
    clustered_labels,
    similarity_matrix,
    features,
    year,
    semester,
    course_name,
    period,
    output_dir,
):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(30, 20))
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0], polar=True)
    ax4 = fig.add_subplot(gs[1, 1], polar=True)

    # Plot original teams heatmap
    original_order = np.argsort(original_labels)
    original_reordered_matrix = similarity_matrix[original_order][:, original_order]
    sns.heatmap(
        original_reordered_matrix,
        ax=ax1,
        cmap="YlGnBu",
        xticklabels=original_order + 1,
        yticklabels=original_order + 1,
    )
    ax1.set_title(f"{course_name} {year} {semester} {period} Original Teams")
    ax1.set_xlabel("Student ID")
    ax1.set_ylabel("Student ID")

    # Add borders to original teams
    add_cluster_borders(ax1, original_labels[original_order])

    # Plot clustered solution heatmap
    clustered_order = np.argsort(clustered_labels)
    clustered_reordered_matrix = similarity_matrix[clustered_order][:, clustered_order]
    sns.heatmap(
        clustered_reordered_matrix,
        ax=ax2,
        cmap="YlGnBu",
        xticklabels=clustered_order + 1,
        yticklabels=clustered_order + 1,
    )
    ax2.set_title(f"{course_name} {year} {semester} {period} Clustered Teams")
    ax2.set_xlabel("Student ID")
    ax2.set_ylabel("Student ID")

    # Add borders to clustered teams
    add_cluster_borders(ax2, clustered_labels[clustered_order])

    # Create radar plot for original teams
    create_radar_plot(
        ax3, features, original_labels, course_name, year, semester, "Original"
    )

    # Create radar plot for clustered teams
    create_radar_plot(
        ax4, features, clustered_labels, course_name, year, semester, "Clustered"
    )

    plt.tight_layout()
    # Sanitize filename
    safe_filename = re.sub(
        r'[<>:"/\\|?*]', "_", f"{course_name}_{year}_{semester}_{period}_comparison.png"
    )
    save_path = os.path.join(output_dir, safe_filename)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Print some statistics
    original_coherence = np.mean(original_reordered_matrix)
    clustered_coherence = np.mean(clustered_reordered_matrix)
    print(f"Original grouping coherence: {original_coherence:.4f}")
    print(f"Clustered grouping coherence: {clustered_coherence:.4f}")

    # Additional diagnostics
    print(f"Original matrix shape: {original_reordered_matrix.shape}")
    print(f"Clustered matrix shape: {clustered_reordered_matrix.shape}")
    print(f"Original matrix sum: {np.sum(original_reordered_matrix):.4f}")
    print(f"Clustered matrix sum: {np.sum(clustered_reordered_matrix):.4f}")
    print(
        f"Are matrices identical? {np.allclose(original_reordered_matrix, clustered_reordered_matrix)}"
    )

    # Check diagonal values
    original_diagonal = np.diagonal(original_reordered_matrix)
    clustered_diagonal = np.diagonal(clustered_reordered_matrix)
    print(f"Original diagonal mean: {np.mean(original_diagonal):.4f}")
    print(f"Clustered diagonal mean: {np.mean(clustered_diagonal):.4f}")

    # Print the first few rows of each matrix
    print("First 5 rows of original matrix:")
    print(original_reordered_matrix[:5, :5])
    print("First 5 rows of clustered matrix:")
    print(clustered_reordered_matrix[:5, :5])

    # Calculate coherence for each team
    def team_coherence(labels, matrix):
        unique_labels = np.unique(labels)
        coherences = []
        for label in unique_labels:
            team_indices = np.where(labels == label)[0]
            team_matrix = matrix[np.ix_(team_indices, team_indices)]
            coherences.append(np.mean(team_matrix))
        return np.mean(coherences)

    original_team_coherence = team_coherence(original_labels, similarity_matrix)
    clustered_team_coherence = team_coherence(clustered_labels, similarity_matrix)
    print(f"Original team coherence: {original_team_coherence:.4f}")
    print(f"Clustered team coherence: {clustered_team_coherence:.4f}")


def plot_features_correlation(data_csv_path, output_dir):
    # get dataset from csv folder
    df = get_data_from_csv(data_csv_path)
    # Select only end period features
    # Get only rows where checkpoint is 'end'
    df = df[df["checkpoint"] == "end"]

    end_features, end_target, end_user_ids = process_course_data(df)

    # Define feature groups and their colors
    feature_groups = {
        "Performance": ["avg_grade", "total_points"],
        "Engagement": [
            "submissions_count",
            "total_activities",
            "percentage_content_completed",
        ],
        "Time Management": [
            "avg_assessment_time",
            "avg_time_to_submission",
            "avg_pages_per_day",
            "avg_time_per_page_seconds",
            "avg_points_per_activity",
        ],
    }

    group_colors = {
        "Performance": "#1f77b4",  # Blue
        "Engagement": "#ff7f0e",  # Light Orange
        "Time Management": "#b8860b",  # Dark Yellow
    }

    # Create a list of features in desired order
    ordered_features = []
    for group in ["Performance", "Engagement", "Time Management"]:
        ordered_features.extend(feature_groups[group])

    # Calculate correlation matrix
    correlation_matrix = end_features[ordered_features].corr()

    # Create figure
    plt.figure(figsize=(12, 10))

    # Create mask for upper triangle
    mask = np.triu(np.ones_like(correlation_matrix), k=1)

    # Create color mapping for features
    feature_colors = []
    for feature in ordered_features:
        for group, features in feature_groups.items():
            if feature in features:
                feature_colors.append(group_colors[group])

    # Plot heatmap
    sns.heatmap(
        correlation_matrix,
        mask=mask,
        annot=True,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        fmt=".2f",
        square=True,
    )

    # Customize axes
    plt.xticks(
        np.arange(len(ordered_features)) + 0.5,
        [f.replace("_end", "").replace("_", " ") for f in ordered_features],
        rotation=45,
        ha="right",
    )
    plt.yticks(
        np.arange(len(ordered_features)) + 0.5,
        [f.replace("_end", "").replace("_", " ") for f in ordered_features],
        rotation=0,
    )

    # Color code the axis labels
    ax = plt.gca()
    for i, label in enumerate(ax.get_xticklabels()):
        label.set_color(feature_colors[i])
    for i, label in enumerate(ax.get_yticklabels()):
        label.set_color(feature_colors[i])

    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], color=color, lw=4, label=group)
        for group, color in group_colors.items()
    ]
    plt.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1.3, 0.8))

    plt.title("Feature Correlations (End Period)")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "feature_correlations.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()

    print(
        "Feature correlation plot saved to time_based_analysis/feature_correlations.png"
    )


def plot_teams_by_semester(assignments_path):
    """
    Plot the number of teams per semester, organized chronologically.
    """
    # Load team assignments
    with open(assignments_path, "r") as f:
        team_assignments = json.load(f)

    # Get semesters and organize by year and type
    teams_per_semester = {}

    for sem in team_assignments["original"].keys():
        teams_per_semester[sem] = len(team_assignments["original"][sem])

    # Sort semesters chronologically
    def semester_sort_key(x):
        sem, year = x.split("_")
        # Assign numeric values to semesters to sort Fall -> Summer -> Spring
        sem_order = {"fall": 0, "summer": 1, "spring": 2}
        return (year, sem_order[sem.lower()])

    sorted_semesters = sorted(teams_per_semester.keys(), key=semester_sort_key)

    # Create x-axis points and corresponding y values
    x = range(len(sorted_semesters))
    teams_y = [teams_per_semester[sem] for sem in sorted_semesters]

    # Create the plot
    plt.figure(figsize=(12, 6))

    # Plot single line
    plt.plot(x, teams_y, marker="o", color="skyblue", linewidth=2)

    # Customize the plot
    plt.title("Number of Teams per Semester")
    plt.xlabel("Semester")
    plt.ylabel("Number of Teams")
    plt.xticks(
        x, [sem.replace("_", " ").title() for sem in sorted_semesters], rotation=45
    )
    plt.grid(True, alpha=0.3)

    # Add value labels above points
    for i, v in enumerate(teams_y):
        plt.text(i, v, str(v), ha="center", va="bottom")

    # Save plot
    plt.tight_layout()
    plt.savefig("time_based_analysis/teams_per_semester.png")
    plt.close()

    print("Teams per semester plot saved to time_based_analysis/teams_per_semester.png")


def plot_students_by_semesters(assignments_path):
    """
    Plot the number of students per semester and average team size over time.
    """
    with open(assignments_path, "r") as f:
        team_assignments = json.load(f)

    # Initialize dictionaries to store counts
    students_per_semester = {}
    avg_team_size_per_semester = {}

    # Calculate counts for each semester
    for semester in team_assignments["original"]:
        # Count total students
        total_students = sum(
            len(team) for team in team_assignments["original"][semester].values()
        )
        students_per_semester[semester] = total_students

        # Calculate average team size
        num_teams = len(team_assignments["original"][semester])
        avg_team_size = total_students / num_teams if num_teams > 0 else 0
        avg_team_size_per_semester[semester] = avg_team_size

    # Sort semesters by year and then Fall -> Summer -> Spring
    def semester_sort_key(x):
        sem, year = x.split("_")
        # Assign numeric values to semesters to sort Fall -> Summer -> Spring
        sem_order = {"fall": 0, "summer": 1, "spring": 2}
        return (year, sem_order[sem.lower()])

    sorted_semesters = sorted(students_per_semester.keys(), key=semester_sort_key)

    # Create x-axis points and corresponding y values
    x = range(len(sorted_semesters))
    students_y = [students_per_semester[sem] for sem in sorted_semesters]
    team_size_y = [avg_team_size_per_semester[sem] for sem in sorted_semesters]

    # Create the plot
    plt.figure(figsize=(12, 6))

    # Plot both lines
    plt.plot(
        x, students_y, marker="o", label="Total Students", color="skyblue", linewidth=2
    )
    plt.plot(
        x,
        team_size_y,
        marker="s",
        label="Average Team Size",
        color="lightcoral",
        linewidth=2,
    )

    # Customize the plot
    plt.title("Students per Semester and Average Team Size")
    plt.xlabel("Semester")
    plt.ylabel("Count")
    plt.xticks(
        x, [sem.replace("_", " ").title() for sem in sorted_semesters], rotation=45
    )
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add value labels on the points
    for i, (students, team_size) in enumerate(zip(students_y, team_size_y)):
        plt.text(i, students, f"{int(students)}", ha="center", va="bottom")
        plt.text(i, team_size, f"{team_size:.1f}", ha="center", va="bottom")

    # Save plot
    plt.tight_layout()
    plt.savefig("time_based_analysis/feature_plots/students_per_semester.png")
    plt.close()

    print("Students per semester plot saved to time_based_analysis/feature_plots/")


def add_cluster_borders(ax, labels):
    current_label = labels[0]
    start = 0
    for i, label in enumerate(labels[1:], 1):
        if label != current_label:
            ax.add_patch(
                plt.Rectangle(
                    (start, start),
                    i - start,
                    i - start,
                    fill=False,
                    edgecolor="red",
                    lw=2,
                )
            )
            start = i
            current_label = label
    ax.add_patch(
        plt.Rectangle(
            (start, start),
            len(labels) - start,
            len(labels) - start,
            fill=False,
            edgecolor="red",
            lw=2,
        )
    )


def create_radar_plot(ax, features, labels, course_name, year, semester, team_type):
    unique_labels = np.unique(labels)
    feature_means = np.array(
        [features[labels == label].mean(axis=0) for label in unique_labels]
    )
    feature_stds = np.array(
        [features[labels == label].std(axis=0) for label in unique_labels]
    )
    team_sizes = np.array([np.sum(labels == label) for label in unique_labels])
    feature_names = features.columns

    # Number of variables
    num_vars = len(feature_names)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # complete the circle

    # Plot
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), feature_names)

    for i, (team_means, team_stds, team_size) in enumerate(
        zip(feature_means, feature_stds, team_sizes)
    ):
        values = team_means.tolist()
        values += values[:1]  # complete the polygon
        ax.plot(angles, values, "o-", linewidth=2, label=f"Team {i+1} (n={team_size})")

        # Plot standard deviation
        upper = np.minimum(team_means + team_stds, 1)  # Cap at 1
        lower = np.maximum(team_means - team_stds, 0)  # Cap at 0
        upper = upper.tolist()
        lower = lower.tolist()
        upper += upper[:1]
        lower += lower[:1]
        ax.fill_between(angles, lower, upper, alpha=0.2)

    ax.set_ylim(0, 1)
    ax.set_title(
        f"{course_name} {year} {semester}\n{team_type} Team Feature Comparison"
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

    # Add labels to the radar chart
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], color="grey", size=7)
    plt.ylim(0, 1)


def create_cluster_evolution_plots(
    all_comparison_tables, semester_year_keys, output_dir
):
    """
    Create separate bar plots for each combination of method and metric.
    """
    # Create the combined DataFrame
    combined_df = pd.concat(
        all_comparison_tables,
        keys=semester_year_keys,
        names=["semester", "year", "period", "method"],
    )

    # Define metrics and methods
    metrics = [
        # "silhouette_score",
        "avg_similarity",
        "avg_distance_to_centroid",
        "team_assignment_dissimilarity",
    ]
    methods = [
        "original",
        "kmeans",
        "hierarchical",
        "dbscan",
    ]
    periods = ["beginning", "middle", "end"]

    # Color scheme for each algorithm
    colors = {
        "original": "#d62728",  # red
        "kmeans": "#2ca02c",  # green
        "hierarchical": "#ff7f0e",  # orange
        "dbscan": "#1f77b4",  # blue
    }

    # Create individual plots for each method-metric combination
    for method in methods:
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            ax = plt.gca()

            # Calculate mean values for each period
            means = []
            for period in periods:
                try:
                    val = combined_df.xs((period, method), level=("period", "method"))[
                        metric
                    ].mean()
                    means.append(val)
                except:
                    means.append(np.nan)

            # Create bar plot
            bars = ax.bar(periods, means, color=[colors[method] for _ in periods])

            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                if not np.isnan(height):
                    va = "bottom" if height >= 0 else "top"
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        height,
                        f"{height:.2f}",
                        ha="center",
                        va=va,
                        rotation=0,
                    )

            # Customize plot
            plt.title(
                f'{method.capitalize()}: {metric.replace("_", " ").title()}',
                pad=20,
                fontsize=14,
            )
            plt.xlabel("Period", fontsize=12)
            plt.ylabel(metric.replace("_", " ").title(), fontsize=12)
            ax.grid(True, axis="y", linestyle="--", alpha=0.7)

            # Set y-axis limits based on data
            valid_values = [v for v in means if not np.isnan(v)]
            if valid_values:
                data_range = max(valid_values) - min(valid_values)
                # Use larger padding to stretch the y-axis and show more difference
                padding = data_range * 0.5  # Increased from 0.1 to 0.5
                # Set y_min further from minimum value
                y_min = min(valid_values) - padding
                y_max = max(valid_values) + padding
                # Set y-axis limits with more padding to stretch the view
                ax.set_ylim(y_min, y_max)
                ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5, alpha=0.5)

            # Save individual plot
            plt.tight_layout()
            filename = f"clustering_evolution_{method}_{metric}.png"
            plt.savefig(
                os.path.join(output_dir, filename), bbox_inches="tight", dpi=300
            )
            plt.close()

    # After the individual method plots, add comparison plots for original vs hierarchical
    comparison_methods = [
        # "original",
        "hierarchical",
    ]
    comparison_colors = {
        # "original": colors["original"],
        "hierarchical": colors["hierarchical"],
    }

    for metric in metrics:
        plt.figure(figsize=(12, 7))
        ax = plt.gca()

        # Set width for bars
        width = 0.35
        x = np.arange(len(periods))

        # Plot bars for both methods
        for i, method in enumerate(comparison_methods):
            means = []
            for period in periods:
                try:
                    val = combined_df.xs((period, method), level=("period", "method"))[
                        metric
                    ].mean()
                    means.append(val)
                except:
                    means.append(np.nan)

            # Offset bars for each method
            # offset = width * (i - 0.5)
            bars = ax.bar(
                x,
                means,
                width,
                label=method.capitalize(),
                color=comparison_colors[method],
                alpha=0.8,
            )

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                if not np.isnan(height):
                    va = "bottom" if height >= 0 else "top"
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        height,
                        f"{height:.2f}",
                        ha="center",
                        va=va,
                        fontsize=10,
                    )

        # Customize plot
        plt.title(
            f'Hierarchical: {metric.replace("_", " ").title()}',
            pad=20,
            fontsize=14,
        )
        plt.xlabel("Period", fontsize=12)
        plt.ylabel(metric.replace("_", " ").title(), fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(periods)
        ax.grid(True, axis="y", linestyle="--", alpha=0.7)
        ax.legend()

        # Set y-axis limits
        all_values = [
            v
            for method in comparison_methods
            for period in periods
            for v in [
                combined_df.xs((period, method), level=("period", "method"))[
                    metric
                ].mean()
            ]
            if not np.isnan(v)
        ]

        if all_values:
            y_min = 0
            y_max = max(all_values) * 1.2
            ax.set_ylim(y_min, y_max)
            ax.yaxis.set_major_locator(plt.MultipleLocator(0.05))
            ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5, alpha=0.5)

        # Save comparison plot
        plt.tight_layout()
        filename = f"clustering_comparison_original_vs_hierarchical_{metric}.png"
        plt.savefig(os.path.join(output_dir, filename), bbox_inches="tight", dpi=300)
        plt.close()

