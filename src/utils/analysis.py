from datetime import datetime
import json
import os

from src.utils.data_processing import get_data_from_csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


def analyze_team_assignments(assignments_path, data_csv_path):
    """
    Analyze team assignments using actual feature values, organized by period.
    """
    with open(assignments_path, "r") as f:
        team_assignments = json.load(f)

    # get original data csv
    original_df = get_data_from_csv(data_csv_path)

    # Filter for CEN4072 data
    cen4072_data = original_df[original_df["course_name"] == "CEN4072"]

    # Create dictionary to store features by semester and period
    features_by_semester = {}

    # Process each semester's data
    for (semester, year, checkpoint), semester_data in cen4072_data.groupby(
        ["semester", "year", "checkpoint"]
    ):
        semester_key = f"{checkpoint}_{semester}_{year}"

        # Get numeric columns for features
        feature_columns = [
            "submissions_count",
            "avg_grade",
            "avg_assessment_time",
            "avg_time_to_submission",
            "total_activities",
            "total_points",
            "unique_pages_viewed",
            "percentage_content_completed",
            "avg_pages_per_day",
            "avg_time_per_page_seconds",
        ]

        # Create features DataFrame indexed by user_id
        features_df = semester_data[feature_columns].copy()
        features_df.index = semester_data["user_id"]

        # Store in dictionary
        features_by_semester[semester_key] = features_df

    # Replace the input features_dict with actual data
    features_dict = features_by_semester
    results = {"original": {}, "clustered": {}}

    # Process both original and clustered assignments
    for assignment_type in ["original", "clustered"]:
        assignments = team_assignments.get(assignment_type, {})

        for semester, teams in assignments.items():
            # Check for each period
            for period in ["beginning", "middle", "end"]:
                period_semester_key = f"{period}_{semester}"
                semester_features = features_dict.get(period_semester_key)

                if semester_features is None:
                    continue

                semester_averages = {}
                print(f"\nProcessing {assignment_type} teams for {period_semester_key}")
                print(f"Number of teams: {len(teams)}")
                if isinstance(semester_features, pd.DataFrame):
                    print(f"Features shape: {semester_features.shape}")
                    print(f"Features index: {semester_features.index.tolist()}")

                for team_id, user_ids in teams.items():
                    matching_features = semester_features.loc[
                        semester_features.index.isin(user_ids)
                    ]

                    if not matching_features.empty:
                        print(f"\nTeam {team_id}")
                        print(f"User IDs: {user_ids}")
                        print(f"Found {len(matching_features)} matching users")

                        # Calculate team averages using actual values
                        team_averages = matching_features.mean().to_dict()
                        semester_averages[team_id] = team_averages
                    else:
                        print(f"\nTeam {team_id}")
                        print(f"User IDs: {user_ids}")
                        print("No matching users found")

                if semester_averages:
                    results[assignment_type][period_semester_key] = semester_averages

    # Save results to JSON
    output_path = os.path.join(
        os.path.dirname(assignments_path), "team_feature_averages.json"
    )
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\nTeam feature averages have been saved to:", output_path)

    return results


def analyze_period_dissimilarity(combined_df, output_dir):
    """
    Analyze and visualize period dissimilarity with a 4x4 matrix.
    """
    import seaborn as sns

    try:
        print("\nProcessing 4x4 dissimilarity matrix...")
        dissimilarity_matrix, mean_dissimilarity, comparison_counts = (
            calculate_period_dissimilarity(combined_df, "hierarchical")
        )

        # Create visualization
        plt.figure(figsize=(15, 6))

        # Plot dissimilarity matrix
        plt.subplot(1, 2, 1)
        sns.heatmap(
            dissimilarity_matrix,
            annot=True,
            cmap="YlOrRd",
            fmt=".3f",
            square=True,
            cbar_kws={"label": "Dissimilarity"},
        )
        plt.title("Period and Original Team Dissimilarity Matrix")

        # Plot comparison counts
        plt.subplot(1, 2, 2)
        sns.heatmap(
            comparison_counts,
            annot=True,
            cmap="Blues",
            fmt="d",
            square=True,
            cbar_kws={"label": "Number of Comparisons"},
        )
        plt.title("Number of Valid Comparisons")

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "period_dissimilarity.png"),
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()

        return dissimilarity_matrix, mean_dissimilarity, comparison_counts

    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        import traceback

        traceback.print_exc()
        return None, None, None


def calculate_period_dissimilarity(combined_df, method):
    """
    Calculate dissimilarity between original teams and hierarchical clustering.
    Since we're using mapped labels, period-to-period dissimilarity should be 0.
    """
    states = ["original", "beginning", "middle", "end"]
    n_states = len(states)

    dissimilarity_matrix = pd.DataFrame(0.0, index=states, columns=states)
    comparison_counts = pd.DataFrame(0, index=states, columns=states)

    # Group by semester and year
    for (semester, year), cohort_data in combined_df.groupby(["semester", "year"]):
        print(f"\nProcessing {semester} {year}")

        # Only calculate dissimilarity between original and each period
        for period in ["beginning", "middle", "end"]:
            try:
                # Get data for original and hierarchical
                orig_data = cohort_data.xs(
                    (period, "original"), level=("period", "method")
                )
                hier_data = cohort_data.xs((period, method), level=("period", "method"))

                # Calculate dissimilarity
                dissim = hier_data["team_assignment_dissimilarity"].iloc[0]

                # Update matrix
                dissimilarity_matrix.loc["original", period] += dissim
                dissimilarity_matrix.loc[period, "original"] += dissim

                # Update comparison counts
                comparison_counts.loc["original", period] += 1
                comparison_counts.loc[period, "original"] += 1

                print(f"Dissimilarity between original and {period}: {dissim:.3f}")

            except KeyError as e:
                print(f"Error processing {period}: {str(e)}")
                continue
            except Exception as e:
                print(f"Error processing {period}: {str(e)}")
                continue

    # Calculate averages based on valid comparisons
    for period in ["beginning", "middle", "end"]:
        count = comparison_counts.loc["original", period]
        if count > 0:
            dissimilarity_matrix.loc["original", period] /= count
            dissimilarity_matrix.loc[period, "original"] /= count

    # Set period-to-period dissimilarity to exactly 0
    for i, period1 in enumerate(["beginning", "middle", "end"]):
        for j, period2 in enumerate(["beginning", "middle", "end"]):
            dissimilarity_matrix.loc[period1, period2] = 0.0

    print("\nDissimilarity Matrix:")
    print(dissimilarity_matrix.round(3))

    print("\nComparison Counts:")
    print(comparison_counts)

    return (
        dissimilarity_matrix,
        np.mean(dissimilarity_matrix.loc["original", ["beginning", "middle", "end"]]),
        comparison_counts,
    )


def compare_team_grades(assignments_path, grades_file_path, output_dir):
    """
    Compare the grade distributions between original and clustered teams,
    focusing on team averages and standard deviations.

    Args:
        assignments_path (str): Path to the team assignments JSON file
        grades_file_path (str): Path to the Excel file containing final grades
        output_dir (str): Path to the directory where the results will be saved
    """
    # Load assignments
    with open(assignments_path, "r") as f:
        team_assignments = json.load(f)

    # Load grades
    grades_df = pd.read_excel(grades_file_path)
    grades_df["user_id"] = grades_df["user_id"].astype(str)

    # Initialize results storage
    results = {
        "original": {"semester_stats": {}, "team_averages": [], "team_stds": []},
        "clustered": {"semester_stats": {}, "team_averages": [], "team_stds": []},
    }

    # Process each assignment type
    for assignment_type in ["original", "clustered"]:
        assignments = team_assignments[assignment_type]

        # Process each semester
        for semester, teams in assignments.items():
            semester_team_stats = []

            # Process each team
            for team_id, user_ids in teams.items():
                # Convert user_ids to strings for matching
                user_ids = [str(uid) for uid in user_ids]

                # Get grades for team members
                team_grades = grades_df[grades_df["user_id"].isin(user_ids)][
                    "Total"
                ].tolist()

                if team_grades:
                    # Calculate team statistics
                    team_avg = np.mean(team_grades)
                    team_std = np.std(team_grades) if len(team_grades) > 1 else 0

                    team_stat = {
                        "semester": semester,
                        "team_id": team_id,
                        "size": len(team_grades),
                        "average": team_avg,
                        "std_dev": team_std,
                        "min": min(team_grades),
                        "max": max(team_grades),
                    }
                    semester_team_stats.append(team_stat)
                    results[assignment_type]["team_averages"].append(team_avg)
                    results[assignment_type]["team_stds"].append(team_std)

            # Store semester statistics
            if semester_team_stats:
                results[assignment_type]["semester_stats"][semester] = {
                    "teams": semester_team_stats,
                    "avg_team_average": np.mean(
                        [stat["average"] for stat in semester_team_stats]
                    ),
                    "avg_team_std": np.mean(
                        [stat["std_dev"] for stat in semester_team_stats]
                    ),
                    "total_teams": len(semester_team_stats),
                }

    # Replace the plotting section with:
    plt.figure(figsize=(15, 8))

    # Get all team averages and sort them
    original_grades = sorted(results["original"]["team_averages"])
    clustered_grades = sorted(results["clustered"]["team_averages"])

    # Create x-axis points (percentiles)
    original_x = np.linspace(0, 100, len(original_grades))
    clustered_x = np.linspace(0, 100, len(clustered_grades))

    # Plot sorted averages
    plt.plot(
        original_x,
        original_grades,
        label="Original Teams",
        color="lightcoral",
        linewidth=2,
    )
    plt.plot(
        clustered_x,
        clustered_grades,
        label="Clustered Teams",
        color="skyblue",
        linewidth=2,
    )

    plt.title("Distribution of Team Grade Averages (Sorted)")
    plt.xlabel("Percentile")
    plt.ylabel("Team Average Grade")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # # Add statistics annotations
    # plt.axhline(y=np.mean(original_grades), color="lightcoral", linestyle="--", alpha=0.5)
    # plt.axhline(
    #     y=np.mean(clustered_grades), color="skyblue", linestyle="--", alpha=0.5
    # )

    # Add text box with statistics
    stats_text = (
        f"Original Teams Mean: {np.mean(original_grades):.1f}\n"
        f"Clustered Teams Mean: {np.mean(clustered_grades):.1f}\n"
        f"Original Teams Std: {np.std(original_grades):.1f}\n"
        f"Clustered Teams Std: {np.std(clustered_grades):.1f}"
    )
    plt.text(
        0.02,
        0.98,
        stats_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "team_grade_distribution_sorted.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()

    # Print statistical comparison
    print("\nTeam Grade Average Comparison:")
    print("\nOriginal Teams:")
    _print_team_stats(results["original"])
    print("\nClustered Teams:")
    _print_team_stats(results["clustered"])

    return results


def _print_team_stats(assignment_results):
    """Helper function to print team statistics."""
    team_avgs = assignment_results["team_averages"]
    team_stds = assignment_results["team_stds"]
    print(f"Number of teams: {len(team_avgs)}")
    print(f"Overall average team grade: {np.mean(team_avgs):.1f}")
    print(f"Average within-team standard deviation: {np.mean(team_stds):.1f}")
    print(f"Standard deviation of team averages: {np.std(team_avgs):.1f}")
    print(f"Range of team averages: {min(team_avgs):.1f} - {max(team_avgs):.1f}")

    print("\nSemester breakdown:")
    for semester, stats in assignment_results["semester_stats"].items():
        print(f"\n{semester}:")
        print(f"  Teams: {stats['total_teams']}")
        print(f"  Average team grade: {stats['avg_team_average']:.1f}")
        print(f"  Average within-team std dev: {stats['avg_team_std']:.1f}")


def analyze_clustering_performance(
    all_comparison_tables, semester_year_keys, output_dir
):
    """Analyze and visualize the performance of different clustering algorithms."""

    # Create a DataFrame with MultiIndex for better analysis
    combined_df = pd.concat(all_comparison_tables, keys=semester_year_keys)
    combined_df.index.names = ["semester", "year", "period", "method"]

    # Calculate average performance metrics across all periods
    avg_performance = combined_df.groupby("method").mean()

    # Create visualizations
    plt.figure(figsize=(15, 10))

    # 1. Silhouette Score Comparison
    plt.subplot(2, 2, 1)
    avg_performance["silhouette_score"].plot(kind="bar")
    plt.title("Average Silhouette Score by Method")
    plt.ylabel("Silhouette Score")
    plt.xticks(rotation=45)

    # 2. Average Similarity Comparison
    plt.subplot(2, 2, 2)
    avg_performance["avg_similarity"].plot(kind="bar")
    plt.title("Average Similarity by Method")
    plt.ylabel("Average Similarity")
    plt.xticks(rotation=45)

    # 3. Average Distance to Centroid
    plt.subplot(2, 2, 3)
    avg_performance["avg_distance_to_centroid"].plot(kind="bar")
    plt.title("Average Distance to Centroid by Method")
    plt.ylabel("Average Distance")
    plt.xticks(rotation=45)

    # 4. Matrix Distance
    plt.subplot(2, 2, 4)
    avg_performance["team_assignment_dissimilarity"].plot(kind="bar")
    plt.title("Matrix Distance by Method")
    plt.ylabel("Distance")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "clustering_performance_comparison.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()

    # Create period-wise analysis
    period_performance = combined_df.groupby(["period", "method"]).mean()

    # Visualize performance trends across periods
    metrics = ["silhouette_score", "avg_similarity", "avg_distance_to_centroid"]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(15, 5 * len(metrics)))

    for i, metric in enumerate(metrics):
        period_pivot = period_performance[metric].unstack()
        period_pivot.plot(kind="bar", ax=axes[i])
        axes[i].set_title(f"{metric} Across Periods")
        axes[i].set_ylabel(metric)
        axes[i].legend(title="Method")
        axes[i].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "clustering_performance_by_period.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()

    # Generate statistical summary
    summary_stats = {
        "Overall Performance": avg_performance,
        "Period-wise Performance": period_performance,
    }

    # Save detailed analysis to Excel
    with pd.ExcelWriter(os.path.join(output_dir, "clustering_analysis.xlsx")) as writer:
        avg_performance.to_excel(writer, sheet_name="Overall Performance")
        period_performance.to_excel(writer, sheet_name="Period Performance")

        # Add statistical tests
        methods = combined_df.index.get_level_values("method").unique()
        stats_df = pd.DataFrame(index=methods, columns=methods)

        for m1 in methods:
            for m2 in methods:
                if m1 != m2:
                    # Perform statistical test (e.g., t-test) between methods
                    t_stat, p_value = stats.ttest_ind(
                        combined_df.xs(m1, level="method")["silhouette_score"].dropna(),
                        combined_df.xs(m2, level="method")["silhouette_score"].dropna(),
                    )
                    stats_df.loc[m1, m2] = p_value

        stats_df.to_excel(writer, sheet_name="Statistical Tests")

    # Generate summary text
    best_method = avg_performance["silhouette_score"].idxmax()
    best_similarity = avg_performance["avg_similarity"].idxmax()
    best_distance = avg_performance["avg_distance_to_centroid"].idxmin()

    summary_text = f"""
    Clustering Analysis Summary:
    ---------------------------
    1. Best Overall Method (by Silhouette Score): {best_method}
    2. Best Method for Similarity: {best_similarity}
    3. Best Method for Cluster Cohesion: {best_distance}
    
    Key Findings:
    - Average Silhouette Score: {avg_performance.loc[best_method, 'silhouette_score']:.3f}
    - Average Similarity: {avg_performance.loc[best_similarity, 'avg_similarity']:.3f}
    - Average Distance to Centroid: {avg_performance.loc[best_distance, 'avg_distance_to_centroid']:.3f}
    
    Period-wise Analysis:
    - Beginning: {period_performance.xs('beginning')['silhouette_score'].idxmax()} performs best
    - Middle: {period_performance.xs('middle')['silhouette_score'].idxmax()} performs best
    - End: {period_performance.xs('end')['silhouette_score'].idxmax()} performs best
    """

    with open(os.path.join(output_dir, "clustering_summary.txt"), "w") as f:
        f.write(summary_text)

    return summary_stats, summary_text


def save_and_analyze_results(
    all_comparison_tables,
    all_team_metrics_dict,
    semester_year_keys,
    output_dir,
    visual_dir,
):
    """Save and analyze clustering results."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Create combined DataFrame
    combined_df = pd.concat(
        all_comparison_tables,
        keys=semester_year_keys,
        names=["semester", "year", "period", "method"],
    )

    # Save to Excel
    with pd.ExcelWriter(
        os.path.join(output_dir, f"clustering_results_{timestamp}.xlsx")
    ) as writer:
        combined_df.to_excel(writer, sheet_name="Comparison")

        for key, df in all_team_metrics_dict.items():
            sheet_name = f"Team_Metrics_{key}"[:31]  # Excel has 31 char limit
            df.to_excel(writer, sheet_name=sheet_name)
    from src.visualization.plots import create_cluster_evolution_plots

    # Create evolution plots
    create_cluster_evolution_plots(
        all_comparison_tables, semester_year_keys, visual_dir
    )

    # Analyze results
    summary = {
        "Best Method by Period": {},
        "Overall Best Method": None,
        "Trend Analysis": {},
    }

    # Group by period and find best method for each
    period_groups = combined_df.groupby("period")

    for period, period_data in period_groups:
        # Find method with highest silhouette score
        best_method = period_data["silhouette_score"].idxmax()[
            -1
        ]  # Get just the method name
        best_score = period_data.loc[
            period_data.index.get_level_values("method") == best_method,
            "silhouette_score",
        ].iloc[0]

        summary["Best Method by Period"][period] = {
            "method": best_method,
            "silhouette_score": best_score,
        }

    # Find overall best method
    method_means = combined_df.groupby("method")["silhouette_score"].mean()
    best_overall_method = method_means.idxmax()
    summary["Overall Best Method"] = {
        "method": best_overall_method,
        "average_silhouette": method_means[best_overall_method],
    }

    # Analyze trends
    metrics = ["silhouette_score", "avg_similarity", "avg_distance_to_centroid"]
    method_trends = {}

    for method in combined_df.index.get_level_values("method").unique():
        method_data = combined_df.xs(method, level="method")
        method_trends[method] = pd.DataFrame(
            {metric: method_data[metric].values for metric in metrics}
        )

        summary["Trend Analysis"][method] = {
            metric: (
                "Improving"
                if method_trends[method][metric].is_monotonic_increasing
                else (
                    "Declining"
                    if method_trends[method][metric].is_monotonic_decreasing
                    else "Mixed"
                )
            )
            for metric in metrics
        }

    # Add period dissimilarity analysis
    print("\nAnalyzing period dissimilarity...")
    dissimilarity_matrix, mean_dissimilarity, comparison_counts = (
        analyze_period_dissimilarity(combined_df, output_dir)
    )

    # Add results to summary
    if dissimilarity_matrix is not None:
        summary["Period Dissimilarity"] = {
            "mean_dissimilarity": mean_dissimilarity,
            "matrix": dissimilarity_matrix.to_dict(),
            "comparison_counts": comparison_counts.to_dict(),
        }
    else:
        summary["Period Dissimilarity"] = {
            "mean_dissimilarity": None,
            "matrix": None,
            "comparison_counts": None,
        }

    # Save summary to text file
    with open(
        os.path.join(output_dir, f"clustering_analysis_summary_{timestamp}.txt"), "w"
    ) as f:
        f.write("Clustering Analysis Summary\n")
        f.write("=========================\n\n")

        f.write("Best Method by Period:\n")
        for period, data in summary["Best Method by Period"].items():
            f.write(
                f"{period}: {data['method']} (silhouette score: {data['silhouette_score']:.3f})\n"
            )

        f.write("\nOverall Best Method:\n")
        f.write(
            f"{summary['Overall Best Method']['method']} "
            f"(average silhouette: {summary['Overall Best Method']['average_silhouette']:.3f})\n"
        )

        f.write("\nTrend Analysis:\n")
        for method, trends in summary["Trend Analysis"].items():
            f.write(f"\n{method}:\n")
            for metric, direction in trends.items():
                f.write(f"  {metric}: {direction}\n")

    return summary
