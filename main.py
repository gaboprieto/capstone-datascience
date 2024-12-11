import argparse
import os
import json
import numpy as np
from datetime import datetime

from src.clustering.metrics import (
    calculate_clustering_metrics_with_mapping,
    calculate_similarity_matrix,
    compare_clustering_methods,
)
from src.utils.data_processing import get_data_from_csv, process_course_data
from src.utils.analysis import (
    analyze_clustering_performance,
    analyze_team_assignments,
    compare_team_grades,
    save_and_analyze_results,
)
from src.visualization.comparison import (
    plot_feature_averages,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Team Formation Analysis and Clustering"
    )
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--periods",
        nargs="+",
        default=["beginning", "middle", "end"],
        help="Time periods to analyze",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["kmeans", "hierarchical", "dbscan"],
        help="Clustering methods to use",
    )
    parser.add_argument("--grades-file", help="Path to grades Excel file")
    return parser.parse_args()


def setup_directories(base_dir):
    """Create organized output directories."""
    dirs = {
        "plots": {
            "clustering": os.path.join(base_dir, "plots", "clustering"),
            "features": os.path.join(base_dir, "plots", "features"),
            "evolution": os.path.join(base_dir, "plots", "evolution"),
            "comparison": os.path.join(base_dir, "plots", "comparison"),
        },
        "data": os.path.join(base_dir, "data"),
        "results": {
            "metrics": os.path.join(base_dir, "results", "metrics"),
            "assignments": os.path.join(base_dir, "results", "assignments"),
            "analysis": os.path.join(base_dir, "results", "analysis"),
        },
    }

    # Create all directories
    for category in dirs.values():
        if isinstance(category, dict):
            for directory in category.values():
                os.makedirs(directory, exist_ok=True)
        else:
            os.makedirs(category, exist_ok=True)

    return dirs


def main():
    """Main execution function."""
    args = parse_args()
    dirs = setup_directories(args.output)

    # Load and process data
    print("\nLoading and processing data...")
    df = get_data_from_csv(args.input)

    cen4072_data = df[df["course_name"] == "CEN4072"]

    all_comparison_tables = []
    all_team_metrics_dict = {}
    semester_year_keys = []

    team_assignments = {
        "original": {},  # semester_key -> {team_id -> [user_ids]}
        "clustered": {},  # semester_key -> {team_id -> [user_ids]}
    }

    # Store all processed features
    all_features = {}

    # Process each semester/year combination
    for (semester, year), semester_data in cen4072_data.groupby(["semester", "year"]):
        print(f"\nProcessing {semester} {year}")
        semester_key = f"{semester}_{year}"

        # First, get end period clustering results
        end_data_df = semester_data[semester_data["checkpoint"] == "end"]
        end_features, end_target, end_user_ids = process_course_data(end_data_df)

        if end_features is not None and len(end_features) >= 2:

            all_features[semester_key] = end_features

            end_similarities = calculate_similarity_matrix(end_features)
            end_original_labels = end_target.values

            # Store original team assignments
            team_assignments["original"][semester_key] = {
                int(label): end_user_ids[end_original_labels == label].tolist()
                for label in np.unique(end_original_labels)
            }

            # Get clustering results for end period
            end_clustering_results, end_comparison_table, end_team_metrics = (
                compare_clustering_methods(
                    end_features,
                    end_original_labels,
                    end_similarities,
                    year,
                    semester,
                    "CEN4072",
                    "end",
                    dirs["plots"]["clustering"],
                )
            )

            # Store clustered team assignments (using hierarchical clustering)
            if "hierarchical" in end_clustering_results:
                hierarchical_labels = end_clustering_results["hierarchical"]
                team_assignments["clustered"][semester_key] = {
                    int(label): end_user_ids[hierarchical_labels == label].tolist()
                    for label in np.unique(hierarchical_labels)
                    if label != -1  # Exclude noise points if any
                }

            # Store end period results
            all_comparison_tables.append(end_comparison_table)
            all_team_metrics_dict.update(end_team_metrics)
            semester_year_keys.append((semester, year, "end"))

            # Map end period clusters to other periods
            for period in ["beginning", "middle"]:
                period_data_df = semester_data[semester_data["checkpoint"] == period]
                period_features, period_target, period_user_ids = process_course_data(
                    period_data_df
                )

                if period_features is not None and len(period_features) >= 2:
                    # Map clusters based on user_ids
                    mapped_clustering_results = {}
                    for method, end_labels in end_clustering_results.items():
                        mapped_labels = np.full(len(period_user_ids), -1)

                        # Create mapping from end period to current period
                        for i, user_id in enumerate(period_user_ids):
                            end_idx = np.where(end_user_ids == user_id)[0]
                            if len(end_idx) > 0:
                                mapped_labels[i] = end_labels[end_idx[0]]

                        mapped_clustering_results[method] = mapped_labels

                    # Calculate metrics using mapped clusters
                    period_similarities = calculate_similarity_matrix(period_features)
                    period_original_labels = period_target.values

                    period_comparison_table, period_team_metrics = (
                        calculate_clustering_metrics_with_mapping(
                            period_features,
                            period_original_labels,
                            mapped_clustering_results,
                            period_similarities,
                            year,
                            semester,
                            "CEN4072",
                            period,
                        )
                    )

                    if period_comparison_table is not None:
                        all_comparison_tables.append(period_comparison_table)
                        all_team_metrics_dict.update(period_team_metrics)
                        semester_year_keys.append((semester, year, period))

    # Save team assignments to JSON file
    team_assignments_path = os.path.join(dirs["data"], "team_assignments.json")
    with open(team_assignments_path, "w") as f:
        json.dump(team_assignments, f, indent=2)

    analyze_team_assignments(team_assignments_path, args.input)

    feature_averages_path = os.path.join(dirs["data"], "team_feature_averages.json")

    grade_comparison = compare_team_grades(
        assignments_path=team_assignments_path,
        grades_file_path=args.grades_file,
        output_dir=dirs["results"]["analysis"],
    )

    plot_feature_averages(feature_averages_path, dirs["plots"]["features"])

    # Analyze results
    summary_stats, summary_text = analyze_clustering_performance(
        all_comparison_tables, semester_year_keys, dirs["results"]["analysis"]
    )

    # Save and analyze final results
    save_and_analyze_results(
        all_comparison_tables,
        all_team_metrics_dict,
        semester_year_keys,
        dirs["results"]["analysis"],
        dirs["plots"]["evolution"],
    )


if __name__ == "__main__":
    start_time = datetime.now()
    try:
        main()
    except Exception as e:
        print(f"\nError in main execution: {str(e)}")
        raise
    finally:
        execution_time = datetime.now() - start_time
        print(f"\nTotal execution time: {execution_time}")
