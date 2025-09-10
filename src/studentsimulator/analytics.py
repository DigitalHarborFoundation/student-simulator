"""Analytics utilities for Student Simulator.

Contains helpers to plot skill trajectories and analyze student performance
using the end-of-day skill state snapshots maintained by the Student class.

The simulator maintains dual histories:
1. Event history - discrete events (learning, practice, wait)
2. End-of-day skill states - skill level snapshots at end of each day

This module uses the end-of-day snapshots for efficient trajectory plotting and analysis.
"""
from __future__ import annotations

import warnings
from collections import defaultdict
from typing import List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns

from studentsimulator.event import ItemResponseEvent, LearningEvent, WaitEvent
from studentsimulator.student import Student


def create_event_table(student: Student) -> List[dict]:
    """Create a table of events with standardized columns for analysis and plotting.

    Returns:
        List of dictionaries, each representing an event with keys:
        - day: int (days since initialization)
        - event_type: str (learning, practice, wait)
        - skill_name: str or None
        - learned: bool or None
        - skill_level: float or None
        - item_id: str or None
        - activity_provider: str or None
    """
    events = []

    for event in student.skills.get_individual_events():
        event_dict = {
            "day": event.timestamp_in_days_since_initialization,
            "event_type": event.__class__.__name__.replace("Event", "").lower(),
            "skill_name": None,
            "learned": None,
            "skill_level": None,
            "item_id": None,
            "activity_provider": None,
        }

        if isinstance(event, LearningEvent):
            event_dict.update(
                {
                    "skill_name": event.skill.name,
                    "learned": True,
                    "skill_level": student.skills.end_of_day_skill_states.get_skill_state_for_single_skill(
                        event.skill.name
                    ).skill_level,
                }
            )
        elif isinstance(event, ItemResponseEvent):
            event_dict.update(
                {
                    "skill_name": event.skill.name,
                    "learned": student.skills.end_of_day_skill_states.get_skill_state_for_single_skill(
                        event.skill.name
                    ).learned,
                    "skill_level": student.skills.end_of_day_skill_states.get_skill_state_for_single_skill(
                        event.skill.name
                    ).skill_level,
                    "item_id": event.item.id if event.item else None,
                    "activity_provider": event.item.activity_provider_name
                    if event.item
                    else None,
                }
            )
        elif isinstance(event, WaitEvent):
            # For wait events, we don't have specific skill info
            pass

        events.append(event_dict)

    return events


def plot_skill_mastery(skill_space, students, filename="skill_mastery.png"):
    """
    Plot the skill dependency graph with node size proportional to the number of students who have each skill,
    and edge width proportional to the number of students who have both prerequisite and dependent skills.
    """

    matplotlib.use("Agg")  # Ensure non-interactive backend

    print("Generating skill mastery plot...")
    G = nx.DiGraph()
    for skill in skill_space.skills:
        G.add_node(skill.name)
        if skill.prerequisites is not None and hasattr(
            skill.prerequisites, "parent_names"
        ):
            for parent in skill.prerequisites.parent_names:
                G.add_edge(parent, skill.name)

    # Count students who have each skill (learned=True)
    skill_counts = defaultdict(int, {skill.name: 0 for skill in skill_space.skills})
    for student in students:
        for skill_name, skill_state in student.skills.get_skill_states().items():
            if getattr(skill_state, "learned", False):
                skill_counts[skill_name] += 1

    # Count students who have both skills for each edge
    edge_counts = defaultdict(int)
    for student in students:
        learned_skills = {
            k
            for k, v in student.skills.get_skill_states().items()
            if getattr(v, "learned", False)
        }
        for u, v in G.edges():
            if u in learned_skills and v in learned_skills:
                edge_counts[(u, v)] += 1

    # Normalize node sizes and edge widths
    min_node_size = 300
    max_node_size = 3000
    min_edge_width = 1
    max_edge_width = 10
    if skill_counts:
        max_count = max(skill_counts.values())
    else:
        max_count = 1
    if edge_counts:
        max_edge_count = max(edge_counts.values())
    else:
        max_edge_count = 1
    node_sizes = [
        min_node_size
        + (max_node_size - min_node_size) * (skill_counts.get(n, 0) / max_count)
        for n in G.nodes()
    ]
    edge_widths = [
        min_edge_width
        + (max_edge_width - min_edge_width)
        * (edge_counts.get((u, v), 0) / max_edge_count)
        for u, v in G.edges()
    ]

    # Layout
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except Exception as e:
        print(
            "Warning: Could not use Graphviz layout for skill mastery plot. Reason:",
            e,
        )
        print(
            "Falling back to hierarchical layout. For best results, install Graphviz (dot)."
        )
        # Use hierarchical layout as fallback
        pos = nx.kamada_kawai_layout(G)
        # Adjust positions to be more hierarchical
        # Get topological sort to determine levels
        try:
            topo_order = list(nx.topological_sort(G))
            # Create a hierarchical layout manually
            levels = {}
            for node in topo_order:
                # Find the level of this node (max level of parents + 1)
                parent_levels = [levels.get(pred, 0) for pred in G.predecessors(node)]
                level = max(parent_levels) + 1 if parent_levels else 0
                levels[node] = level

            # Position nodes by level
            max_level = max(levels.values()) if levels else 0
            for node, level in levels.items():
                # Y position: higher level = higher Y (top of plot)
                y = 1.0 - (level / max_level) if max_level > 0 else 0.5
                # X position: distribute nodes at same level horizontally
                nodes_at_level = [n for n, level1 in levels.items() if level1 == level]
                if len(nodes_at_level) > 1:
                    idx = nodes_at_level.index(node)
                    x = (idx - (len(nodes_at_level) - 1) / 2) / max(
                        len(nodes_at_level), 1
                    )
                else:
                    x = 0
                pos[node] = (x, y)
        except Exception:
            # If topological sort fails, use spring layout as last resort
            pos = nx.spring_layout(G, seed=42)
    nx.draw(
        G,
        pos,
        with_labels=False,  # We'll add custom labels
        node_color="lightblue",
        edge_color="gray",
        node_size=node_sizes,
        width=edge_widths,
        arrowsize=20,
    )

    # Add custom labels with skill name and count
    for node, (x, y) in pos.items():
        count = skill_counts.get(node, 0)
        plt.text(
            x,
            y,
            f"{node}\n({count})",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    plt.title("Skill Mastery and Dependencies")

    # Use constrained_layout instead of tight_layout to avoid warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        try:
            plt.tight_layout()
        except Exception:
            pass  # Ignore tight_layout errors

    plt.savefig(filename)
    plt.close()


def plot_skill_trajectory(
    student: Student,
    skill_name: str = None,
    filename: str = None,
    faceted: bool = False,
):
    """Plot skill trajectory over time using daily skill states DataFrame with event coloring.

    Args:
        student: Student whose trajectory to plot
        skill_name: Specific skill to plot, or None for all skills
        filename: Save to file if provided, otherwise show plot
        faceted: If True and plotting multiple skills, use separate subplots
    """
    # Set seaborn style for better-looking plots
    sns.set_style("whitegrid")
    sns.set_palette("husl")

    # Get the daily skill states DataFrame
    df = student.skills.get_daily_skill_states_dataframe()

    if skill_name:
        # Plot single skill
        skill_df = df[df["skill_name"] == skill_name].sort_values("day")
        if skill_df.empty:
            print(f"No data found for skill '{skill_name}'")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot the line
        ax.plot(
            skill_df["day"],
            skill_df["skill_level"],
            color="steelblue",
            linewidth=4,
            alpha=0.8,
        )

        # Plot points with color coding
        for _, row in skill_df.iterrows():
            day, level = row["day"], row["skill_level"]

            if row["num_learning_events"] > 0:
                # Black dot for learning events
                ax.scatter(day, level, color="black", s=100, zorder=5, alpha=0.8)
            elif row["num_practice_events"] > 0:
                # Gray dot for practice events (only if no learning occurred)
                ax.scatter(day, level, color="gray", s=60, zorder=4, alpha=0.6)

        ax.set_xlabel("Time (days)", fontsize=12)
        ax.set_ylabel("Skill level", fontsize=12)
        ax.set_title(f"Skill Trajectory - {skill_name}", fontsize=14, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

        # Add legend for point types
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="black",
                markersize=8,
                label="Learning Event",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="gray",
                markersize=6,
                label="Practice Event",
            ),
        ]
        ax.legend(handles=legend_elements, loc="lower right")

    else:
        # Plot all skills
        if df.empty:
            print("No trajectory data found")
            return

        if faceted:
            # Separate subplots for each skill
            skills = df["skill_name"].unique()
            n_skills = len(skills)
            fig, axes = plt.subplots(n_skills, 1, figsize=(10, 4 * n_skills))
            if n_skills == 1:
                axes = [axes]

            for i, skill in enumerate(skills):
                skill_df = df[df["skill_name"] == skill].sort_values("day")

                # Plot the line
                axes[i].plot(
                    skill_df["day"],
                    skill_df["skill_level"],
                    color="steelblue",
                    linewidth=4,
                    alpha=0.8,
                )

                # Plot points with color coding
                for _, row in skill_df.iterrows():
                    day, level = row["day"], row["skill_level"]

                    if row["num_learning_events"] > 0:
                        axes[i].scatter(
                            day, level, color="black", s=100, zorder=5, alpha=0.8
                        )
                    elif row["num_practice_events"] > 0:
                        axes[i].scatter(
                            day, level, color="gray", s=60, zorder=4, alpha=0.6
                        )

                axes[i].set_title(f"Skill: {skill}", fontsize=12, fontweight="bold")
                axes[i].set_ylabel("Skill level", fontsize=10)
                axes[i].set_ylim(0, 1)
                axes[i].grid(True, alpha=0.3)

            axes[-1].set_xlabel("Time (days)", fontsize=10)
            plt.tight_layout()
        else:
            # All skills on one plot
            fig, ax = plt.subplots(figsize=(12, 8))
            colors = sns.color_palette("husl", len(df["skill_name"].unique()))

            for idx, skill in enumerate(df["skill_name"].unique()):
                skill_df = df[df["skill_name"] == skill].sort_values("day")

                # Plot the line
                ax.plot(
                    skill_df["day"],
                    skill_df["skill_level"],
                    color=colors[idx],
                    linewidth=4,
                    alpha=0.8,
                    label=skill,
                )

                # Plot points with color coding
                for _, row in skill_df.iterrows():
                    day, level = row["day"], row["skill_level"]

                    if row["num_learning_events"] > 0:
                        ax.scatter(
                            day, level, color="black", s=100, zorder=5, alpha=0.8
                        )
                    elif row["num_practice_events"] > 0:
                        ax.scatter(day, level, color="gray", s=60, zorder=4, alpha=0.6)

            ax.set_xlabel("Time (days)", fontsize=12)
            ax.set_ylabel("Skill level", fontsize=12)
            ax.set_title("Skill Trajectories", fontsize=14, fontweight="bold")
            ax.set_ylim(0, 1)
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            ax.grid(True, alpha=0.3)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                plt.tight_layout()

    if filename:
        plt.savefig(filename, bbox_inches="tight", dpi=300)
        plt.close()
    else:
        # Suppress show() warning in non-interactive environments
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            try:
                plt.show()
            except Exception:
                pass  # Ignore show() errors in non-interactive environments


def pair_item_responses_with_skill(
    student: Student,
) -> List[Tuple[ItemResponseEvent, float]]:
    """Return list of (ItemResponseEvent, skill_level_at_that_time) using end-of-day snapshots."""
    pairs: List[Tuple[ItemResponseEvent, float]] = []

    for event in student.skills.get_individual_events():
        if not isinstance(event, ItemResponseEvent) or event.item is None:
            continue

        skill_name = event.item.skill.name
        event_day = int(event.timestamp_in_days_since_initialization)

        # Get skill level from end-of-day snapshot at or before the event day
        trajectories = student.skills.end_of_day_skill_states.get_skill_trajectories(
            skill_name
        )
        skill_level = 0.0

        if skill_name in trajectories:
            for day, skill_state in trajectories[skill_name]:
                if day <= event_day:
                    skill_level = skill_state.skill_level
                else:
                    break

        pairs.append((event, skill_level))

    return pairs


def plot_accuracy_vs_skill(
    student: Student, skill_name: str, bins: int = 5, filename: str = None
):
    """Plot accuracy vs skill level for a given student's skill using daily snapshots.

    Args:
        student: The student whose data to plot
        skill_name: The skill to analyze
        bins: Number of bins to group skill levels into
        filename: Save to file if provided, otherwise show plot
    """
    pairs = pair_item_responses_with_skill(student)
    data = [
        (level, event.score)
        for event, level in pairs
        if event.item.skill.name == skill_name and event.score is not None
    ]

    if not data:
        print(f"No item responses found for skill '{skill_name}'.")
        return

    levels, scores = zip(*data)

    # Bin data
    levels_arr = np.array(levels)
    scores_arr = np.array(scores)
    bin_edges = np.linspace(0, 1, bins + 1)

    bin_centers = []
    bin_accuracies = []
    bin_counts = []

    for i in range(bins):
        mask = (levels_arr >= bin_edges[i]) & (levels_arr < bin_edges[i + 1])
        if i == bins - 1:  # Include the right edge for the last bin
            mask = (levels_arr >= bin_edges[i]) & (levels_arr <= bin_edges[i + 1])

        if mask.any():
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_accuracies.append(scores_arr[mask].mean())
            bin_counts.append(mask.sum())

    plt.figure(figsize=(6, 4))
    plt.plot(bin_centers, bin_accuracies, marker="o", linewidth=2, markersize=6)
    plt.xlabel("Skill proficiency (binned)")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy vs Skill Level - {student.name} - {skill_name}")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)

    # Add count annotations
    for x, y, count in zip(bin_centers, bin_accuracies, bin_counts):
        plt.annotate(
            f"n={count}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        plt.tight_layout()

    if filename:
        plt.savefig(filename, bbox_inches="tight")
        plt.close()
    else:
        # Suppress show() warning in non-interactive environments
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            try:
                plt.show()
            except Exception:
                pass  # Ignore show() errors in non-interactive environments
