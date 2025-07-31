"""Analytics utilities for Student Simulator.

Contains helpers to plot skill trajectories and analyze student performance
using the end-of-day skill state snapshots maintained by the Student class.

The simulator maintains dual histories:
1. Event history - discrete events (learning, practice, wait)
2. End-of-day skill states - skill level snapshots at end of each day

This module uses the end-of-day snapshots for efficient trajectory plotting and analysis.
"""
from __future__ import annotations

from collections import defaultdict
from typing import List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from studentsimulator.event import ItemResponseEvent
from studentsimulator.student import Student


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
    """Plot skill trajectory over time using end-of-day snapshots.

    Args:
        student: Student whose trajectory to plot
        skill_name: Specific skill to plot, or None for all skills
        filename: Save to file if provided, otherwise show plot
        faceted: If True and plotting multiple skills, use separate subplots
    """
    if skill_name:
        # Plot single skill
        trajectory = student.end_of_day_skill_states.get_skill_trajectory(skill_name)
        if not trajectory:
            print(f"No trajectory data found for skill '{skill_name}'")
            return

        plt.figure(figsize=(6, 4))
        times, levels = zip(*trajectory)
        plt.plot(times, levels, label=skill_name, marker="o", markersize=2)
        plt.xlabel("Time (days)")
        plt.ylabel("Skill level")
        plt.title(f"Skill Trajectory - {skill_name}")
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
    else:
        # Plot all skills
        all_trajectories = student.end_of_day_skill_states.get_all_skill_trajectories()
        if not all_trajectories:
            print("No trajectory data found")
            return

        if faceted:
            # Separate subplots for each skill
            n_skills = len(all_trajectories)
            fig, axes = plt.subplots(n_skills, 1, figsize=(8, 3 * n_skills))
            if n_skills == 1:
                axes = [axes]

            for i, (skill, trajectory) in enumerate(all_trajectories.items()):
                times, levels = zip(*trajectory)
                axes[i].plot(times, levels, marker="o", markersize=2)
                axes[i].set_title(f"Skill: {skill}")
                axes[i].set_ylabel("Skill level")
                axes[i].set_ylim(0, 1)
                axes[i].grid(True, alpha=0.3)

            axes[-1].set_xlabel("Time (days)")
            plt.tight_layout()
        else:
            # All skills on one plot
            plt.figure(figsize=(8, 6))
            for skill, trajectory in all_trajectories.items():
                times, levels = zip(*trajectory)
                plt.plot(times, levels, label=skill, marker="o", markersize=2)

            plt.xlabel("Time (days)")
            plt.ylabel("Skill level")
            plt.title("Skill Trajectories")
            plt.ylim(0, 1)
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

    if filename:
        plt.savefig(filename, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def pair_item_responses_with_skill(
    student: Student,
) -> List[Tuple[ItemResponseEvent, float]]:
    """Return list of (ItemResponseEvent, skill_level_at_that_time) using end-of-day snapshots."""
    pairs: List[Tuple[ItemResponseEvent, float]] = []

    for event in student.history.get_events():
        if not isinstance(event, ItemResponseEvent) or event.item is None:
            continue

        skill_name = event.item.skill.name
        event_day = int(event.timestamp_in_days_since_initialization)

        # Get skill level from end-of-day snapshot at or before the event day
        trajectory = student.end_of_day_skill_states.get_skill_trajectory(skill_name)
        skill_level = 0.0

        for day, level in trajectory:
            if day <= event_day:
                skill_level = level
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

    plt.tight_layout()

    if filename:
        plt.savefig(filename, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
