"""Analytics utilities for Student Simulator.

Contains helpers to plot skill trajectories and analyze student performance
using the end-of-day skill state snapshots maintained by the Student class.

The simulator maintains dual histories:
1. Event history - discrete events (learning, practice, wait)
2. End-of-day skill states - skill level snapshots at end of each day

This module uses the end-of-day snapshots for efficient trajectory plotting and analysis.
"""
from __future__ import annotations

from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from studentsimulator.student import ItemResponseEvent, Student


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

    for event in student.event_history.get_events():
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
