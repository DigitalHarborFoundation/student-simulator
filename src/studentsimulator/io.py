import csv
import os
import random
from typing import List, Optional, Tuple

from sklearn.metrics import roc_auc_score

from studentsimulator.math import logit
from studentsimulator.student import Student


def calculate_auc(y_true: List[float], y_pred: List[float]) -> Optional[float]:
    """Calculate AUC score for binary classification.

    Args:
        y_true: List of actual outcomes (0 or 1)
        y_pred: List of predicted probabilities

    Returns:
        AUC score or None if calculation fails
    """
    try:
        return roc_auc_score(y_true, y_pred)
    except (ValueError, TypeError):
        return None


def prepare_directory(filename: str) -> str:
    """Ensure the directory for the given filename exists and return the full path."""
    # Ensure .csv extension
    if not filename.endswith(".csv"):
        filename += ".csv"

    # Get the directory path
    directory = os.path.dirname(filename)

    # If no directory specified, use current directory
    if directory == "":
        return filename

    # Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)

    # Return the full path
    return filename


def save_student_daily_skill_states_to_csv(
    students: List[Student], filename: str
) -> None:
    """Save student skill states to a CSV file."""

    path = prepare_directory(filename)
    with open(path, mode="w", newline="") as file:
        writer = csv.writer(file)
        # Write header
        header = [
            "student_id",
            "student_name",
            "skill_id",
            "skill_name",
            "day",
            "skill_level",
            "skill_level_logit",
        ]
        writer.writerow(header)

        # Write each student's skill state
        for student in students:
            for (
                skill_name,
                skill_trajectory,
            ) in (
                student.skills.end_of_day_skill_states.get_skill_trajectories().items()
            ):
                # Get the skill object for this skill name
                skill = student.skill_space.get_skill(skill_name)
                for day, skill_level in skill_trajectory:
                    writer.writerow(
                        [
                            student.id,
                            student.name,
                            skill.id,
                            skill_name,
                            day,
                            round(skill_level, 4),
                            round(logit(skill_level), 4),
                        ]
                    )


def save_student_events_to_csv(
    students: List[Student],
    filename: str,
    train_val_test_split: Optional[Tuple[float, float, float]] = None,
    observation_rate: float = 1.0,
    activity_provider_name: Optional[str] = None,
) -> None:
    """Save student activity (behavior events) to a CSV file."""

    # Validate observation rate
    if not 0.0 <= observation_rate <= 1.0:
        raise ValueError("observation_rate must be between 0.0 and 1.0")

    # Create train/validation/test split if requested
    student_splits = {}
    if train_val_test_split is not None:
        train_pct, val_pct, test_pct = train_val_test_split
        if abs(train_pct + val_pct + test_pct - 1.0) > 1e-6:
            raise ValueError("train_val_test_split percentages must sum to 1.0")

        # Shuffle students for random split
        student_ids = [student.id for student in students]
        random.shuffle(student_ids)

        # Calculate split points
        n_students = len(student_ids)
        train_end = int(n_students * train_pct)
        val_end = train_end + int(n_students * val_pct)

        # Assign splits
        for i, student_id in enumerate(student_ids):
            if i < train_end:
                student_splits[student_id] = 0  # train
            elif i < val_end:
                student_splits[student_id] = 1  # validation
            else:
                student_splits[student_id] = 2  # test

    # Collect data for AUC calculation
    y_true = []
    y_pred = []

    path = prepare_directory(filename)
    with open(path, mode="w", newline="") as file:
        writer = csv.writer(file)
        # Write header
        header = [
            "student_id",
            "day",
            "event_type",
            "skill_id",
            "item_id",
            "score",
            "prob_correct",
        ]
        if train_val_test_split is not None:
            header.append("train_val_test")
        if observation_rate < 1.0:
            header.append("observed")
        writer.writerow(header)

        # Write each student's behavior events
        for student in students:
            for event in student.skills.get_individual_events():
                if (
                    activity_provider_name is not None
                    and event.activity_provider_name != activity_provider_name
                ):
                    continue
                # Collect data for AUC calculation
                if (
                    hasattr(event, "score")
                    and hasattr(event, "prob_correct")
                    and event.score is not None
                    and event.prob_correct is not None
                ):
                    y_true.append(float(event.score))
                    y_pred.append(float(event.prob_correct))

                row = [
                    student.id,
                    event.timestamp_in_days_since_initialization,
                    type(event).__name__,
                    event.skill.id
                    if hasattr(event, "skill") and event.skill is not None
                    else None,
                    event.item.id
                    if hasattr(event, "item") and event.item is not None
                    else None,
                    event.score if hasattr(event, "score") else None,
                    round(event.prob_correct, 4)
                    if hasattr(event, "prob_correct") and event.prob_correct is not None
                    else None,
                ]
                if train_val_test_split is not None:
                    row.append(student_splits.get(student.id, 0))
                if observation_rate < 1.0:
                    # Randomly determine if this event is observed
                    is_observed = random.random() < observation_rate
                    row.append(1 if is_observed else 0)
                writer.writerow(row)

    # Calculate and report AUC
    print(f"Debug: Collected {len(y_true)} responses for AUC calculation")
    if y_true and y_pred:
        auc_score = calculate_auc(y_true, y_pred)
        if auc_score is not None:
            print(f"AUC Score: {auc_score:.4f} (based on {len(y_true)} responses)")
        else:
            print("Could not calculate AUC score")
    else:
        print("No valid data for AUC calculation")
