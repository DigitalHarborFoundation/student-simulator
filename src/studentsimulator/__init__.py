"""
Student Simulator - A Python framework for simulating student learning and assessment interactions.

This package provides tools for:
- Creating skill-based learning models with prerequisites
- Simulating student responses to assessments
- Generating educational data for research and machine learning
- Visualizing skill dependencies and mastery
"""

import matplotlib

from .activity_provider import ActivityProvider
from .factory import create_random_students
from .general import Skill, SkillSpace
from .student import Student, save_student_activity_to_csv, save_student_profile_to_csv

matplotlib.use("Agg")

__version__ = "0.1.1"
__all__ = [
    "Skill",
    "SkillSpace",
    "Student",
    "ActivityProvider",
    "create_random_students",
    "save_student_profile_to_csv",
    "save_student_activity_to_csv",
]
