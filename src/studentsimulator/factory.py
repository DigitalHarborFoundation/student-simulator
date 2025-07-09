from typing import List, Union

from studentsimulator.general import SkillSpace
from studentsimulator.student import Student


def create_random_students(
    skill_space: SkillSpace,
    n_students: int,
    practice_count: Union[int, list[int]] = [1, 9],
    name_prefix: str = "student",
) -> List[Student]:
    """Create n students with randomly initialized skill values."""
    students = []
    for i in range(n_students):
        student = Student(
            name=f"{name_prefix}_{i}", skill_space=skill_space
        ).initialize_skill_values(practice_count=practice_count)
        students.append(student)
    return students
