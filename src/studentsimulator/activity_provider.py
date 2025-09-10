"""This class maintains objects that are created by an Activity Provider
including an Item, an ItemPool, and a FixedFormAssessment."""

import random
from typing import Any, Dict, List, Optional, Union

import joblib

from studentsimulator.event import BehaviorEventCollection
from studentsimulator.general import Model
from studentsimulator.item import Item, ItemPool
from studentsimulator.skill import Skill, SkillSpace
from studentsimulator.student import Student


class FixedFormAssessment(Model):
    """A fixed-form assessment is a set of items that are presented to the student
    in a fixed order. This is typically used for summative assessments where the
    items are not adaptive and the student does not receive feedback on their responses.
    """

    items: Optional[List[Item]]  # list of item IDs in the assessment

    def __iter__(self):
        # If self.items is None, return an empty iterator; else, iterate over items
        return iter(self.items or [])


class ActivityProvider(Model):
    """The ActivityProvider creates learning experiences for students.
    The generator takes the perspective of a "provider" of learning or assessment experiences.
    This encapsulates logic for things like:
    - a lesson, video, or other instructional content ('gating event')
    - a tutoring session (interleaved instruction and practice)
    - formative practice with feedback (practice leading to learning)
    - summative assessment without feedback (practice not leading to learning)

    Notably, the ActivityProvider is *stateless*. It does not keep a
    memory of past interactions with students. Rather, each interaction accepts
    a student's history and generates a new event based on that history.

    Pragmatically, this design decision allows the 'single source of truth' for
    student history to be the student object itself.

    A single ActivityProvider may or may not have visibility into the full
    history of a student's learning.
    """

    skill_space: SkillSpace = SkillSpace(
        skills=[]
    )  # skills that this provider can generate items for
    item_pools: Dict[str, "ItemPool"] = {}
    name: str = "ActivityProvider"

    def construct_item_pool(
        self, name: str, skills: List[Skill], n_items_per_skill: int, **kwargs: Any
    ) -> "ItemPool":
        """Construct an item pool with a specified number of items for the given skills.

        Additional kwargs are passed to the Item constructor.
        """
        item_pool = []
        skills = self.validate_skill_list(skills)
        for skill in skills:
            for _ in range(n_items_per_skill):
                item = Item(skill=skill, activity_provider_name=self.name, **kwargs)
                item_pool.append(item)
        item_pool = ItemPool(name=name, items=item_pool)
        self.item_pools[name] = item_pool
        return item_pool

    def register_skills(self, skill_space: SkillSpace) -> None:
        """Register a list of skills with the provider."""
        self.skill_space = skill_space

    def get_skill(self, skill_id: Union[str, Skill]) -> Optional[Skill]:
        """Get a skill by its ID."""
        for skill in self.skill_space.skills:
            if skill.name == skill:
                return skill

        return None

    def validate_skill_list(
        self, skills: Union[List[Skill], List[str], SkillSpace]
    ) -> list[Skill]:
        """Validate and convert a list of skills to Skill objects."""
        # if it's a SkillSpace, return the skills directly
        if isinstance(skills, SkillSpace):
            return skills.skills
        elif isinstance(skills, list):
            skill_list = []
            for skill in skills:
                if isinstance(skill, Skill):
                    skill_list.append(skill)
                elif isinstance(skill, str):
                    skill_obj = self.skill_space.get_skill(skill)
                    if skill_obj is None:
                        raise ValueError(f"Skill not found: {skill}")
                    skill_list.append(skill_obj)
                elif isinstance(skill, dict):
                    # If it's a dict, retrieve it by name
                    skill_name = skill.get("name")
                    if skill_name is None:
                        raise ValueError("Skill dict must have a 'name' key.")
                    skill_obj = self.skill_space.get_skill(skill_name)
                    if skill_obj is None:
                        raise ValueError(f"Skill not found: {skill_name}")
                    skill_list.append(skill_obj)
                else:
                    raise ValueError(
                        "Skills must be a list of Skill objects or strings."
                    )
            return skill_list
        else:
            raise ValueError(
                "Skills must be a SkillSpace or a list of Skill objects or strings."
            )

    def generate_fixed_form_assessment(
        self,
        item_pool: Union[str, "ItemPool"],
        n_items: int,
        skills: Union[list[Skill], list[str], SkillSpace] = [],
    ) -> "FixedFormAssessment":
        """Generate a random assessment with a specified number of items.

        Args:
            item_pool: The name of the item pool to use, or the item pool object itself.
            n_items: The number of items to include in the assessment.
            skills: The skills to include in the assessment. If None, all skills in the skill space will be used.
        """

        # Create a list of skill names that we can use for filtering
        # if skills is an empty list, use all skills in the skill space
        skills_to_use = (
            self.validate_skill_list(skills) if skills else self.skill_space.skills
        )
        # Filter items in the item pool based on the skills to use

        if isinstance(item_pool, str):
            item_pool = self.item_pools[item_pool]
        valid_items = []
        for item in item_pool:
            if item.skill.name in [s.name for s in skills_to_use]:
                valid_items.append(item)

        # Randomly select n items from the filtered item pool, without replacement
        if len(valid_items) < n_items:
            raise ValueError(
                f"Not enough items in the pool for the requested number of items. "
                f"Requested {n_items}, valid items: {len(valid_items)}."
            )
        selected_items = random.sample(valid_items, n_items)

        return FixedFormAssessment(items=selected_items)

    def administer_fixed_form_assessment(
        self,
        student_or_students: Union[Student, list[Student]],
        test: FixedFormAssessment,
        threads=1,
    ) -> "List[BehaviorEventCollection]":
        """Simulate taking a test with no formative feedback."""
        # if student is a single student, wrap it in a list
        if isinstance(student_or_students, Student):
            students = [student_or_students]
        else:
            students = student_or_students
        if threads == 1:
            test_results = [
                self._administer_fixed_form_assessment_to_single_student(student, test)
                for student in students
            ]
        else:
            test_results = joblib.Parallel(n_jobs=threads)(
                joblib.delayed(
                    self._administer_fixed_form_assessment_to_single_student
                )(student, test)
                for student in students
            )
        assert len(test_results) == len(
            students
        ), "Got a different number of test results than students"
        # Update student histories
        for student, test_result in zip(students, test_results):
            student.skills.record_event(student, test_result)
        return test_results

    def _administer_fixed_form_assessment_to_single_student(
        self, student: Student, test: FixedFormAssessment
    ) -> "BehaviorEventCollection":
        """Simulate taking a test with no formative feedback."""
        responses = []
        for item in test:
            response = student._respond_to_item(
                item=item, feedback=False, category="Assessment"
            )
            responses.append(response)
        test_result = BehaviorEventCollection(
            student_id=student.id,
            timestamp_in_days_since_initialization=student.days_since_initialization,
            behavioral_events=responses,
            activity_provider_name=self.name,
        )

        return test_result

    def administer_lesson(
        self,
        student: Student,
        skill: Union[Skill, str],
    ):
        """Administer a lesson to a student. If skill is None, administer a lesson for all skills in the provider's skill_space."""
        if isinstance(skill, str):
            skill_obj = self.skill_space.get_skill(skill)
            if skill_obj is None:
                raise ValueError(f"Skill not found: {skill}")
            skill = skill_obj
        student.learn(skill, activity_provider_name=self.name)

    def administer_practice(
        self,
        student: Student,
        skill: Union[Skill, str],
        n_items: int,
        item_pool: Union[str, ItemPool] = None,
    ):
        """Administer practice for a skill.

        Args:
            student: The student to administer practice to.
            skill: The skill to practice (name or Skill object).
            n_items: Number of items to practice with.
            item_pool: Either an ItemPool object or the name of a registered pool.
        """
        # Resolve skill if string
        if isinstance(skill, str):
            skill_obj = self.skill_space.get_skill(skill)
            if skill_obj is None:
                raise ValueError(f"Skill not found: {skill}")
            skill = skill_obj

        # Resolve item pool
        if item_pool is None:
            raise ValueError("Item pool is required for practice")
        if isinstance(item_pool, str):
            if item_pool not in self.item_pools:
                raise ValueError(f"Item pool not found: {item_pool}")
            item_pool = self.item_pools[item_pool]

        # Draw n_items from item pool without replacement
        if len(item_pool) < n_items:
            raise ValueError(
                f"Not enough items in the pool for the requested number of items. "
                f"Requested {n_items}, valid items: {len(item_pool)}."
            )
        selected_items = random.sample(item_pool, n_items)

        for item in selected_items:
            student.practice(skill, item=item)
