from __future__ import annotations

from functools import wraps
from typing import ClassVar

from pydantic import BaseModel, ConfigDict


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # one counter per subclass, created automatically
    _counter: ClassVar[int] = 0  # ensure base class can be instantiated

    id: int = -1  # not required—filled in later

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._counter = 0  # fresh counter for this subclass
        # PEP 487 guarantees this runs once per subclass
        # [oai_citation:1‡peps.python.org](https://peps.python.org/pep-0487/?utm_source=chatgpt.com)

    def __init__(self, **data):
        super().__init__(**data)  # normal Pydantic validation
        if self.id < 0:  # assign only if caller didn’t supply one
            cls = self.__class__
            self.id = cls._counter
            cls._counter += 1


def validate_int_list(field_name: str):
    """Decorator for validating student-related function inputs."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Validate practice counts
            if field_name in kwargs:
                int_list = kwargs[field_name]
                if isinstance(int_list, int) and int_list < 0:
                    raise ValueError(
                        f"Input for field '{field_name}' count must be non-negative, got '{int_list}'"
                    )
                elif isinstance(int_list, (list, tuple)) and len(int_list) == 2:
                    if not (int_list[0] >= 0 and int_list[1] >= int_list[0]):
                        raise ValueError(
                            f"Invalid input for field '{field_name}': {int_list}"
                        )

            return func(*args, **kwargs)

        return wrapper

    return decorator
