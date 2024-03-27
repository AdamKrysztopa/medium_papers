import typing
from typing import Callable, Union, Tuple, Dict, Any
import inspect


class TypingBasedTypeError(TypeError):
    """Custom exception class for type checking errors based on the typing module."""

    def __init__(
        self,
        func: callable,
        key: str,
        expected_type: Union[type, Tuple],
        received_type: type,
    ):
        self.func_name = func.__name__
        self.key = key
        self.expected_type = expected_type
        self.received_type = received_type
        super().__init__(
            f'Parameter "{key}" in the function {self.func_name} must be of type {expected_type}, but got {received_type} instead.'
        )


def get_types(func: Callable) -> Dict:
    """
    It is a modified function from multimethod.
    It gets the function and returns the dictionary in the format: variable_name: variable_type.
    """
    if not hasattr(func, "__annotations__"):
        return ()
    annotations = dict(typing.get_type_hints(func))
    annotations.pop("return", None)
    params = inspect.signature(func).parameters
    return {name: annotations.pop(name, object) for name in params}


def get_tuple_from_union(union_type: typing._SpecialForm) -> Tuple:
    """Returns a tuple from a Union type."""
    return tuple(union_type.__args__)


def update_typing_objects(types_dict: Dict) -> Dict:
    """Changes all Union types to tuples of types and makes objects from Any."""
    for var_name, _type in types_dict.items():
        if type(_type) == typing._GenericAlias:
            if _type.__origin__ == Union:
                types_dict[var_name] = get_tuple_from_union(_type)
        elif _type == Any:
            types_dict[var_name] = object
    return types_dict


def validate_by_typing(func: callable) -> callable:
    """Decorator to raise an error when the type data are inconsistent with typing."""

    def wrapper(*args, **kwargs):
        kwargs.update(zip(func.__code__.co_varnames, args))
        _types = update_typing_objects(get_types(func))
        for key, value in kwargs.items():
            if not isinstance(value, _types[key]):
                raise TypingBasedTypeError(
                    key=key,
                    func=func,
                    expected_type=_types[key],
                    received_type=type(value),
                )

        return func(**kwargs)

    return wrapper


@validate_by_typing
def rectangle_area(x: Union[int, float], y: Union[int, float]) -> float:
    return float(x * y)


def rectangle_test():
    x = 2.5
    y = 3
    print(f"Correct types: x: {type(x)}, y: {type(y)}")
    print(rectangle_area(x, y))

    x = 2.5
    y = "3.0"
    print("Now incorrect values:")
    print(f"correct type: x: {type(x)}, incorrect type y: {type(y)}")
    print(rectangle_area(x, y))


if __name__ == "__main__":
    try:
        rectangle_test()
    except Exception as e:
        print("!! ERROR !!")
        print(e)
