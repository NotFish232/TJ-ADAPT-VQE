import inspect
from abc import ABC, abstractmethod

from typing_extensions import Any, Self, Type, TypeVar

SerializableType = TypeVar("SerializableType", bound="Serializable")


class Serializable(ABC):
    """
    Helper class that allows all classes, like Ansatz and pools, to be decomposed into saveable JSONs,
    saved to files, and then recreated. Inherits from `abc.ABC`. Base class for other base classes in this project.
    """

    @staticmethod
    @abstractmethod
    def _type() -> str:
        """
        Returns the type of the class. Should be overriden by the next in line class, which in turn is a base class.

        Returns:
            str: A string representation of the type of the class.
        """

        pass

    @staticmethod
    @abstractmethod
    def _name() -> str:
        """
        Returns the name of the class. Should be overriden by the classes that override the base class that inherits from this class.

        Returns:
            str: A string representation of the name of the class.
        """

        pass

    @property
    def _config_params(self: Self) -> list[str]:
        """
        Returns the properties of the class that are necessary to reconstruct it. These are normally the
        mandatory arguments to the subclass' constructor.

        Args:
            self (Self): A reference to the current class instance.

        Returns:
            list[str]: The parameters of the class required for configuration.
        """

        return []

    @classmethod
    def all(cls: Type[SerializableType]) -> list[Type[SerializableType]]:
        """
        Returns a list of all of current class and subclasses recursively.
        Only returns those that are not abstract.

        Args:
            cls (Type[SerializableType]): A reference to the current class.

        Returns:
            list[Type[SerializableType]]: A list of all subclasses.
        """

        classes = cls.__subclasses__()
        classes += [s for c in classes for s in c.all()]
        classes += [cls]
        classes = [c for c in classes if not inspect.isabstract(c)]

        return classes

    @staticmethod
    def _filter_class_config(config: dict[str, Any]) -> dict[str, Any]:
        """
        Filters the config for keys that should be passed to the class constructor.
        This simply should be all keys that do not start with an underscore.

        Args:
           config (dict[str,Any]): The configuration to filter.

         Returns:
            dict[str, Any]: A filtered dictionary where no key starts with an underscore
        """

        return {k: v for k, v in config.items() if not k.startswith("_")}

    def to_config(self: Self) -> dict[str, Any]:
        """
        Converts the class instance to a dictionary config where keys are strings and values are serializable.
        Adds the "_name" and "_type" keys representing the name and type of the object.

        Args:
            self (Self): A reference to the current class instance.

        Returns:
            dict[str, Any]: The dictionary representation of the object's config.
        """

        config = {"_type": self._type(), "_name": self._name()}
        config |= {p: getattr(self, p) for p in self._config_params}

        return config

    @classmethod
    def from_config(
        cls: Type[SerializableType],
        config: dict[str, Any],
        **class_kwargs: Any,
    ) -> SerializableType:
        """
        Converts back from a config to the actual class.

        Args:
            cls (Type[SerializableType]): A reference to the current class.
            config (dict[str, Any]): The config to convert.
            class_kwargs (**dict[str, Any]): Additional non-serializable kwargs that should be passed to the class constructor.

        Returns:
            Serializable: An instance of the class
        """

        # do it this way to support molecules that all have the same class
        possible_classes = cls.all()
        possible_classes = [c for c in possible_classes if c._type() == config["_type"]]

        if len(possible_classes) != 1:
            possible_classes = [
                c for c in possible_classes if c._name() == config["_name"]
            ]

        cls = possible_classes[0]

        class_config = Serializable._filter_class_config(config)

        return cls(**class_config, **class_kwargs)

    def __str__(self: Self) -> str:
        class_config = Serializable._filter_class_config(self.to_config())

        name = self.__class__.__name__

        return f"{name}({', '.join(f'{k}={v}' for k, v in class_config.items())})"

    def __repr__(self: Self) -> str:
        return repr(str(self))
