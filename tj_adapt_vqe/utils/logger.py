from typing_extensions import Any, Self


class Logger:
    """
    Logger class that maanges all the data wanting to be saved during training
    Will be repurposed later for checkpointing and loading / writing to files
    """

    def __init__(self: Self) -> None:
        self.config_options: dict[str, Any] = {}
        self.logged_values: dict[str, list[Any]] = {}

    def add_config_option(self: Self, name: str, config: Any) -> None:
        """
        Add a new config option from training, example includes which Pool or which Optimizer
        """
        self.config_options[name] = config
    
    def add_logged_value(self: Self, name: str, value: Any) -> None:
        """
        Adds a new logged value to the end of the list of the name
        Examples of logged values include observable values
        """

        if name not in self.logged_values:
            self.logged_values[name] = []

        self.logged_values[name].append(value)