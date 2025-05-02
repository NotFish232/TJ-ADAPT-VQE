import typer
from typing_extensions import Annotated, Any

from .optimizers import AvailableOptimizer
from .utils.arg_parser import typer_json_parser


def main(
    optimizer_name: Annotated[
        AvailableOptimizer,
        typer.Option("-o", "--optimizer", help="The Optimizer to use for ADAPT-VQE."),
    ] = AvailableOptimizer.SGD,
    optimizer_kwargs: Annotated[
        dict[str, Any],
        typer.Option(
            "-ok",
            "--optimizer-kwargs",
            parser=typer_json_parser,
            help="Keyword Arguments to pass to the Optimizer Constructor",
        ),
    ] = {}
) -> None:
    try:
        optimizer = optimizer_name.construct(optimizer_kwargs) 
    except Exception as e:
        raise typer.BadParameter(f"Invalid Keyword Arguments for {optimizer_name}.") from e

    print(optimizer)


if __name__ == "__main__":
    typer.run(main)
