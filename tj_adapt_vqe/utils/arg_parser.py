import json

import typer
from typing_extensions import Any


def typer_json_parser(json_str: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(json_str, dict):
        return json_str
    
    try:
        return json.loads(json_str)
    except Exception as e:
        raise typer.BadParameter("Argument is not valid JSON") from e

