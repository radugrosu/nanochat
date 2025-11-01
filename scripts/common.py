from typing import Any, TypeVar

import typer

T = TypeVar("T")


def help(msg: str):
    return typer.Option(help=msg)


def config_from_context(ctx: typer.Context) -> tuple[dict[str, Any], list[tuple[str, T, T]]]:
    user_config = ctx.params
    overrides = []
    for param in ctx.command.params:
        name = param.name
        assert name is not None
        if param.default != user_config[name]:
            override = name, param.default, user_config[name]
            overrides.append(override)
    return user_config, overrides
