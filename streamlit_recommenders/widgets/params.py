from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class ParamSpec:
    kind: Literal["slider", "selectbox"]
    label: str
    default: Any
    min_value: float | None = None
    max_value: float | None = None
    step: float | None = None
    options: tuple[Any, ...] | None = None


def slider(
    label: str,
    min_value: float,
    max_value: float,
    value: float,
    step: float | None = None,
) -> ParamSpec:
    return ParamSpec(
        kind="slider",
        label=label,
        default=value,
        min_value=min_value,
        max_value=max_value,
        step=step,
    )


def selectbox(label: str, options: list[Any], index: int = 0) -> ParamSpec:
    default = options[index] if options else None
    return ParamSpec(
        kind="selectbox",
        label=label,
        default=default,
        options=tuple(options),
    )


def resolve_params(
    params: dict[str, Any] | None,
    yaml_defs: list[dict] | None = None,
) -> dict[str, Any]:
    """Render sidebar widgets and return resolved param values."""
    import streamlit as st

    resolved: dict[str, Any] = {}

    if yaml_defs:
        for spec in yaml_defs:
            name = spec["name"]
            if params and name in params and not isinstance(params[name], ParamSpec):
                resolved[name] = params[name]
                continue
            resolved[name] = _render_from_yaml(spec)

    if params:
        for name, value in params.items():
            if name in resolved:
                continue
            if isinstance(value, ParamSpec):
                resolved[name] = _render_spec(name, value)
            else:
                resolved[name] = value

    return resolved


def _render_from_yaml(spec: dict) -> Any:
    import streamlit as st

    name = spec["name"]
    ptype = spec.get("type", "slider")
    if ptype == "slider":
        return st.sidebar.slider(
            spec.get("label", name),
            float(spec["min"]),
            float(spec["max"]),
            float(spec.get("default", spec["min"])),
            step=spec.get("step"),
            key=f"sr_param_{name}",
        )
    if ptype == "selectbox":
        options = spec["options"]
        default = spec.get("default", options[0])
        index = options.index(default) if default in options else 0
        return st.sidebar.selectbox(
            spec.get("label", name),
            options,
            index=index,
            key=f"sr_param_{name}",
        )
    raise ValueError(f"Unknown param type: {ptype}")


def _render_spec(name: str, spec: ParamSpec) -> Any:
    import streamlit as st

    if spec.kind == "slider":
        return st.sidebar.slider(
            spec.label,
            spec.min_value,
            spec.max_value,
            spec.default,
            step=spec.step,
            key=f"sr_param_{name}",
        )
    if spec.kind == "selectbox":
        options = list(spec.options or [])
        index = options.index(spec.default) if spec.default in options else 0
        return st.sidebar.selectbox(
            spec.label,
            options,
            index=index,
            key=f"sr_param_{name}",
        )
    raise ValueError(f"Unknown ParamSpec kind: {spec.kind}")
