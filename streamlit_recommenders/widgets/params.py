from dataclasses import dataclass
from typing import Any, Literal

from streamlit_recommenders.runtime.keys import param_key


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
    *,
    key_prefix: str = "",
    container: Any | None = None,
) -> dict[str, Any]:
    """Render sidebar widgets and return resolved param values."""
    import streamlit as st

    ui = container or st.sidebar
    resolved: dict[str, Any] = {}

    if yaml_defs:
        for spec in yaml_defs:
            name = spec["name"]
            if params and name in params and not isinstance(params[name], ParamSpec):
                resolved[name] = params[name]
                continue
            resolved[name] = _render_from_yaml(spec, key_prefix=key_prefix, container=ui)

    if params:
        for name, value in params.items():
            if name in resolved:
                continue
            if isinstance(value, ParamSpec):
                resolved[name] = _render_spec(name, value, key_prefix=key_prefix, container=ui)
            else:
                resolved[name] = value

    return resolved


def split_model_params(
    params: dict[str, Any] | None,
    labels: list[str],
) -> tuple[dict[str, Any] | None, dict[str, dict[str, Any]]]:
    if not params:
        return None, {}
    labels_set = set(labels)
    global_params: dict[str, Any] = {}
    model_params: dict[str, dict[str, Any]] = {}
    for name, value in params.items():
        if name in labels_set and isinstance(value, dict):
            model_params[name] = value
        else:
            global_params[name] = value
    return global_params or None, model_params


def resolve_model_params(
    labels: list[str],
    params: dict[str, dict[str, Any]] | None = None,
    yaml_defs: dict[str, list[dict]] | None = None,
) -> dict[str, dict[str, Any]]:
    import streamlit as st

    params = params or {}
    yaml_defs = yaml_defs or {}
    active = [label for label in labels if label in params or label in yaml_defs]
    if not active:
        return {}

    resolved: dict[str, dict[str, Any]] = {}
    with st.sidebar.expander("Model parameters", expanded=False):
        for label in active:
            st.markdown(f"**{label}**")
            resolved[label] = resolve_params(
                params.get(label),
                yaml_defs.get(label),
                key_prefix=f"model.{label}",
                container=st,
            )
    return resolved


def _param_key(name: str, key_prefix: str = "") -> str:
    return param_key(f"{key_prefix}.{name}" if key_prefix else name)


def _render_from_yaml(spec: dict, *, key_prefix: str = "", container: Any) -> Any:
    name = spec["name"]
    ptype = spec.get("type", "slider")
    if ptype == "slider":
        return container.slider(
            spec.get("label", name),
            float(spec["min"]),
            float(spec["max"]),
            float(spec.get("default", spec["min"])),
            step=spec.get("step"),
            key=_param_key(name, key_prefix),
        )
    if ptype == "selectbox":
        options = spec["options"]
        default = spec.get("default", options[0])
        index = options.index(default) if default in options else 0
        return container.selectbox(
            spec.get("label", name),
            options,
            index=index,
            key=_param_key(name, key_prefix),
        )
    raise ValueError(f"Unknown param type: {ptype}")


def _render_spec(name: str, spec: ParamSpec, *, key_prefix: str = "", container: Any) -> Any:
    if spec.kind == "slider":
        return container.slider(
            spec.label,
            spec.min_value,
            spec.max_value,
            spec.default,
            step=spec.step,
            key=_param_key(name, key_prefix),
        )
    if spec.kind == "selectbox":
        options = list(spec.options or [])
        index = options.index(spec.default) if spec.default in options else 0
        return container.selectbox(
            spec.label,
            options,
            index=index,
            key=_param_key(name, key_prefix),
        )
    raise ValueError(f"Unknown ParamSpec kind: {spec.kind}")
