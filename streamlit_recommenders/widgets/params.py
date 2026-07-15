"""Declarative parameter specs and helpers for rendering sidebar widgets."""

from dataclasses import dataclass
from typing import Any, Literal

from streamlit_recommenders.runtime.keys import param_key


@dataclass(frozen=True)
class ParamSpec:
    """Declarative spec for a single tunable parameter widget.

    Attributes:
        kind: Widget type, either ``"slider"`` or ``"selectbox"``.
        label: Human-readable widget label.
        default: Default (and slider) value.
        min_value: Slider lower bound.
        max_value: Slider upper bound.
        step: Slider step size.
        options: Selectbox choices.
    """

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
    """Build a slider :class:`ParamSpec`.

    Args:
        label: Widget label.
        min_value: Slider lower bound.
        max_value: Slider upper bound.
        value: Default value.
        step: Slider step size.

    Returns:
        A slider ``ParamSpec``.
    """
    return ParamSpec(
        kind="slider",
        label=label,
        default=value,
        min_value=min_value,
        max_value=max_value,
        step=step,
    )


def selectbox(label: str, options: list[Any], index: int = 0) -> ParamSpec:
    """Build a selectbox :class:`ParamSpec`.

    Args:
        label: Widget label.
        options: Selectable choices.
        index: Index of the default choice; ``None`` default when empty.

    Returns:
        A selectbox ``ParamSpec``.
    """
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
    """Render sidebar widgets and return resolved param values.

    YAML defs are rendered first, then any ``ParamSpec`` entries in ``params``;
    plain (non-spec) values in ``params`` pass through unchanged and take
    precedence over a matching YAML def.

    Args:
        params: Mapping of param name to a ``ParamSpec`` or a literal value.
        yaml_defs: List of YAML-derived widget definitions.
        key_prefix: Prefix used to namespace widget state keys.
        container: Streamlit container to render into; defaults to the sidebar.

    Returns:
        Mapping of param name to its resolved value.
    """
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
    """Split params into global params and per-model param dicts.

    An entry whose name matches a model label and whose value is a dict is
    treated as that model's params; everything else is global.

    Args:
        params: Combined mapping of global and per-model params.
        labels: Known model labels.

    Returns:
        A ``(global_params, model_params)`` tuple; ``global_params`` is ``None``
        when empty and ``model_params`` maps label to its param dict.
    """
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
    """Render a per-model parameter expander and return resolved values.

    Only labels present in ``params`` or ``yaml_defs`` are rendered.

    Args:
        labels: Model labels to consider, in display order.
        params: Optional per-model mapping of param specs/values.
        yaml_defs: Optional per-model list of YAML widget definitions.

    Returns:
        Mapping of model label to its resolved param values; empty when no
        label has any params.
    """
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
    """Build a namespaced Streamlit widget state key for ``name``."""
    return param_key(f"{key_prefix}.{name}" if key_prefix else name)


def _render_from_yaml(spec: dict, *, key_prefix: str = "", container: Any) -> Any:
    """Render a widget from a YAML spec dict and return its value."""
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
    """Render a widget from a :class:`ParamSpec` and return its value."""
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
