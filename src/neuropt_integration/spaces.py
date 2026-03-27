"""Translate YAML search-space definitions to neuropt objects."""


def build_search_space(space_cfg: dict) -> dict:
    from neuropt import LogUniform, Categorical

    result = {}
    for key, spec in space_cfg.items():
        if spec["type"] == "log_uniform":
            result[key] = LogUniform(spec["low"], spec["high"])
        elif spec["type"] == "uniform":
            result[key] = (spec["low"], spec["high"])
        elif spec["type"] == "categorical":
            result[key] = Categorical(spec["values"])
    return result
