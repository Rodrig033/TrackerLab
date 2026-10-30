__all__ = [
    "generate_eda"
]


def __getattr__(name):
    # Carga perezosa: 'import stats' no importa matplotlib; solo al pedir generate_eda.
    if name == "generate_eda":
        from stats.eda import generate_eda
        return generate_eda
    raise AttributeError(f"module 'stats' has no attribute {name!r}")