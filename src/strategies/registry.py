"""
Strategy Plugin Registry.

Design
------
A central ``_REGISTRY`` dictionary maps strategy names to their classes.
New strategies register themselves by calling ``register()`` or by using
the ``@strategy_plugin`` decorator.

Auto-discovery scans the ``src.strategies`` package for any
``BaseStrategy`` subclass and registers it automatically — no manual
import lists to maintain.

Usage
-----
Register explicitly:
    from src.strategies.registry import register
    register(MovingAverageCrossover)

Register via decorator:
    @strategy_plugin("My Custom Strategy")
    class MyStrategy(BaseStrategy):
        ...

Build from name + params:
    strategy = build_strategy("Moving Average Crossover", {"short_window": 20, "long_window": 50})

List all available:
    names = list_strategies()
"""

import importlib
import logging
import pkgutil
from typing import Any

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

# ── Registry store ────────────────────────────────────────────────────────────

_REGISTRY: dict[str, type[BaseStrategy]] = {}


# ── Registration API ──────────────────────────────────────────────────────────

def register(cls: type[BaseStrategy], name: str | None = None) -> type[BaseStrategy]:
    """
    Add a strategy class to the registry.

    Parameters
    ----------
    cls : type[BaseStrategy]
        The strategy class to register.
    name : str or None
        Registry key.  Defaults to ``cls.__name__``.

    Returns
    -------
    type[BaseStrategy]
        The class itself (so this can be used as a decorator).
    """
    key = name or cls.__name__
    if key in _REGISTRY and _REGISTRY[key] is not cls:
        logger.warning(
            "Registry: '%s' already registered as %s; overwriting with %s.",
            key, _REGISTRY[key].__name__, cls.__name__,
        )
    _REGISTRY[key] = cls
    logger.debug("Registered strategy: '%s' → %s", key, cls.__name__)
    return cls


def strategy_plugin(name: str | None = None):
    """
    Class decorator that registers a strategy under an optional display name.

    Usage
    -----
    @strategy_plugin("My Breakout Strategy")
    class BreakoutStrategy(BaseStrategy):
        ...
    """
    def decorator(cls: type[BaseStrategy]) -> type[BaseStrategy]:
        return register(cls, name=name)
    return decorator


def unregister(name: str) -> None:
    """Remove a strategy from the registry by name."""
    _REGISTRY.pop(name, None)


# ── Query API ─────────────────────────────────────────────────────────────────

def list_strategies() -> list[str]:
    """Return a sorted list of all registered strategy names."""
    return sorted(_REGISTRY.keys())


def get_strategy_class(name: str) -> type[BaseStrategy]:
    """
    Retrieve a strategy class by its registered name.

    Raises
    ------
    KeyError
        If ``name`` is not in the registry.
    """
    if name not in _REGISTRY:
        available = list_strategies()
        raise KeyError(
            f"Strategy '{name}' not found. "
            f"Available: {available}"
        )
    return _REGISTRY[name]


def build_strategy(name: str, params: dict[str, Any] | None = None) -> BaseStrategy:
    """
    Instantiate a strategy from its registered name and a params dict.

    Parameters
    ----------
    name : str
        Registered strategy name (e.g. "Moving Average Crossover").
    params : dict or None
        Constructor keyword arguments.  Empty dict = use strategy defaults.

    Returns
    -------
    BaseStrategy
    """
    cls = get_strategy_class(name)
    params = params or {}
    try:
        return cls(**params)
    except TypeError as e:
        raise TypeError(
            f"Could not instantiate '{name}' with params={params}: {e}"
        ) from e


def strategy_info(name: str) -> dict:
    """
    Return metadata about a registered strategy.

    Returns
    -------
    dict with keys: name, class_name, module, docstring.
    """
    cls = get_strategy_class(name)
    return {
        "name":       name,
        "class_name": cls.__name__,
        "module":     cls.__module__,
        "docstring":  (cls.__doc__ or "").strip().split("\n")[0],
    }


# ── Auto-discovery ─────────────────────────────────────────────────────────────

def discover_strategies(package: str = "src.strategies") -> int:
    """
    Auto-discover and register all BaseStrategy subclasses in ``package``.

    Walks the package directory, imports every module (excluding __init__
    and base_strategy), and registers any BaseStrategy subclass found.

    Parameters
    ----------
    package : str
        Dotted Python package name.

    Returns
    -------
    int
        Number of strategies registered during this call.
    """
    registered_count = 0

    try:
        pkg = importlib.import_module(package)
    except ImportError as e:
        logger.error("Could not import package '%s': %s", package, e)
        return 0

    for _importer, module_name, _ispkg in pkgutil.iter_modules(pkg.__path__):
        if module_name in ("__init__", "base_strategy", "registry"):
            continue
        full_name = f"{package}.{module_name}"
        try:
            module = importlib.import_module(full_name)
        except ImportError as e:
            logger.warning("Could not import '%s': %s", full_name, e)
            continue

        for attr_name in dir(module):
            obj = getattr(module, attr_name)
            if (
                isinstance(obj, type)
                and issubclass(obj, BaseStrategy)
                and obj is not BaseStrategy
                and obj.__module__ == full_name  # skip re-exports
            ):
                # Use the instance .name attribute as the registry key
                # (fall back to class name if not available as class attr)
                try:
                    # Temporarily instantiate to read name — only valid
                    # for strategies with no required parameters.
                    # For parameterised strategies, use class name.
                    display_name = getattr(obj, "_registry_name", obj.__name__)
                except Exception:
                    display_name = obj.__name__

                if display_name not in _REGISTRY:
                    register(obj, name=display_name)
                    registered_count += 1

    logger.info("discover_strategies: registered %d strategies.", registered_count)
    return registered_count


# ── Seed the registry with all built-in strategies ────────────────────────────

def _bootstrap() -> None:
    """
    Explicitly register all built-in strategies with their canonical names.
    Called once at module import time.
    """
    from .moving_average import MovingAverageCrossover
    from .mean_reversion import MeanReversion
    from .momentum import Momentum
    from .rsi import RSIMeanReversion
    from .bollinger import BollingerBands
    from .buy_hold import BuyAndHold

    _entries = [
        ("Moving Average Crossover", MovingAverageCrossover),
        ("Mean Reversion",           MeanReversion),
        ("Momentum",                 Momentum),
        ("RSI Mean Reversion",       RSIMeanReversion),
        ("Bollinger Bands",          BollingerBands),
        ("Buy & Hold",               BuyAndHold),
    ]
    for name, cls in _entries:
        register(cls, name=name)


_bootstrap()
