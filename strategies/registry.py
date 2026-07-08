"""Decorator-based registration for pFL methods, attacks and defenses (Layer 2).

Strategies register themselves at import time::

    @register_pfl("fedbn")
    class FedBN(PFLMethod): ...

and ``experiments`` looks them up by the string names in a config, so that any
``(fl_method, partition, attack, defense)`` combination is dispatchable without
bespoke glue code.
"""

from strategies.base import AttackHook, DefenseHook, PFLMethod

_PFL_METHODS: dict = {}
_ATTACKS: dict = {}
_DEFENSES: dict = {}


def _make_register(table, base_cls, kind):
    def register(name):
        def deco(cls):
            if not issubclass(cls, base_cls):
                raise TypeError(f"{cls.__name__} must subclass {base_cls.__name__}")
            if name in table:
                raise KeyError(f"{kind} {name!r} already registered")
            cls.name = name
            table[name] = cls
            return cls
        return deco
    return register


register_pfl = _make_register(_PFL_METHODS, PFLMethod, "pFL method")
register_attack = _make_register(_ATTACKS, AttackHook, "attack")
register_defense = _make_register(_DEFENSES, DefenseHook, "defense")


def get_pfl(name):
    if name not in _PFL_METHODS:
        raise KeyError(f"unknown pFL method {name!r}; registered: {sorted(_PFL_METHODS)}")
    return _PFL_METHODS[name]


def get_attack(name):
    if name not in _ATTACKS:
        raise KeyError(f"unknown attack {name!r}; registered: {sorted(_ATTACKS)}")
    return _ATTACKS[name]


def get_defense(name):
    if name not in _DEFENSES:
        raise KeyError(f"unknown defense {name!r}; registered: {sorted(_DEFENSES)}")
    return _DEFENSES[name]
