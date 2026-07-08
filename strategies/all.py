"""Import side-effect module: importing this registers every built-in strategy.

``experiments`` imports this once so that config strings like ``fl_method: fedrep``
or ``attack: bad_pfl`` resolve through the registry.
"""

# pFL methods
import strategies.pfl.fedbn  # noqa: F401
import strategies.pfl.fedrep  # noqa: F401

# attacks
import strategies.attacks.atk1  # noqa: F401
