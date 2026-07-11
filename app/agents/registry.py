"""Agent registry: lists all agents by name and constructs them.

The framework hands each agent ONLY the board in `select_move` -- agents get no
solver handle and no access to the sealed label sets.
"""
from __future__ import annotations

from app.agents.base import Agent
from app.agents.baselines import (
    RandomAgent, HeuristicAgent, MinimaxAgent, PerfectAgent,
)

# name -> zero-arg constructor
_FACTORIES: dict[str, callable] = {
    "random": RandomAgent,
    "heuristic": HeuristicAgent,
    "minimax-2": lambda: MinimaxAgent(2),
    "minimax-4": lambda: MinimaxAgent(4),
    "perfect": PerfectAgent,
}


def _net_factory():
    from app.agents.net import NetAgent
    return NetAgent()


def _net1_factory():
    from app.agents.net1 import Net1Agent
    return Net1Agent()


def _net2_factory():
    from app.agents.net2 import Net2Agent
    return Net2Agent()


def _net0_factory():
    from app.agents.net0 import Net0Agent
    return Net0Agent()


def _net0d_factory():
    from app.agents.net0d import Net0dAgent
    return Net0dAgent()


def _net4_factory():
    from app.agents.net4 import Net4Agent
    return Net4Agent()


def _net0b_factory():
    from app.agents.net0b import Net0bAgent
    return Net0bAgent()


def _net5_factory():
    from app.agents.net5 import Net5Agent
    return Net5Agent()


def _net13_factory():
    from app.agents.net13 import Net13Agent
    return Net13Agent()


def _net14_factory():
    from app.agents.net14 import Net14Agent
    return Net14Agent()


def _net15_factory():
    from app.agents.net15 import Net15Agent
    return Net15Agent()


def _net15s_factory():
    from app.agents.net15s import Net15SAgent
    return Net15SAgent()


def _net16_factory():
    from app.agents.net16 import Net16Agent
    return Net16Agent()


def _net16s_factory():
    from app.agents.net16s import Net16SAgent
    return Net16SAgent()


def _net16b_factory():
    from app.agents.net16b import Net16BAgent
    return Net16BAgent()


# learned agents are registered only if their artifact exists (so the module
# imports cleanly before the first training run).
def _has_net() -> bool:
    import os
    from app.agents.net import DEFAULT_ARTIFACT
    return os.path.exists(DEFAULT_ARTIFACT)


def _has_net1() -> bool:
    import os
    from app.agents.net1 import DEFAULT_ARTIFACT
    return os.path.exists(DEFAULT_ARTIFACT)


def _has_net2() -> bool:
    import os
    from app.agents.net1 import DEFAULT_ARTIFACT   # net2 reuses net1's artifact
    return os.path.exists(DEFAULT_ARTIFACT)


def _has_net0() -> bool:
    import os
    from app.agents.net0 import DEFAULT_ARTIFACT
    return os.path.exists(DEFAULT_ARTIFACT)


def _has_net0d() -> bool:
    import os
    from app.agents.net0 import DEFAULT_ARTIFACT   # net0d reuses net0's artifact
    return os.path.exists(DEFAULT_ARTIFACT)


def _has_net4() -> bool:
    import os
    from app.agents.net1 import DEFAULT_ARTIFACT   # net4 (default) reuses net1's artifact
    return os.path.exists(DEFAULT_ARTIFACT)


def _has_net0b() -> bool:
    import os
    from app.agents.net0 import DEFAULT_ARTIFACT   # net0b reuses net0's artifact
    return os.path.exists(DEFAULT_ARTIFACT)


def _has_net5() -> bool:
    import os
    from app.agents.net5 import DEFAULT_ARTIFACT
    return os.path.exists(DEFAULT_ARTIFACT)


def _has_net13() -> bool:
    import os
    from app.agents.net1 import DEFAULT_ARTIFACT   # net13 (default) reuses net1's artifact
    return os.path.exists(DEFAULT_ARTIFACT)


def _has_net15() -> bool:
    import os
    from app.agents.net15 import DEFAULT_ARTIFACT
    return os.path.exists(DEFAULT_ARTIFACT)


def _has_net16() -> bool:
    import os
    from app.agents.net16 import DEFAULT_ARTIFACT
    return os.path.exists(DEFAULT_ARTIFACT)


def agent_names(include_net: bool = True) -> list[str]:
    names = list(_FACTORIES.keys())
    if include_net and _has_net():
        names.append("neurofour-net")
    if include_net and _has_net1():
        names.append("neurofour-net1")
    if include_net and _has_net2():
        names.append("neurofour-net2")
    if include_net and _has_net0():
        names.append("neurofour-net0")
    if include_net and _has_net0d():
        names.append("neurofour-net0d")
    if include_net and _has_net4():
        names.append("neurofour-net4")
    if include_net and _has_net0b():
        names.append("neurofour-net0b")
    if include_net and _has_net5():
        names.append("neurofour-net5")
    if include_net and _has_net13():
        names.append("neurofour-net13")
    # gen-9 T3: neurofour-net14 REGISTERED, overriding gen-8 T4's "DO NOT
    # REGISTER" verdict -- see net14.py's module docstring "gen-9 T3
    # REGISTRATION" section for the decision-rule re-read (clause (b) does
    # not apply to a Pareto-non-dominated, sub-HEADLINE agent) and the
    # independently-verified sec.7 dominance table. Zero-byte, zero-artifact
    # agent: unconditional, no `_has_net14()` artifact gate needed (there is
    # no artifact to gate on).
    if include_net:
        names.append("neurofour-net14")
    if include_net and _has_net15():
        names.append("neurofour-net15")
        names.append("neurofour-net15s")   # shares net15's artifact, gated the same way
    if include_net and _has_net16():
        names.append("neurofour-net16")
        names.append("neurofour-net16s")   # shares net16's artifact, gated the same way
        names.append("neurofour-net16b")   # shares net16's artifact, gated the same way
    return names


def make_agent(name: str) -> Agent:
    if name == "neurofour-net":
        return _net_factory()
    if name == "neurofour-net1":
        return _net1_factory()
    if name == "neurofour-net2":
        return _net2_factory()
    if name == "neurofour-net0":
        return _net0_factory()
    if name == "neurofour-net0d":
        return _net0d_factory()
    if name == "neurofour-net4":
        return _net4_factory()
    if name == "neurofour-net0b":
        return _net0b_factory()
    if name == "neurofour-net5":
        return _net5_factory()
    if name == "neurofour-net13":
        return _net13_factory()
    if name == "neurofour-net14":
        return _net14_factory()
    if name == "neurofour-net15":
        return _net15_factory()
    if name == "neurofour-net15s":
        return _net15s_factory()
    if name == "neurofour-net16":
        return _net16_factory()
    if name == "neurofour-net16s":
        return _net16s_factory()
    if name == "neurofour-net16b":
        return _net16b_factory()
    if name not in _FACTORIES:
        raise KeyError(f"unknown agent: {name}")
    return _FACTORIES[name]()


def all_agents(include_net: bool = True) -> list[Agent]:
    return [make_agent(n) for n in agent_names(include_net)]
