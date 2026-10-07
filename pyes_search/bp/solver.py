"""Loopy min-sum belief propagation over small factor graphs, with exact and
ordered-statistics (OSD) candidate search."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass



@dataclass(frozen=True)
class HyperFactor:
    target: int
    scope: tuple[int, ...]
    assignments: tuple[tuple[int, ...], ...]
    energies: tuple[float, ...]

    def __post_init__(self):
        if not self.scope or self.scope[0] != self.target:
            raise ValueError("hyperfactor scope must begin with its target")
        if len(set(self.scope)) != len(self.scope):
            raise ValueError("hyperfactor variables must be unique")
        if len(self.assignments) != len(self.energies) or not self.assignments:
            raise ValueError("hyperfactor assignments and energies must align")


@dataclass(frozen=True)
class BPResult:
    block: tuple[int, ...]
    beliefs: tuple[tuple[float, ...], ...]
    iterations: int
    residual: float
    energy: float


@dataclass(frozen=True)
class ExactFactorMapResult:
    block: tuple[int, ...]
    energy: float
    assignments: int


@dataclass(frozen=True)
class OSDResult:
    blocks: tuple[tuple[int, ...], ...]
    reliabilities: tuple[float, ...]
    scopes: tuple[tuple[int, ...], ...]
    enumerated: int


def _normalize(values) -> tuple[float, ...]:
    floor = min(values)
    return tuple(float(value - floor) for value in values)


def _soft_min(values, temperature: float) -> float:
    floor = min(values)
    if temperature == 0:
        return float(floor)
    return float(
        floor
        - temperature
        * math.log(sum(math.exp(-(value - floor) / temperature) for value in values))
    )


def _factor_value(factor, local) -> float:
    try:
        index = factor.assignments.index(local)
    except ValueError as exc:
        raise ValueError("block value is absent from a hyperfactor table") from exc
    return factor.energies[index]


def factor_energy(block, factors) -> float:
    total = 0.0
    for factor in factors:
        local = tuple(block[variable] for variable in factor.scope)
        total += _factor_value(factor, local)
    return total


def exact_factor_map(domains, factors, *, max_assignments: int = 65_536):
    """Return the exact MAP of an already-scored factor model when it is small.

    This performs no model calls.  It is both a quality path for compact domains
    and an oracle for checking whether loopy BP found the factor-model MAP.
    """

    domains, factors = _validate(domains, factors)
    assignments = math.prod(len(domain) for domain in domains)
    if assignments > max_assignments:
        return None
    lookups = tuple(dict(zip(factor.assignments, factor.energies, strict=True)) for factor in factors)

    def energy(block):
        return sum(
            lookup[tuple(block[variable] for variable in factor.scope)]
            for factor, lookup in zip(factors, lookups, strict=True)
        )

    block = min(itertools.product(*domains), key=lambda candidate: (energy(candidate), candidate))
    return ExactFactorMapResult(tuple(block), energy(block), assignments)


def ordered_statistics_candidates(
    domains,
    factors,
    beliefs,
    *,
    variable_limit: int = 3,
    assignment_cap: int = 256,
    shortlist: int = 4,
    seed_count: int = 2,
) -> OSDResult:
    """Enumerate bounded least-reliable BP completions without model calls."""

    domains, factors = _validate(domains, factors)
    beliefs = tuple(tuple(float(value) for value in row) for row in beliefs)
    if len(beliefs) != len(domains) or any(
        len(row) != len(domain) or any(not math.isfinite(value) for value in row)
        for row, domain in zip(beliefs, domains, strict=True)
    ):
        raise ValueError("OSD beliefs must align with BP domains")
    if min(variable_limit, assignment_cap, shortlist, seed_count) < 1:
        raise ValueError("OSD limits must be positive")

    base = tuple(
        min(domain, key=beliefs[index].__getitem__)
        for index, domain in enumerate(domains)
    )
    reliabilities = tuple(
        sorted(row)[1] - sorted(row)[0] if len(row) > 1 else math.inf
        for row in beliefs
    )
    uncertain = tuple(sorted(range(len(domains)), key=lambda node: (reliabilities[node], node)))
    adjacency = [set() for _ in domains]
    for factor in factors:
        for left, right in itertools.combinations(factor.scope, 2):
            adjacency[left].add(right)
            adjacency[right].add(left)

    def grow(seed, *, connected):
        scope = [seed]
        size = len(domains[seed])
        while len(scope) < variable_limit:
            candidates = [
                node
                for node in uncertain
                if node not in scope
                and (not connected or any(node in adjacency[member] for member in scope))
                and size * len(domains[node]) <= assignment_cap
            ]
            if not candidates:
                break
            selected = candidates[0]
            scope.append(selected)
            size *= len(domains[selected])
        return tuple(sorted(scope))

    scopes = []
    # Classic OSD: jointly enumerate the globally least reliable variables.
    scopes.append(grow(uncertain[0], connected=False))
    # Factor-aware OSD: also enumerate connected uncertainty neighborhoods.
    for seed in uncertain[:seed_count]:
        scope = grow(seed, connected=True)
        if scope not in scopes:
            scopes.append(scope)

    pool = {base}
    enumerated = 0
    for scope in scopes:
        combinations = math.prod(len(domains[node]) for node in scope)
        if enumerated + combinations > assignment_cap:
            continue
        enumerated += combinations
        for values in itertools.product(*(domains[node] for node in scope)):
            block = list(base)
            for node, value in zip(scope, values, strict=True):
                block[node] = value
            pool.add(tuple(block))

    # First-order leftovers mirror low-order OSD error patterns and ensure a
    # variable excluded by the connected scopes still receives a chance.
    for node in uncertain:
        for value in domains[node]:
            if value == base[node] or enumerated >= assignment_cap:
                continue
            block = list(base)
            block[node] = value
            pool.add(tuple(block))
            enumerated += 1

    energy_ranked = sorted(pool, key=lambda block: (factor_energy(block, factors), block))

    def belief_cost(block):
        return sum(
            beliefs[node][value] - min(beliefs[node])
            for node, value in enumerate(block)
        )

    belief_ranked = sorted(pool, key=lambda block: (belief_cost(block), block))
    by_order = []
    for order in range(1, variable_limit + 1):
        matches = [
            block
            for block in pool
            if sum(left != right for left, right in zip(block, base, strict=True)) == order
        ]
        if matches:
            by_order.append(min(matches, key=lambda block: (belief_cost(block), block)))

    # Ordered-statistics diversity is intentional: reserve the best pattern at
    # each Hamming order before interleaving belief and factor-energy leaders.
    selected = []
    for block in [energy_ranked[0], base, *by_order]:
        if block not in selected and len(selected) < shortlist:
            selected.append(block)
    for belief_block, energy_block in itertools.zip_longest(
        belief_ranked, energy_ranked
    ):
        for block in (belief_block, energy_block):
            if block is not None and block not in selected:
                selected.append(block)
                if len(selected) == shortlist:
                    break
        if len(selected) == shortlist:
            break
    return OSDResult(tuple(selected), reliabilities, tuple(scopes), enumerated)


def _validate(domains, factors):
    domains = tuple(tuple(domain) for domain in domains)
    if not domains or any(domain != tuple(range(len(domain))) for domain in domains):
        raise ValueError("BP domains must be non-empty contiguous local indices")
    factors = tuple(factors)
    for factor in factors:
        if not 0 <= factor.target < len(domains) or any(
            not 0 <= variable < len(domains) for variable in factor.scope
        ):
            raise ValueError("hyperfactor scope is out of range")
        expected = tuple(
            itertools.product(*(domains[variable] for variable in factor.scope))
        )
        if factor.assignments != expected:
            raise ValueError("hyperfactor table must contain its full Cartesian domain")
        if any(not math.isfinite(value) for value in factor.energies):
            raise ValueError("hyperfactor energies must be finite")
    return domains, factors


def loopy_min_sum(
    domains,
    factors,
    *,
    iterations: int = 32,
    damping: float = 0.25,
    tolerance: float = 1e-8,
    temperature: float = 0.0,
) -> BPResult:
    """Synchronous damped min-sum; ``temperature > 0`` softens the min (soft-min)."""

    domains, factors = _validate(domains, factors)
    if (
        iterations < 1
        or not 0 <= damping < 1
        or tolerance <= 0
        or not math.isfinite(temperature)
        or temperature < 0
    ):
        raise ValueError("invalid BP iteration, damping, tolerance, or temperature")
    incident = [[] for _ in domains]
    for factor_index, factor in enumerate(factors):
        for variable in factor.scope:
            incident[variable].append(factor_index)

    factor_messages = {
        (factor_index, variable): (0.0,) * len(domains[variable])
        for factor_index, factor in enumerate(factors)
        for variable in factor.scope
    }
    residual = 0.0
    used = 0
    stage_temperature = float(temperature)
    for _ in range(iterations):
        variable_messages = {}
        for variable, factor_indices in enumerate(incident):
            for destination in factor_indices:
                variable_messages[(variable, destination)] = tuple(
                    sum(
                        factor_messages[(source, variable)][value]
                        for source in factor_indices
                        if source != destination
                    )
                    for value in domains[variable]
                )

        updated = {}
        residual = 0.0
        for factor_index, factor in enumerate(factors):
            for position, variable in enumerate(factor.scope):
                raw = []
                for value in domains[variable]:
                    choices = []
                    for assignment, energy in zip(
                        factor.assignments, factor.energies, strict=True
                    ):
                        if assignment[position] != value:
                            continue
                        choices.append(
                            energy
                            + sum(
                                variable_messages[(other, factor_index)][
                                    assignment[other_pos]
                                ]
                                for other_pos, other in enumerate(factor.scope)
                                if other != variable
                            )
                        )
                    raw.append(_soft_min(choices, stage_temperature))
                proposed = _normalize(raw)
                previous = factor_messages[(factor_index, variable)]
                damped = _normalize(
                    tuple(
                        damping * old + (1 - damping) * new
                        for old, new in zip(previous, proposed, strict=True)
                    )
                )
                residual = max(
                    residual,
                    max(abs(new - old) for new, old in zip(damped, previous, strict=True)),
                )
                updated[(factor_index, variable)] = damped
        factor_messages = updated
        used += 1
        if residual < tolerance:
            break

    beliefs = tuple(
        _normalize(
            tuple(
                sum(
                    factor_messages[(factor_index, variable)][value]
                    for factor_index in incident[variable]
                )
                for value in domain
            )
        )
        for variable, domain in enumerate(domains)
    )
    block = tuple(
        min(domain, key=beliefs[index].__getitem__)
        for index, domain in enumerate(domains)
    )
    return BPResult(block, beliefs, used, residual, factor_energy(block, factors))
