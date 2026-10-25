"""
src/panel_registry.py
=============================================================================
Unified Clinical Panel Registry and Invariant Interface.

Provides an extensible abstraction for laboratory panels:
  - Required fields and physiological bounds
  - Invariant evaluation with panel-specific tolerances
  - Closed-form counterfactual algebraic inversions (K=1)
  - Clinical abstention rules and advisory flags
"""

from typing import Protocol, runtime_checkable, Any, Callable
import logging

logger = logging.getLogger("BioTrace.PanelRegistry")


@runtime_checkable
class Panel(Protocol):
    """Protocol defining the standard interface for a clinical laboratory panel."""

    name: str
    required_fields: list[str]
    tolerance: float
    abstention_rules: list[str]

    def verify_bounds(self, values: dict[str, float]) -> tuple[bool, str]:
        """Validates physiological bounds for the panel analytes."""
        ...

    def identity_fn(self, values: dict[str, float]) -> float:
        """
        Computes the clinical panel identity residual: |observed - calculated|.
        Returns raw absolute residual.
        """
        ...

    def evaluate_invariant(self, values: dict[str, float], tolerance: float | None = None) -> tuple[str, float, float]:
        """
        Evaluates the primary mathematical equality identity for the panel.
        Returns:
            (status, absolute_residual, relative_error)
            status in {"CONVERGED", "VIOLATION", "ABSTAIN_...", "ADVISORY_..."}
        """
        ...

    def solve_inversion(self, values: dict[str, float], target_field: str, **kwargs) -> float | None:
        """Computes the closed-form K=1 counterfactual algebraic inversion for target_field."""
        ...


class LipidPanel:
    """
    Standard Lipid Profile Panel.
    Routes between Friedewald, Sampson (NIH 2020), and Martin-Hopkins equations.
    """

    name: str = "lipid"
    required_fields: list[str] = ["total_cholesterol", "hdl", "ldl", "triglycerides"]
    tolerance: float = 0.10
    abstention_rules: list[str] = [
        "triglycerides > 800.0 (chylomicronemia / severe hypertriglyceridemia)",
        "non-positive analyte values",
        "subfractions exceed total cholesterol (HDL + LDL >= TC)",
    ]

    def verify_bounds(self, values: dict[str, float]) -> tuple[bool, str]:
        from src.stage2_constraints import verify_physiological_bounds
        return verify_physiological_bounds(values)

    def identity_fn(self, values: dict[str, float], active_equation: str | None = None) -> float:
        """
        Computes the active lipid residual |observed - calculated| across Friedewald, Sampson, or Martin-Hopkins.
        """
        tc = float(values.get("total_cholesterol", 0.0))
        hdl = float(values.get("hdl", 0.0))
        ldl = float(values.get("ldl", 0.0))
        tg = float(values.get("triglycerides", 0.0))

        route = active_equation
        if route is None:
            if tg > 400.0:
                route = "SAMPSON"
            else:
                route = "FRIEDEWALD"

        if route == "SAMPSON":
            from src.stage2_constraints import calculate_sampson_ldl
            calc_ldl = calculate_sampson_ldl(tc, hdl, tg)
            return abs(ldl - calc_ldl)
        elif route == "MARTIN_HOPKINS":
            from src.stage2_constraints import calculate_martin_hopkins_ldl
            calc_ldl = calculate_martin_hopkins_ldl(tc, hdl, tg)
            return abs(ldl - calc_ldl)
        else:
            calc_tc = hdl + ldl + (tg / 5.0)
            return abs(tc - calc_tc)

    def evaluate_invariant(self, values: dict[str, float], tolerance: float | None = None, active_equation: str | None = None) -> tuple[str, float, float]:
        tol = tolerance if tolerance is not None else self.tolerance
        tg = float(values.get("triglycerides", 0.0))

        if tg > 800.0:
            return "ABSTAIN_TG800", 1.0, 1.0

        if active_equation == "MARTIN_HOPKINS":
            from src.stage2_constraints import evaluate_martin_hopkins_invariant
            status, rel_err = evaluate_martin_hopkins_invariant(values, tol)
            res = self.identity_fn(values, active_equation="MARTIN_HOPKINS")
            return status, res, rel_err

        if active_equation == "SAMPSON" or (active_equation is None and tg > 400.0):
            from src.stage2_constraints import evaluate_sampson_invariant
            status, rel_err = evaluate_sampson_invariant(values, tol)
            res = self.identity_fn(values, active_equation="SAMPSON")
            return status, res, rel_err

        from src.stage2_constraints import evaluate_friedewald_invariant
        return evaluate_friedewald_invariant(values, tol)

    def solve_inversion(self, values: dict[str, float], target_field: str, active_equation: str = "FRIEDEWALD", **kwargs) -> float | None:
        from src.stage2_isolation import solve_field_closed_form
        return solve_field_closed_form(values, target_field, active_equation=active_equation)


class PanelRegistry:
    """Global registry mapping panel names to clinical Panel implementations."""

    _registry: dict[str, Panel] = {}

    @classmethod
    def register(cls, panel: Panel) -> None:
        cls._registry[panel.name.lower()] = panel
        logger.debug(f"Registered clinical panel: {panel.name}")

    @classmethod
    def get(cls, name: str) -> Panel:
        name_clean = name.lower().strip()
        if name_clean not in cls._registry:
            raise KeyError(f"Unknown clinical panel: '{name}'. Available: {list(cls._registry.keys())}")
        return cls._registry[name_clean]

    @classmethod
    def list_panels(cls) -> list[str]:
        return list(cls._registry.keys())

    @classmethod
    def has(cls, name: str) -> bool:
        return name.lower().strip() in cls._registry


# Pre-register default Lipid panel
PanelRegistry.register(LipidPanel())
