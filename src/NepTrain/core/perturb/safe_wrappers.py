"""Error-handling wrappers for perturbation generators.

Keep the orchestration-heavy ``run.py`` focused on sampling/flow control.  This
module centralizes the repetitive try/except boilerplate that turns low-level
implementation errors into user-facing ``PerturbationError`` messages.
"""

from ase import Atoms

from NepTrain.exceptions import PerturbationError

from .amorphous import generate_amorphous
from .dislocation import generate_dislocation
from .grain_boundary import generate_grain_boundary
from .magnetic import apply_magnetic_perturbation
from .stacking_fault import generate_stacking_fault
from .surface import generate_surface
from .twinning import generate_twinning


def _safe_generate_surface(atoms: Atoms, **kwargs) -> Atoms:
    """Safely generate a surface with contextual error reporting."""
    try:
        return generate_surface(atoms, **kwargs)
    except Exception as e:
        raise PerturbationError(
            f"Surface generation failed: {e}",
            operation="surface",
            suggestion="Check surface indices and vacuum parameters",
        ) from e


def _safe_generate_grain_boundary(atoms: Atoms, **kwargs) -> Atoms:
    """Safely generate a grain boundary with contextual error reporting."""
    try:
        return generate_grain_boundary(atoms, **kwargs)
    except Exception as e:
        raise PerturbationError(
            f"Grain boundary generation failed: {e}",
            operation="grain_boundary",
            suggestion="Check axis, angle, and overlap distance parameters",
        ) from e


def _safe_generate_dislocation(atoms: Atoms, **kwargs) -> Atoms:
    """Safely generate a dislocation with contextual error reporting."""
    try:
        return generate_dislocation(atoms, **kwargs)
    except Exception as e:
        raise PerturbationError(
            f"Dislocation generation failed: {e}",
            operation="dislocation",
            suggestion="Check axis, burgers vector, and center parameters",
        ) from e


def _safe_generate_twinning(atoms: Atoms, **kwargs) -> Atoms:
    """Safely generate a twin with contextual error reporting."""
    try:
        return generate_twinning(atoms, **kwargs)
    except Exception as e:
        raise PerturbationError(
            f"Twinning generation failed: {e}",
            operation="twinning",
            suggestion="Check Miller indices and minimum distance parameters",
        ) from e


def _safe_generate_stacking_fault(atoms: Atoms, **kwargs) -> Atoms:
    """Safely generate a stacking fault with contextual error reporting."""
    try:
        return generate_stacking_fault(atoms, **kwargs)
    except Exception as e:
        raise PerturbationError(
            f"Stacking fault generation failed: {e}",
            operation="stacking_fault",
            suggestion="Check plane normal and shift vector parameters",
        ) from e


def _safe_generate_amorphous(atoms: Atoms, **kwargs) -> Atoms:
    """Safely generate an amorphous structure with contextual error reporting."""
    try:
        return generate_amorphous(atoms, **kwargs)
    except Exception as e:
        raise PerturbationError(
            f"Amorphous generation failed: {e}",
            operation="amorphous",
            suggestion="Check minimum distance and rattle strength parameters",
        ) from e


def _safe_apply_magnetic_perturbation(atoms: Atoms, **kwargs) -> Atoms:
    """Safely apply magnetic perturbation with contextual error reporting.

    Preserve the raw ``ValueError`` for Sobol-random-value accounting so tests
    and callers can distinguish dimension mismatches from generic user input
    errors.
    """
    try:
        return apply_magnetic_perturbation(atoms, **kwargs)
    except ValueError as e:
        if "Not enough random values" in str(e):
            raise
        raise PerturbationError(
            f"Magnetic perturbation failed: {e}",
            operation="magnetic",
            suggestion="Check magnetic mode and configuration parameters",
        ) from e
