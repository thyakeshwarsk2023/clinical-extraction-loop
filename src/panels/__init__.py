"""
src/panels
=============================================================================
Package defining clinically-validated laboratory panels.
"""

from src.panels.protein import ProteinPanel
from src.panels.electrolytes import ElectrolytePanel

__all__ = ["ProteinPanel", "ElectrolytePanel"]
