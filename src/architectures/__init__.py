from .dram import DRAMGenerator
from .finfet import FinFETGenerator

ARCHITECTURE_MAP = {
    'DRAM': DRAMGenerator,
    'FinFET': FinFETGenerator,
}

__all__ = ['DRAMGenerator', 'FinFETGenerator', 'ARCHITECTURE_MAP']
