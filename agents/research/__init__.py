"""
Research components for the Blog Accelerator Agent.

This package contains modules for different types of research analysis:
- Industry Analysis
- Solution Analysis
- Paradigm Analysis
- Audience Analysis
- Analogy Generation
- Visual Asset Collection
"""

from agents.research.industry_analysis import IndustryAnalyzer
from agents.research.solution_analysis import SolutionAnalyzer
from agents.research.paradigm_analysis import ParadigmAnalyzer
from agents.research.audience_analysis import AudienceAnalyzer
from agents.research.analogy_generator import AnalogyGenerator
from agents.research.visual_asset_collector import VisualAssetCollector

__all__ = [
    'IndustryAnalyzer',
    'SolutionAnalyzer',
    'ParadigmAnalyzer',
    'AudienceAnalyzer',
    'AnalogyGenerator',
    'VisualAssetCollector'
]
