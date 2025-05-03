"""
Research components for the Blog Accelerator Agent.

This module contains modular research components for each subtopic:
- Industry/System Analysis
- Proposed Solution Analysis
- Current Paradigm Analysis
- Audience Analysis
"""

from agents.research.industry_analysis import IndustryAnalyzer
from agents.research.solution_analysis import SolutionAnalyzer
from agents.research.paradigm_analysis import ParadigmAnalyzer
from agents.research.audience_analysis import AudienceAnalyzer
from agents.research.analogy_generator import AnalogyGenerator

__all__ = [
    'IndustryAnalyzer',
    'SolutionAnalyzer',
    'ParadigmAnalyzer',
    'AudienceAnalyzer',
    'AnalogyGenerator'
] 