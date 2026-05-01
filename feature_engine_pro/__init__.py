"""
Feature Engine Pro — Industry-Grade Automated Feature Engineering & Selection

A deterministic, Scikit-Learn compatible library for automated feature
engineering, mathematical feature selection, and transparent audit reporting.
"""
__version__ = '2.0.0'
__author__ = 'Ayush Pani'

from feature_engine_pro.engine import FeatureEngine
from feature_engine_pro.reporter import Reporter
from feature_engine_pro.evaluator import Evaluator

__all__ = ['FeatureEngine', 'Reporter', 'Evaluator']
