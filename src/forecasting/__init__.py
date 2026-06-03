"""
AlphaForge — Forecasting Laboratory
=====================================
Probabilistic market forecasting modules.

Modules
-------
monte_carlo   Geometric Brownian Motion path simulation
regimes       Gaussian HMM market regime detection
volatility    Rolling / EWMA / GARCH volatility forecasting
confidence    Bootstrap + analytical confidence intervals
validation    Walk-forward forecast validation engine
factors       Factor-based conditional scenario analysis

Design philosophy
-----------------
Every module produces *probability distributions* over future outcomes,
not point predictions.  Uncertainty quantification is the primary goal.
"""
