"""
oracle.integrations — optional adapters for third-party frameworks.

Each integration is import-guarded: importing oracle.integrations.X
raises a clear ImportError if X's package isn't installed, but never
affects core oracle usage (Tracer, TraceStore, AnomalyDetector, etc.).
"""
