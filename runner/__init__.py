"""Runner package: SQLite persistence and the HF Inference execution wrapper.

Day 1 vertical slice — one model x one prompt, end to end, against the
Hugging Face Inference Providers serverless API.
"""

__all__ = ["db", "executor"]
