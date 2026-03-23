"""
agents/reasoning_agent.py - Performs step-by-step clinical reasoning using LLMs.

This is the core diagnostic reasoning engine that:
  1. Takes symptoms + RAG context as input
  2. Applies chain-of-thought reasoning
  3. Returns possible conditions with confidence levels
  4. Generates reasoning steps for explainability
"""

import json
import re
from typing import List, Tuple

