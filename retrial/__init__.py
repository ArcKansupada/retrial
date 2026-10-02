"""retrial - git for agent trajectories.

Branch, diff, and bisect LLM agent runs, backed by real re-execution instead of
static logs.
"""

from .bisect import bisect
from .diff import diff
from .errors import (
    AmbiguousSha,
    ExportFormatError,
    IntegrationError,
    NotFound,
    ReplayIntegrityError,
    RetrialError,
    SchemaVersionError,
)
from .explore import ablate, sweep
from .fork import fork
from .patch import apply_patch
from .pricing import FREE, cost_of, register_prices, trajectory_cost

# Provider adapters. Each imports its SDK lazily, so none is a hard dependency.
from .providers import ModelResponse, gemini_adapter, openai_adapter, tool_result, tool_uses
from .record import record
from .regress import rerun
from .storage import Store
from .trajectory import trajectory
from .transfer import export, import_

# The shapes the functions above return, re-exported for annotations. See types.py.
from .types import (
    JSON,
    AblateProbe,
    AblateResult,
    Agent,
    BisectProbe,
    BisectResult,
    Check,
    CheckFunction,
    DiffBlock,
    DiffResult,
    Divergence,
    Edit,
    EditProvenance,
    ExportHeader,
    ExportRow,
    ExportSession,
    ExportStep,
    Origin,
    Patch,
    PatchOp,
    RerunOutcome,
    RerunResult,
    RerunVerdict,
    Session,
    SessionStatus,
    Step,
    StepType,
    SweepBoundary,
    SweepProbe,
    SweepResult,
    SweepRun,
    TrajectoryEntry,
)

__version__ = "0.3.0"

__all__ = [
    "record",
    "fork",
    "diff",
    "bisect",
    "ablate",
    "sweep",
    "rerun",
    "cost_of",
    "trajectory_cost",
    "register_prices",
    "FREE",
    "trajectory",
    # providers
    "ModelResponse",
    "openai_adapter",
    "gemini_adapter",
    "tool_result",
    "tool_uses",
    "export",
    "import_",
    "apply_patch",
    "Store",
    "RetrialError",
    "NotFound",
    "AmbiguousSha",
    "ReplayIntegrityError",
    "IntegrationError",
    "SchemaVersionError",
    "ExportFormatError",
    # types
    "JSON",
    "AblateProbe",
    "AblateResult",
    "Agent",
    "BisectProbe",
    "BisectResult",
    "Check",
    "CheckFunction",
    "DiffBlock",
    "DiffResult",
    "Divergence",
    "Edit",
    "EditProvenance",
    "ExportHeader",
    "ExportRow",
    "ExportSession",
    "ExportStep",
    "Origin",
    "Patch",
    "PatchOp",
    "RerunOutcome",
    "RerunResult",
    "RerunVerdict",
    "Session",
    "SessionStatus",
    "Step",
    "StepType",
    "SweepBoundary",
    "SweepProbe",
    "SweepResult",
    "SweepRun",
    "TrajectoryEntry",
    "__version__",
]
