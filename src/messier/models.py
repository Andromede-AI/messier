import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypedDict
from pydantic import BaseModel, ConfigDict, Field, model_validator


class IRTModelType(str, Enum):
    ONE_PARAM = "1pl"
    TWO_PARAM = "2pl"


@dataclass
class IRTConfig:
    model_type: IRTModelType = IRTModelType.ONE_PARAM
    epochs: int = 1000
    lr: float = 0.01
    seed: int = 42
    def __post_init__(self):
        self.model_type = IRTModelType(self.model_type)


@dataclass
class DatasetBuilderConfig:
    n_workers: int | None = None # uses all available CPUs when None


class ActionSpaceType(str, Enum):
    TEXT = "text" # text returned directly as string
    TOOL_CALLS = "tool_calls" # named functions exposed by an environment
    SHELL = "shell" # terminal commands and filesystem operations
    UI = "ui" # mouse and keyboard actions


class EnvironmentStateType(str, Enum):
    NONE = "none" # no environment state beyond the task inputs
    FILESYSTEM = "filesystem" # files and directories, for example a software repository
    IN_MEMORY = "in_memory" # program objects, for example a simulated database
    LIVE_WEB = "live_web" # data returned by an external website or web service


class EnvironmentStateAccess(str, Enum):
    NONE = "none" # no environment state to access
    READ_ONLY = "read_only" # the agent may inspect but not change the state
    READ_WRITE = "read_write" # the agent may inspect and change the state


# NOTE these benchmark groups were qualitatively defined by us, so the
# boundaries between them are not definitive
class BenchmarkGroup(str, Enum):
    PROGRAMMING = "programming"
    RESEARCH = "research"
    ENTERPRISE = "enterprise"
    GUI = "gui"
    FUNCTION_CALLING = "function_calling"


# NOTE evaluation may inspect a trial turn by turn or only its final output
# or state. Either case may use one or multiple verifiers.
class ScoringRule(str, Enum):
    """
    The rule that maps verifier results to a trial result.
    """

    ALL_PASS = "all_pass"
    THRESHOLD = "threshold"
    DIRECT = "direct"


class RecordType(str, Enum):
    """
    The kind of result stored in one record.
    """

    TRIAL_RESULT = "trial_result"
    VERIFIER_RESULT = "verifier_result"
    SOURCE_SUMMARY = "source_summary"


class ModelReasoningEffort(str, Enum):
    ENABLED = "enabled"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"


class VerifierType(str, Enum):
    """
    The procedure used to produce a verifier result.
    """

    SCRIPT = "script"
    EXACT_MATCH = "exact_match"
    LLM_JUDGE = "llm_judge"
    HUMAN_LABEL = "human_label"


class VerifierMode(str, Enum):
    FINAL = "final"
    SEQUENTIAL = "sequential"


class SourcePlatform(str, Enum):
    GITHUB = "github"
    HUGGINGFACE = "huggingface"


class DataProvider(str, Enum):
    METR = "metr"
    AGENT_PSYCHOMETRICS = "agent_psychometrics"
    BRIDGE = "bridge"
    GENERAL_AGENTBENCH = "general_agentbench"


ALLOW_EXTRA = ConfigDict(extra="allow")


class HumanEffort(BaseModel):
    """
    Human completion time or steps reported by the benchmark source.
    """

    minutes_median: float | None = None
    minutes_low: float | None = None
    minutes_high: float | None = None
    steps: int | None = None


class ActionSpace(BaseModel):
    """
    Actions available to an agent in an environment.
    """

    model_config = ALLOW_EXTRA
    types: list[ActionSpaceType] = Field(min_length=1)
    description: str | None = None # task-specific description of the available actions


class EnvironmentState(BaseModel):
    """
    The environment state from which a task begins.
    """

    model_config = ALLOW_EXTRA
    type: EnvironmentStateType
    access: EnvironmentStateAccess
    ref: str | None = None
    snapshot: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_access(self):
        has_state = self.type != EnvironmentStateType.NONE
        has_access = self.access != EnvironmentStateAccess.NONE
        if has_state != has_access:
            raise ValueError("environment state and access must either both be none or both be specified")

        return self


class Verifier(BaseModel):
    """
    A compact verifier description embedded in a task row.
    """

    model_config = ALLOW_EXTRA
    type: VerifierType
    mode: VerifierMode
    role: str | None = None # verifier purpose defined by the source


class EnvironmentMetadata(BaseModel):
    """
    The action space and initial environment state associated with a task.
    """

    model_config = ALLOW_EXTRA
    action_space: ActionSpace
    environment_state: EnvironmentState
    url: str | None = None


class InputFile(TypedDict):
    """
    A file available to the agent when a task begins.
    """

    __pydantic_config__ = ALLOW_EXTRA
    path: str
    dataset_path: str


class ExtraContext(TypedDict, total=False):
    """
    Information available to the agent outside the initial task description.

    Benchmark-specific fields are retained alongside the common fields below.
    """

    __pydantic_config__ = ALLOW_EXTRA
    input_files: list[InputFile]
    starter_code: str
    system_prompt: str | None


class TaskMetadata(TypedDict, total=False):
    """
    Source-specific task information with a shared extra-context field.
    """

    __pydantic_config__ = ALLOW_EXTRA
    extra_context: ExtraContext | None


class Task(BaseModel):
    """
    One task and its derived summaries in ``tasks.jsonl``.
    """

    model_config = ConfigDict(extra="forbid")

    benchmark: str
    benchmark_group: BenchmarkGroup | None = None
    task_id: str
    environment_id: str | None = None # stable identifier for the environment
    environment_description: str | None = None # stable description for an environment_id
    environment_metadata: EnvironmentMetadata

    task_description: str | None = None
    task_metadata: TaskMetadata = Field(default_factory=dict)
    gold_answer: Any = None
    task_date: str | None = None
    verifiers: list[Verifier] = Field(default_factory=list)

    human: HumanEffort | None = None # present where the source reports human effort

    difficulty_label: str | None = None
    source_platform: SourcePlatform
    data_provider: DataProvider | None = None

    soc_code: str | None = None # minor group from the SOC taxonomy, for example "15-1200"
    naics_code: str | None = None # sector from the NAICS taxonomy, for example "54"

    n_records: int = 0 # populated by DatasetBuilder
    n_verifiers: int = 0 # populated by DatasetBuilder
    scoring_rule: ScoringRule | None = None
    mean_trial_result: float | None = None
    is_saturated: bool = False
    is_unrecorded: bool = False

class VerifierDefinition(BaseModel):
    """
    One verifier row in ``verifiers.jsonl``.
    """

    model_config = ConfigDict(extra="forbid")

    benchmark: str
    task_id: str
    verifier_id: str
    description: str | None = None
    verifier: Verifier
    metadata: dict[str, Any] = Field(default_factory=dict)
    gold_answer: Any = None
    source_platform: SourcePlatform
    data_provider: DataProvider | None = None


class Record(BaseModel):
    """
    One trial result, verifier result, or source summary in ``records.jsonl``.
    """

    model_config = ConfigDict(extra="forbid")

    benchmark: str
    benchmark_group: BenchmarkGroup | None = None
    task_id: str
    verifier_id: str | None = None
    record_type: RecordType = RecordType.TRIAL_RESULT

    agent_id: str # model configuration and scaffold, when available
    agent_model: str
    model_reasoning_effort: ModelReasoningEffort | None = None
    agent_scaffold: str | None = None
    model_date: str | None = None
    trial: int | None = 0

    result: float | int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_platform: SourcePlatform
    data_provider: DataProvider | None = None

    @model_validator(mode="after")
    def validate_result(self):
        if self.record_type == RecordType.VERIFIER_RESULT and self.verifier_id is None:
            raise ValueError("verifier-result records require verifier_id")
        if self.record_type != RecordType.VERIFIER_RESULT and self.verifier_id is not None:
            raise ValueError("verifier_id is only valid for verifier-result records")
        if self.record_type == RecordType.TRIAL_RESULT and self.result not in {None, 0, 1}:
            raise ValueError("trial results must be 0, 1, or null")
        if self.record_type == RecordType.SOURCE_SUMMARY and self.trial is not None:
            raise ValueError("source summaries do not represent individual trials")
        if self.record_type != RecordType.SOURCE_SUMMARY and self.trial is None:
            raise ValueError("trial and verifier results require a trial index")
        if self.trial is not None and self.trial < 0:
            raise ValueError("trial indices cannot be negative")

        if self.result is not None and not math.isfinite(self.result):
            raise ValueError("results must be finite")

        error_type = self.metadata.get("error_type")
        is_non_scorable_error = error_type in {
            "environment_error",
            "infrastructure_error",
            "evaluator_error",
            "missing_score",
            "unknown_error",
        }
        if is_non_scorable_error and self.result is not None:
            raise ValueError("non-scorable errors cannot have a result")

        return self


class Trajectory(BaseModel):
    """
    One trajectory in ``trajectories.jsonl``.
    """

    model_config = ConfigDict(extra="forbid")

    benchmark: str
    task_id: str
    agent_id: str
    agent_model: str
    model_reasoning_effort: ModelReasoningEffort | None = None
    agent_scaffold: str | None = None
    model_date: str | None = None
    trial: int = Field(ge=0)
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_platform: SourcePlatform
    data_provider: DataProvider | None = None


BuildResult = tuple[
    list[Task],
    list[VerifierDefinition],
    list[Record],
]
