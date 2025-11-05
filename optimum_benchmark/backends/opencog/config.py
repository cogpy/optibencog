from dataclasses import dataclass, field
from logging import getLogger
from typing import Any, Dict, List, Optional

from ..config import BackendConfig

LOGGER = getLogger(__name__)


@dataclass
class OpenCogConfig(BackendConfig):
    """Configuration for OpenCog multi-agent orchestration backend.

    OpenCog backend enables autonomous multi-agent orchestration for benchmarking
    across multiple models and backends with cognitive reasoning capabilities.
    """

    name: str = "opencog"
    version: Optional[str] = "1.0.0"
    _target_: str = "optimum_benchmark.backends.opencog.backend.OpenCogBackend"

    # Multi-agent orchestration settings
    num_agents: int = 1
    agent_coordination: str = "sequential"  # sequential, parallel, distributed
    agent_memory_sharing: bool = True

    # Cognitive reasoning settings
    reasoning_engine: str = "pattern_matcher"  # pattern_matcher, pln, moses
    atomspace_size: int = 10000
    enable_attention_allocation: bool = True

    # Agent-specific model settings
    agent_models: Optional[List[str]] = None
    agent_backends: Optional[List[str]] = None
    agent_devices: Optional[List[str]] = None

    # Orchestration workflow
    workflow_strategy: str = "adaptive"  # adaptive, static, dynamic
    enable_self_optimization: bool = True

    # Integration with existing backends
    delegate_backend: str = "pytorch"  # Default backend to delegate actual inference
    delegate_backend_config: Dict[str, Any] = field(default_factory=dict)

    # Performance tuning
    communication_protocol: str = "shared_memory"  # shared_memory, message_passing
    load_balancing: str = "round_robin"  # round_robin, least_loaded, cognitive

    def __post_init__(self):
        super().__post_init__()

        if self.num_agents < 1:
            raise ValueError(f"num_agents must be at least 1, got {self.num_agents}")

        if self.agent_coordination not in ["sequential", "parallel", "distributed"]:
            raise ValueError(
                f"agent_coordination must be 'sequential', 'parallel', or 'distributed', got {self.agent_coordination}"
            )

        if self.reasoning_engine not in ["pattern_matcher", "pln", "moses"]:
            LOGGER.warning(
                f"reasoning_engine '{self.reasoning_engine}' is experimental. "
                "Supported engines are: pattern_matcher, pln, moses"
            )

        # List of known backends - this could be made more dynamic in the future
        # by introspecting the backends module
        known_backends = ["pytorch", "onnxruntime", "openvino", "vllm", "tensorrt-llm", "ipex", "llama-cpp"]
        if self.delegate_backend not in known_backends:
            LOGGER.warning(
                f"delegate_backend '{self.delegate_backend}' may not be fully supported. "
                f"Known backends are: {', '.join(known_backends)}"
            )

        # Initialize agent lists if not provided
        if self.agent_models is None:
            self.agent_models = [self.model] * self.num_agents
        elif len(self.agent_models) != self.num_agents:
            raise ValueError(
                f"agent_models length ({len(self.agent_models)}) must match num_agents ({self.num_agents})"
            )

        if self.agent_backends is None:
            self.agent_backends = [self.delegate_backend] * self.num_agents
        elif len(self.agent_backends) != self.num_agents:
            raise ValueError(
                f"agent_backends length ({len(self.agent_backends)}) must match num_agents ({self.num_agents})"
            )

        if self.agent_devices is None:
            self.agent_devices = [self.device] * self.num_agents
        elif len(self.agent_devices) != self.num_agents:
            raise ValueError(
                f"agent_devices length ({len(self.agent_devices)}) must match num_agents ({self.num_agents})"
            )
