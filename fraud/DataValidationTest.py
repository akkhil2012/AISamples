from dataclasses import dataclass, field
from typing import List

@dataclass
class AgentConfig:
    name: str
    model: str = "gpt-4"
    max_tokens: int = 1024
    tools: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

cfg = AgentConfig(name="planner", tools=["search", "code"])
print(cfg)