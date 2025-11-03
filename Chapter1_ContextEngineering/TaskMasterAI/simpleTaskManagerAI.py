
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime

# Step 1: Define Task class
@dataclass
class Task:
    id: str
    title: str
    description: str
    priority: int  # 1 (low) to 5 (high)
    status: str  # e.g., "pending", "completed"
    deadline: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

# Step 2: Define TaskmasterAI class
class TaskmasterAI:
    def __init__(self):
        self.tasks: Dict[str, Task] = {}

    def add_task(self, task: Task):
        self.tasks[task.id] = task

    def update_task(self, task_id: str, status: Optional[str] = None):
        if task_id in self.tasks:
            if status:
                self.tasks[task_id].status = status
            self.tasks[task_id].updated_at = datetime.now().isoformat()

    def get_context_summary(self) -> str:
        if not self.tasks:
            return "No tasks in memory."
        lines = ["Current tasks:\n"]
        for task in self.tasks.values():
            lines.append(f"- [{task.status.upper()}] {task.title} (Due: {task.deadline or 'N/A'})")
        return "\n".join(lines)


task = Task(
    id="task001",
    title="Call Alice about Q3 report",
    description="Reminder to call Alice and ask about the Q3 report",
    priority=3,
    status="pending",
    deadline="2025-08-08",
    dependencies=[],
    tags=["reminder", "Q3"],
    created_at="2025-08-07",
    updated_at="2025-08-07"
)

ai = TaskmasterAI()
ai.add_task(task)

user_query = "Has Alice replied about that report?"
context = ai.get_context_summary()  # Or ai.to_prompt()

# Combine user query + context for LLM
full_prompt = context + "\n\n" + user_query

print(full_prompt)
