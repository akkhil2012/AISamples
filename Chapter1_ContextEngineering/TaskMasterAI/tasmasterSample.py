import json
import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import re

@dataclass
class Task:
    """Represents a task extracted from conversation"""
    id: str
    title: str
    description: str
    priority: int  # 1-5, 5 being highest
    status: str  # "pending", "in_progress", "completed", "blocked"
    deadline: Optional[str]
    dependencies: List[str]
    tags: List[str]
    created_at: str
    updated_at: str

@dataclass
class Entity:
    """Represents a key entity (person, project, document, etc.)"""
    id: str
    name: str
    type: str  # "person", "project", "document", "company", etc.
    attributes: Dict[str, Any]
    relationships: Dict[str, List[str]]  # relationship_type -> [entity_ids]
    created_at: str
    updated_at: str

class TaskmasterAI:
    """
    Simplified implementation of Taskmaster AI concepts
    Demonstrates persistent memory and context management
    """
    
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.entities: Dict[str, Entity] = {}
        self.conversation_history: List[Dict] = []
        self.memory_store: Dict[str, Any] = {
            "current_context": {},
            "persistent_facts": {},
            "priority_queue": [],
            "relationships": defaultdict(list)
        }
        
    def process_message(self, user_message: str, context_type: str = "general") -> str:
        """
        Main processing function that demonstrates Taskmaster AI's approach
        """
        timestamp = datetime.datetime.now().isoformat()
        
        # Store conversation
        self.conversation_history.append({
            "timestamp": timestamp,
            "user_message": user_message,
            "context_type": context_type
        })
        
        # Extract structured information
        extracted_info = self._extract_structured_info(user_message)
        
        # Update persistent memory
        self._update_memory(extracted_info)
        
        # Generate context-aware response
        response = self._generate_response(user_message, extracted_info)
        
        return response
    
    def _extract_structured_info(self, message: str) -> Dict[str, Any]:
        """
        Extracts tasks, entities, and relationships from conversation
        Simulates Taskmaster AI's intelligent parsing
        """
        extracted = {
            "tasks": [],
            "entities": [],
            "relationships": [],
            "facts": [],
            "priorities": []
        }
        
        # Simple task extraction (could be enhanced with NLP)
        task_patterns = [
            r"need to (.*?)(?:\.|$)",
            r"should (.*?)(?:\.|$)", 
            r"must (.*?)(?:\.|$)",
            r"have to (.*?)(?:\.|$)",
            r"plan to (.*?)(?:\.|$)",
            r"(?:India|US|USA|United States).*?(?:tariff|trade).*?war",  # Add this line
        ]
        
        for pattern in task_patterns:
            matches = re.findall(pattern, message.lower())
            for match in matches:
                task_id = f"task_{len(self.tasks) + 1}"
                task = Task(
                    id=task_id,
                    title=match.strip(),
                    description=f"Extracted from: {message[:100]}...",
                    priority=3,  # Default priority
                    status="pending",
                    deadline=None,
                    dependencies=[],
                    tags=["auto-extracted"],
                    created_at=datetime.datetime.now().isoformat(),
                    updated_at=datetime.datetime.now().isoformat()
                )
                extracted["tasks"].append(task)
        
        # Extract entities (simplified - could use NER)
        entity_patterns = [
            (r"\b([A-Z][a-z]+ [A-Z][a-z]+)\b", "person"),  # Names
            (r"\$([0-9,]+(?:\.[0-9]{2})?(?:[KMB])?)", "money"),  # Money
            (r"\b([A-Z][a-zA-Z]*Corp|[A-Z][a-zA-Z]*Inc|[A-Z][a-zA-Z]*LLC)\b", "company")  # Companies
        ]
        
        for pattern, entity_type in entity_patterns:
            matches = re.findall(pattern, message)
            for match in matches:
                entity_id = f"{entity_type}_{len(self.entities) + 1}"
                entity = Entity(
                    id=entity_id,
                    name=match,
                    type=entity_type,
                    attributes={"source_message": message},
                    relationships={},
                    created_at=datetime.datetime.now().isoformat(),
                    updated_at=datetime.datetime.now().isoformat()
                )
                extracted["entities"].append(entity)
        
        return extracted
    
    def _update_memory(self, extracted_info: Dict[str, Any]):
        """
        Updates persistent memory with extracted information
        Demonstrates priority-based retention
        """
        # Store tasks
        for task in extracted_info["tasks"]:
            self.tasks[task.id] = task
            self.memory_store["priority_queue"].append({
                "type": "task",
                "id": task.id,
                "priority": task.priority,
                "timestamp": task.created_at
            })
        
        # Store entities
        for entity in extracted_info["entities"]:
            self.entities[entity.id] = entity
            self.memory_store["persistent_facts"][entity.id] = {
                "name": entity.name,
                "type": entity.type,
                "attributes": entity.attributes
            }
        
        # Maintain priority queue (keep top 100 items)
        self.memory_store["priority_queue"].sort(
            key=lambda x: (x["priority"], x["timestamp"]), 
            reverse=True
        )
        if len(self.memory_store["priority_queue"]) > 100:
            self.memory_store["priority_queue"] = self.memory_store["priority_queue"][:100]
    
    def _generate_response(self, user_message: str, extracted_info: Dict[str, Any]) -> str:
        """
        Generates context-aware response using persistent memory
        """
        response_parts = []
        
        # Check if this relates to existing context
        relevant_tasks = self._find_relevant_tasks(user_message)
        relevant_entities = self._find_relevant_entities(user_message)
        
        if extracted_info["tasks"]:
            response_parts.append(f"I've identified {len(extracted_info['tasks'])} new task(s) from your message:")
            for task in extracted_info["tasks"]:
                response_parts.append(f"- {task.title}")
        
        if relevant_tasks:
            response_parts.append(f"\nRelated existing tasks ({len(relevant_tasks)}):")
            for task in relevant_tasks[:3]:  # Show top 3
                response_parts.append(f"- {task.title} (Status: {task.status})")
        
        if relevant_entities:
            response_parts.append(f"\nRelevant context from memory:")
            for entity in relevant_entities[:3]:  # Show top 3
                response_parts.append(f"- {entity.name} ({entity.type})")
        
        # Simulate persistent context awareness
        response_parts.append(f"\nCurrent context summary:")
        response_parts.append(f"- Total tracked tasks: {len(self.tasks)}")
        response_parts.append(f"- Total entities: {len(self.entities)}")
        response_parts.append(f"- Memory items: {len(self.memory_store['priority_queue'])}")
        
        return "\n".join(response_parts)
    
    def _find_relevant_tasks(self, message: str) -> List[Task]:
        """Find tasks relevant to current message"""
        relevant = []
        message_lower = message.lower()
        
        for task in self.tasks.values():
            if any(word in task.title.lower() for word in message_lower.split()):
                relevant.append(task)
        
        return sorted(relevant, key=lambda x: x.priority, reverse=True)
    
    def _find_relevant_entities(self, message: str) -> List[Entity]:
        """Find entities relevant to current message"""
        relevant = []
        message_lower = message.lower()
        
        for entity in self.entities.values():
            if entity.name.lower() in message_lower:
                relevant.append(entity)
        
        return relevant
    
    def simulate_context_overflow(self) -> str:
        """
        Simulates what happens when context window overflows
        Demonstrates how Taskmaster AI preserves important information
        """
        preserved_info = {
            "high_priority_tasks": [
                task for task in self.tasks.values() 
                if task.priority >= 4
            ],
            "key_entities": [
                entity for entity in self.entities.values()
                if entity.type in ["company", "person", "project"]
            ],
            "persistent_facts": self.memory_store["persistent_facts"]
        }
        
        summary = "CONTEXT OVERFLOW SIMULATION - Information Preserved:\n\n"
        
        if preserved_info["high_priority_tasks"]:
            summary += "High Priority Tasks:\n"
            for task in preserved_info["high_priority_tasks"]:
                summary += f"- {task.title} (Priority: {task.priority})\n"
        
        if preserved_info["key_entities"]:
            summary += "\nKey Entities:\n"
            for entity in preserved_info["key_entities"]:
                summary += f"- {entity.name} ({entity.type})\n"
        
        summary += f"\nPersistent Facts: {len(preserved_info['persistent_facts'])} items retained"
        
        return summary
    
    def get_status_report(self) -> str:
        """Generate a comprehensive status report"""
        report = "=== TASKMASTER AI STATUS REPORT ===\n\n"
        
        # Task summary
        task_stats = defaultdict(int)
        for task in self.tasks.values():
            task_stats[task.status] += 1
        
        report += "TASK SUMMARY:\n"
        for status, count in task_stats.items():
            report += f"- {status.title()}: {count}\n"
        
        # Entity summary
        entity_stats = defaultdict(int)
        for entity in self.entities.values():
            entity_stats[entity.type] += 1
        
        report += "\nENTITY SUMMARY:\n"
        for entity_type, count in entity_stats.items():
            report += f"- {entity_type.title()}: {count}\n"
        
        # Memory usage
        report += f"\nMEMORY USAGE:\n"
        report += f"- Conversation history: {len(self.conversation_history)} messages\n"
        report += f"- Priority queue: {len(self.memory_store['priority_queue'])} items\n"
        report += f"- Persistent facts: {len(self.memory_store['persistent_facts'])} items\n"
        
        return report

# Example usage and demonstration
def main():
    """Demonstrate Taskmaster AI capabilities"""
    ai = TaskmasterAI()
    
    print("=== TASKMASTER AI DEMO ===\n")
    print("Simulating the banking workflow scenario from our discussion...\n")
    
    # Simulate the banking project conversation
    messages = [
        "I need to build a workflow orchestration project for banking domain focusing on loan processing",
        "The project should handle customer onboarding, KYC verification, and credit scoring for TechBank Corp",
        "We have a $150K budget and need to complete this by Q2 2025 with compliance for GDPR",
        "The team includes 5 developers and 2 compliance officers, and we need integration with existing core banking systems",
        "There's a critical performance requirement - must process 1000 loan applications per hour",
        # This would normally cause context loss in regular LLMs
        "What was our budget again and what are the key performance requirements?"
    ]
    
    # Process each message
    for i, message in enumerate(messages, 1):
        print(f"USER MESSAGE {i}: {message}")
        response = ai.process_message(message, "banking_project")
        print(f"TASKMASTER AI: {response}\n")
        print("-" * 80 + "\n")
    
    # Demonstrate context overflow handling
    print("=== CONTEXT OVERFLOW SIMULATION ===\n")
    overflow_demo = ai.simulate_context_overflow()
    print(overflow_demo)
    print("\n" + "=" * 80 + "\n")
    
    # Show final status
    print(ai.get_status_report())

if __name__ == "__main__":
    main()
