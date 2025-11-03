from tasmasterSample import TaskmasterAI

taskmaster = TaskmasterAI()

# The AI maintains context between messages
taskmaster.process_message("I need to buy a new laptop")
response = taskmaster.process_message("What's my budget for that?")
# The AI will remember the laptop purchase context

print(taskmaster.simulate_context_overflow())

print("********")

print(response)
