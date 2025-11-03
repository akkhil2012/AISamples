from molecules import create_molecular_context

# Use the function with your own data
context = create_molecular_context(
    instruction="Your instruction here",
    examples=[{"input": "in", "output": "out"}],
    new_input="new input"
)