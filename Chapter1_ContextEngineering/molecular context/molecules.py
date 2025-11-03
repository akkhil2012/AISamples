def create_molecular_context(instruction, examples, new_input, 
                           format_type="input-output"):
    """
    Construct a molecular context from examples.
    
    Args:
        instruction (str): The task instruction
        examples (List[Dict]): List of example input/output pairs
        new_input (str): The new input to process
        format_type (str): Template type (input-output, chain-of-thought)
    
    Returns:
        str: The complete molecular context
    """
    context = f"{instruction}\n\n"
    
    # Add examples based on format type
    if format_type == "input-output":
        for example in examples:
            context += f"Input: {example['input']}\n"
            context += f"Output: {example['output']}\n\n"
    elif format_type == "chain-of-thought":
        for example in examples:
            context += f"Input: {example['input']}\n"
            context += f"Thinking: {example['thinking']}\n"
            context += f"Output: {example['output']}\n\n"
    
    # Add the new input
    context += f"Input: {new_input}\nOutput:"
    
    return context

# Example usage
if __name__ == "__main__":
    # Example 1: Input-Output format
    instruction = "Translate the following English sentences to French:"
    examples = [
        {"input": "Hello, how are you?", "output": "Bonjour, comment ça va?"},
        {"input": "What time is it?", "output": "Quelle heure est-il?"}
    ]
    new_input = "Where is the nearest restaurant?"
    
    print("=== Input-Output Format Example ===")
    result = create_molecular_context(instruction, examples, new_input)
    print(result)
    
    # Example 2: Chain-of-Thought format
    instruction = "Solve the following math problems step by step:"
    examples = [
        {
            "input": "What is 5 + 3?",
            "thinking": "I need to add 5 and 3 together.",
            "output": "8"
        },
        {
            "input": "What is 10 - 4?",
            "thinking": "I need to subtract 4 from 10.",
            "output": "6"
        }
    ]
    new_math_input = "What is 7 * 6?"
    
    print("\n=== Chain-of-Thought Format Example ===")
    result = create_molecular_context(
        instruction, 
        examples, 
        new_math_input,
        format_type="chain-of-thought"
    )
    print(result)