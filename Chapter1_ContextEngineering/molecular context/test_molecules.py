import pytest

from molecules import create_molecular_context


def test_create_molecular_context_input_output():
    instruction = "Translate the following English sentences to French:"
    examples = [
        {"input": "Hello, how are you?", "output": "Bonjour, comment ça va?"},
        {"input": "What time is it?", "output": "Quelle heure est-il?"},
    ]
    new_input = "Where is the nearest restaurant?"

    expected = (
        "Translate the following English sentences to French:\n\n"
        "Input: Hello, how are you?\n"
        "Output: Bonjour, comment ça va?\n\n"
        "Input: What time is it?\n"
        "Output: Quelle heure est-il?\n\n"
        "Input: Where is the nearest restaurant?\n"
        "Output:"
    )

    result = create_molecular_context(instruction, examples, new_input)
    assert result == expected


def test_create_molecular_context_chain_of_thought():
    instruction = "Solve the following math problems step by step:"
    examples = [
        {
            "input": "What is 5 + 3?",
            "thinking": "I need to add 5 and 3 together.",
            "output": "8",
        },
        {
            "input": "What is 10 - 4?",
            "thinking": "I need to subtract 4 from 10.",
            "output": "6",
        },
    ]
    new_input = "What is 7 * 6?"

    expected = (
        "Solve the following math problems step by step:\n\n"
        "Input: What is 5 + 3?\n"
        "Thinking: I need to add 5 and 3 together.\n"
        "Output: 8\n\n"
        "Input: What is 10 - 4?\n"
        "Thinking: I need to subtract 4 from 10.\n"
        "Output: 6\n\n"
        "Input: What is 7 * 6?\n"
        "Output:"
    )

    result = create_molecular_context(
        instruction, examples, new_input, format_type="chain-of-thought"
    )
    assert result == expected


def test_create_molecular_context_with_empty_examples():
    instruction = "Summarize the text:"
    examples = []
    new_input = "Large language models are useful."

    expected = (
        "Summarize the text:\n\n"
        "Input: Large language models are useful.\n"
        "Output:"
    )

    result = create_molecular_context(instruction, examples, new_input)
    assert result == expected


def test_create_molecular_context_invalid_format_type_graceful():
    instruction = "Do something:"
    examples = [
        {"input": "A", "output": "B"},
    ]
    new_input = "C"

    # Unknown format_type should skip example rendering and still produce final Input/Output
    expected = (
        "Do something:\n\n"
        "Input: C\n"
        "Output:"
    )

    result = create_molecular_context(instruction, examples, new_input, format_type="unknown")
    assert result == expected


