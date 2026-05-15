import argparse
from ReActExample import run_agent

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the ReAct agent from ReActExample.py"
    )
    parser.add_argument(
        "task",
        nargs="+",
        help="Task to send to the agent",
    )
    args = parser.parse_args()
    task = " ".join(args.task)
    result = run_agent(task)
    print(result)

if __name__ == "__main__":
    main()