import argparse
import logging
import sys
from pathlib import Path
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from macrs.orchestrator import Orchestrator
from macrs.state import ConversationState


def _print_help() -> None:
    print("Commands:")
    print("  /exit or /quit  - End the chat")
    print("  /help           - Show this help")
    print("  /state          - Show current strategy weights and turn id")


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive MACRS chat CLI")
    parser.add_argument("--session-id", default=str(uuid.uuid4()), help="Session ID for the conversation")
    parser.add_argument("--plain", action="store_true", help="Disable rich formatting")
    args = parser.parse_args()

    logging.basicConfig(level="WARNING")
    for name in ["httpx", "httpcore", "langchain", "langchain_groq"]:
        logging.getLogger(name).setLevel(logging.WARNING)

    orchestrator = Orchestrator()
    state = ConversationState(session_id=args.session_id)

    console = None
    if not args.plain:
        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich.progress import Progress, SpinnerColumn, TextColumn

            console = Console()
            console.print(
                Panel.fit(
                    "MACRS CLI ready. Type /help for commands.",
                    title="MACRS",
                    border_style="grey50",
                )
            )
        except Exception:
            console = None

    if console is None:
        print("MACRS CLI ready. Type /help for commands.")

    while True:
        try:
            user_message = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not user_message:
            continue
        if user_message in {"/exit", "/quit"}:
            print("Goodbye.")
            break
        if user_message == "/help":
            _print_help()
            continue
        if user_message == "/state":
            print(f"Turn: {state.turn_id}")
            print(f"Weights: {state.strategy_weights}")
            continue

        if console:
            from rich.panel import Panel
            from rich.progress import Progress, SpinnerColumn, TextColumn

            progress = Progress(
                SpinnerColumn(),
                TextColumn("{task.description}"),
                console=console,
                transient=True,
            )
            task_ids = {
                "ask": progress.add_task("Ask agent waiting...", total=1, start=False),
                "recommend": progress.add_task("Recommend agent waiting...", total=1, start=False),
                "chitchat": progress.add_task("Chit-chat agent waiting...", total=1, start=False),
                "planner": progress.add_task("Planner waiting...", total=1, start=False),
                "reflection": progress.add_task("Reflection waiting...", total=1, start=False),
            }

            class RichNodeHandler(logging.Handler):
                def emit(self, record: logging.LogRecord) -> None:
                    mapping = {
                        "macrs.agent.ask": "ask",
                        "macrs.agent.recommend": "recommend",
                        "macrs.agent.chitchat": "chitchat",
                        "macrs.planner": "planner",
                        "macrs.reflection": "reflection",
                    }
                    key = mapping.get(record.name)
                    if not key:
                        return
                    task_id = task_ids.get(key)
                    if task_id is None:
                        return
                    msg = record.getMessage()
                    if msg.startswith("start"):
                        progress.update(task_id, description=f"{key} running...", completed=0)
                        progress.start_task(task_id)
                    elif msg.startswith("done") or msg.startswith("selected") or msg.startswith("updated"):
                        progress.update(task_id, description=f"{key} done", completed=1)

            handler = RichNodeHandler()
            for logger_name in [
                "macrs.agent.ask",
                "macrs.agent.recommend",
                "macrs.agent.chitchat",
                "macrs.planner",
                "macrs.reflection",
            ]:
                logger = logging.getLogger(logger_name)
                logger.setLevel(logging.INFO)
                logger.propagate = False
                logger.addHandler(handler)

            with progress:
                result = orchestrator.run_turn(state, user_message)

            for logger_name in [
                "macrs.agent.ask",
                "macrs.agent.recommend",
                "macrs.agent.chitchat",
                "macrs.planner",
                "macrs.reflection",
            ]:
                logger = logging.getLogger(logger_name)
                logger.removeHandler(handler)

            decision = result["planner_decision"]
            state = result.get("conversation_state", state)
            console.print(
                Panel(
                    decision.selected_response,
                    title="Assistant",
                    border_style="cyan",
                )
            )
        else:
            result = orchestrator.run_turn(state, user_message)
            decision = result["planner_decision"]
            state = result.get("conversation_state", state)
            print(f"\nAssistant: {decision.selected_response}")


if __name__ == "__main__":
    main()
