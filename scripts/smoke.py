import argparse
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from macrs.orchestrator import Orchestrator
from macrs.state import ConversationState


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single MACRS turn")
    parser.add_argument("message", help="User message to process")
    parser.add_argument("--session-id", default="demo-session")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--stream-graph", action="store_true", help="Stream graph node updates")
    parser.add_argument("--stream-tokens", action="store_true", help="Stream LLM tokens")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    state = ConversationState(session_id=args.session_id)
    orchestrator = Orchestrator()
    if args.stream_tokens:
        os.environ["MACRS_STREAM_TOKENS"] = "1"

    if args.stream_graph:
        final_state = None
        for update in orchestrator.stream_turn(state, args.message):
            for node, payload in update.items():
                print(f"\n[graph] node={node} keys={list(payload.keys())}")
                if node == "reflection":
                    final_state = payload
        if final_state is None:
            raise RuntimeError("No final state received from stream")
        decision = final_state["planner_decision"]
    else:
        result = orchestrator.run_turn(state, args.message)
        decision = result["planner_decision"]

    print(f"\nSelected act: {decision.selected_act}")
    print(f"Response:\n{decision.selected_response}")


if __name__ == "__main__":
    main()
