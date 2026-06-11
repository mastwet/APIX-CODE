import sys
from typing import Annotated, Optional

from langchain.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from langchain_core.messages import ToolMessage

from ..event.stream_writer import AgentStreamWriter, AgentStreamEvent
from .tool_descriptions import REQUEST_USER_INPUT_PROMPT


@tool(description=REQUEST_USER_INPUT_PROMPT)
async def request_user_input(
    questions: list[dict],
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:

    target = state.get("target", {})
    generation_id = state.get("generation_id")

    event_writer = AgentStreamWriter(generation_id)
    event_writer.send_event(
        event=AgentStreamEvent.TOOL_EXEC_START,
        data={
            "event_name": "tool_exec_chunk_rtn",
            "tool_name": "request_user_input",
            "tool_call_id": tool_call_id,
            "content": str(questions),
            "chunk_position": "start",
            "status": "success",
        }
    )

    if not questions:
        event_writer.send_event(
            event=AgentStreamEvent.TOOL_EXEC_END,
            data={
                "event_name": "tool_exec_chunk_rtn",
                "tool_name": "request_user_input",
                "tool_call_id": tool_call_id,
                "content": "No questions provided.",
                "chunk_position": "end",
                "status": "fail",
            }
        )
        return Command(update={
            "messages": [
                ToolMessage("No questions provided.", tool_call_id=tool_call_id)
            ]
        })

    # CLI mode: print questions and read from stdin
    responses: list[dict] = []

    try:
        for q in questions:
            question_text = q.get("question", "The question text is missing.")
            options = q.get("options", [])
            multiselection = q.get("multiselection", False)

            print(f"\n{'='*60}")
            print(f"Question: {question_text}")
            print(f"{'='*60}")

            if options:
                for i, opt in enumerate(options):
                    marker = "[ ]" if multiselection else f"  {i+1}."
                    print(f"  {marker} {opt}")
                if multiselection:
                    print("  (Enter comma-separated numbers, or type your own answer)")
                else:
                    print("  (Enter number or type your own answer)")

            try:
                answer = input("Your answer: ")
            except EOFError:
                answer = "[User did not provide an answer]"

            responses.append({
                "question": question_text,
                "response": answer,
            })

    except Exception as e:
        event_writer.send_event(
            event=AgentStreamEvent.TOOL_EXEC_END,
            data={
                "event_name": "tool_exec_chunk_rtn",
                "tool_name": "request_user_input",
                "tool_call_id": tool_call_id,
                "content": f"Error: {str(e)}",
                "chunk_position": "end",
                "status": "fail",
            }
        )
        return Command(update={
            "messages": [
                ToolMessage(
                    "The user is currently unavailable or refuses to answer.",
                    tool_call_id=tool_call_id,
                )
            ]
        })

    # Format the result
    parsed_lines: list[str] = []
    for r in responses:
        line0 = "QUESTION:  \n" + (r.get("question", "The question text is missing.") or "The question text is missing.") + "  \n"
        resp = r.get("response", "[User did not provide an answer]") or "[User did not provide an answer]"
        parsed_resp = resp
        if isinstance(resp, list):
            parsed_resp = ""
            for ur in resp:
                parsed_resp += "- " + str(ur) + "  \n"
        line1 = "RESPONSE:  \n" + parsed_resp
        parsed_lines.append(line0 + line1)

    parsed_result = "\n\n".join(parsed_lines)

    event_writer.send_event(
        event=AgentStreamEvent.TOOL_EXEC_END,
        data={
            "event_name": "tool_exec_chunk_rtn",
            "tool_name": "request_user_input",
            "tool_call_id": tool_call_id,
            "content": "",
            "chunk_position": "end",
            "status": "success",
        }
    )

    return Command(update={
        "messages": [
            ToolMessage(
                ("## Get response from user:\n\n" + parsed_result) if parsed_result else "The user is currently unavailable or refuses to answer.",
                tool_call_id=tool_call_id,
            )
        ]
    })
