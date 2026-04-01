import operator
import os
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Sequence, TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_experimental.tools import PythonREPLTool

from langgraph.graph import StateGraph, END


def log(message: str) -> None:
    print(
        f"[{datetime.now().isoformat(timespec='seconds')}] [graph] {message}",
        flush=True,
    )


log("Loading environment variables from .env")
load_dotenv(Path(__file__).with_name(".env"), override=True)


# 1. Define State and Routing
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str


class Route(BaseModel):
    next: str = Field(
        description="The next agent to route to. Options are: 'Literature_Analyst', 'Prototyping_Engineer', or 'FINISH'."
    )


# 2. Initialize Tools
log("Initializing tools")
python_repl_tool = PythonREPLTool()

# 3. Initialize LLM (Groq)
groq_api_key = os.getenv("Groq_API_Key") or os.getenv("GROQ_API_KEY")
if not groq_api_key:
    log("WARNING: No Groq_API_Key found in environment variables.")

log("Initializing Groq model (Llama 3.3 70B)")
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=groq_api_key,
    temperature=0,
)

# 4. Define Prompts (no tool-based agents)
analyst_prompt = (
    "You are the Literature Analyst. When given a topic, you provide a clear summary of its core concepts, "
    "mathematical principles, algorithms, and key implementation patterns based on your knowledge. "
    "Be specific and actionable so an engineer can build working code from your explanation."
)

engineer_prompt = (
    "You are the Prototyping Engineer. Based on the research provided, write functional Python code that "
    "demonstrates the concept. Write and test the code. Return clean, complete, ready-to-run Python code."
)


# 5. Define Nodes
def _trim_content(content: Any, max_chars: int = 1500) -> str:
    if isinstance(content, list):
        text_parts = [
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and "text" in part
        ]
        content = "\n".join(text_parts)
    elif not isinstance(content, str):
        content = str(content)

    if len(content) <= max_chars:
        return content
    return content[:max_chars] + "\n...[truncated for context limits]"


def analyst_node(state: AgentState) -> dict:
    log("Literature_Analyst starting")
    compact_messages = list(state["messages"][-2:])

    # Ensure conversation ends with Human message
    if compact_messages and isinstance(compact_messages[-1], AIMessage):
        compact_messages.append(
            HumanMessage(content="Please provide your research summary.")
        )

    # Build message chain with analyst system prompt
    messages_with_system = [SystemMessage(content=analyst_prompt)] + compact_messages

    result = llm.invoke(messages_with_system)
    output_text = _trim_content(result.content)
    log(f"Literature_Analyst done: {output_text[:150]}")
    return {"messages": [AIMessage(content=output_text)]}


def engineer_node(state: AgentState) -> dict:
    log("Prototyping_Engineer starting")
    compact_messages = list(state["messages"][-2:])

    if compact_messages and isinstance(compact_messages[-1], AIMessage):
        compact_messages.append(HumanMessage(content="Write the Python code now."))

    # Build message chain with engineer system prompt
    messages_with_system = [SystemMessage(content=engineer_prompt)] + compact_messages

    result = llm.invoke(messages_with_system)
    output_text = _trim_content(result.content)
    log(f"Prototyping_Engineer done: {output_text[:150]}")
    return {"messages": [AIMessage(content=output_text)]}


def supervisor_node(state: AgentState) -> dict:
    log("Supervisor deciding next agent")
    supervisor_prompt = SystemMessage(
        content="You are the Lead Researcher and Supervisor of a 'Paper-to-Code' pipeline. "
        "Your ultimate goal for ANY user request is to research the concept and deliver a "
        "functional Python prototype of it. \n"
        "Follow this strict pipeline:\n"
        "1. Always route to 'Literature_Analyst' first to find the theoretical concepts or formulas.\n"
        "2. Once the research is clear, you MUST route to 'Prototyping_Engineer' to write the Python code.\n"
        "3. Only route to 'FINISH' after the Python code has been successfully written and tested."
    )

    history_text = ""
    for msg in state["messages"][-4:]:
        role = "User" if isinstance(msg, HumanMessage) else "Agent"
        content = _trim_content(getattr(msg, "content", ""), 800)
        history_text += f"[{role}]: {content}\n\n"

    routing_instruction = HumanMessage(
        content=f"Conversation History:\n{history_text}\nGiven the conversation above, who should act next? (Literature_Analyst, Prototyping_Engineer, or FINISH)"
    )

    messages = [supervisor_prompt, routing_instruction]
    supervisor_chain = llm.with_structured_output(Route)
    decision = supervisor_chain.invoke(messages)
    log(f"Supervisor routed to: {decision.next}")

    return {"next": decision.next}


# 6. Build Graph
log("Building LangGraph workflow")
workflow = StateGraph(AgentState)

workflow.add_node("Literature_Analyst", analyst_node)
workflow.add_node("Prototyping_Engineer", engineer_node)
workflow.add_node("Supervisor", supervisor_node)

workflow.add_edge("Literature_Analyst", "Supervisor")
workflow.add_edge("Prototyping_Engineer", "Supervisor")

workflow.add_conditional_edges(
    "Supervisor",
    lambda x: x["next"],
    {
        "Literature_Analyst": "Literature_Analyst",
        "Prototyping_Engineer": "Prototyping_Engineer",
        "FINISH": END,
    },
)

workflow.set_entry_point("Supervisor")
graph = workflow.compile()
log("Graph compiled and ready")
