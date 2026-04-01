from datetime import datetime
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from graph import graph


def log(message: str) -> None:
    print(
        f"[{datetime.now().isoformat(timespec='seconds')}] [main] {message}", flush=True
    )


log("Import complete. Initializing FastAPI app.")
app = FastAPI(title="Literature Prototyper")
log("FastAPI app initialized.")


class ExecuteRequest(BaseModel):
    query: str


class ExecuteResponse(BaseModel):
    final_output: str
    trace_log: list[dict]
    total_tokens: int


@app.on_event("startup")
async def on_startup() -> None:
    log("Startup event triggered. Service is ready to accept requests.")


@app.middleware("http")
async def request_logger(request: Request, call_next):
    log(f"Incoming request: method={request.method} path={request.url.path}")
    response = await call_next(request)
    log(
        "Request completed: "
        f"method={request.method} path={request.url.path} status={response.status_code}"
    )
    return response


@app.post("/v1/execute", response_model=ExecuteResponse)
async def execute_agent(request: ExecuteRequest):
    log("execute_agent started")
    try:
        initial_state = {"messages": [HumanMessage(content=request.query)]}
        trace_log = []
        final_output = ""
        total_tokens = 0
        log("Workflow started")

        for event in graph.stream(initial_state):
            for node_name, node_data in event.items():
                log(f"Output received from {node_name}")
                trace_entry = {"agent": node_name, "output_summary": str(node_data)}
                trace_log.append(trace_entry)

                if isinstance(node_data, dict) and node_data.get("messages"):
                    candidate_message = node_data["messages"][-1]
                    raw_content = getattr(candidate_message, "content", "")

                    if isinstance(raw_content, list):
                        text_parts = [
                            p.get("text", "")
                            for p in raw_content
                            if isinstance(p, dict) and "text" in p
                        ]
                        parsed_content = "\n".join(text_parts)
                    else:
                        parsed_content = str(raw_content)

                    if parsed_content.strip():
                        final_output = parsed_content

                    # Grok/OpenAI token extraction
                    usage_metadata = getattr(candidate_message, "usage_metadata", None)
                    response_metadata = getattr(
                        candidate_message, "response_metadata", {}
                    )

                    if (
                        isinstance(usage_metadata, dict)
                        and "total_tokens" in usage_metadata
                    ):
                        total_tokens = max(total_tokens, usage_metadata["total_tokens"])
                    elif "token_usage" in response_metadata:
                        total_tokens = max(
                            total_tokens,
                            response_metadata["token_usage"].get("total_tokens", 0),
                        )

        if not final_output:
            log("No final output captured")
            raise ValueError("No final output produced by workflow")

        log("Workflow complete")
        log(f"API final output: {final_output[:300]}")
        log(f"Total tokens: {total_tokens}")

        log("Returning response")
        return ExecuteResponse(
            final_output=final_output,
            trace_log=trace_log,
            total_tokens=total_tokens,
        )

    except Exception as e:
        log(f"Exception in execute_agent: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    log("Starting uvicorn server on 0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
