import asyncio
import requests
from engine import SGlangEngine
from utils import process_response
import runpod
import os

# Initialize the engine
engine = SGlangEngine()
engine.start_server()
engine.wait_for_server()


def get_max_concurrency(default=300):
    """
    Returns the maximum concurrency value.
    By default, it uses 50 unless the 'MAX_CONCURRENCY' environment variable is set.

    Args:
        default (int): The default concurrency value if the environment variable is not set.

    Returns:
        int: The maximum concurrency value.
    """
    return int(os.getenv("MAX_CONCURRENCY", default))


async def async_handler(job):
    """Handle the requests asynchronously."""
    job_input = job["input"]

    try:
        # Case 1: full OpenAI style payload where caller already specifies the route.
        if job_input.get("openai_route"):
            openai_route, openai_input = job_input.get("openai_route"), job_input.get(
                "openai_input"
            )

            openai_url = f"{engine.base_url}" + openai_route
            headers = {"Content-Type": "application/json"}

            try:
                response = requests.post(openai_url, headers=headers, json=openai_input, timeout=300)
            except requests.RequestException as e:
                yield {"error": f"Failed to reach backend: {str(e)}"}
                return

            if not response.ok:
                yield {"error": f"Backend returned {response.status_code}", "details": response.text[:1000]}
                return

            try:
                if openai_input.get("stream", False):
                    for formated_chunk in process_response(response):
                        yield formated_chunk
                else:
                    for chunk in response.iter_lines():
                        if chunk:
                            yield chunk.decode("utf-8")
            except Exception as e:
                yield {"error": f"Error processing response: {str(e)}", "type": type(e).__name__}
                return

        # Case 2: payload looks like OpenAI chat/completions but omits the wrapper.
        elif "messages" in job_input:
            openai_url = f"{engine.base_url}/v1/chat/completions"
            headers = {"Content-Type": "application/json"}

            # Make sure model is set; fall back to default.
            if "model" not in job_input:
                job_input["model"] = engine.model or "default"

            try:
                response = requests.post(openai_url, headers=headers, json=job_input, timeout=300)
            except requests.RequestException as e:
                yield {"error": f"Failed to reach backend: {str(e)}"}
                return

            if not response.ok:
                yield {"error": f"Backend returned {response.status_code}", "details": response.text[:1000]}
                return

            try:
                if job_input.get("stream", False):
                    for formated_chunk in process_response(response):
                        yield formated_chunk
                else:
                    for chunk in response.iter_lines():
                        if chunk:
                            yield chunk.decode("utf-8")
            except Exception as e:
                yield {"error": f"Error processing response: {str(e)}", "type": type(e).__name__}
                return

        # Case 3: assume user meant the native /generate endpoint.
        else:
            generate_url = f"{engine.base_url}/generate"
            headers = {"Content-Type": "application/json"}

            try:
                response = requests.post(generate_url, json=job_input, headers=headers, timeout=300)
            except requests.RequestException as e:
                yield {"error": f"Failed to reach backend: {str(e)}"}
                return

            if response.ok:
                try:
                    yield response.json()
                except Exception as e:
                    yield {"error": f"Failed to parse response JSON: {str(e)}", "raw": response.text[:500]}
            else:
                yield {"error": f"Generate request failed with status code {response.status_code}", "details": response.text[:1000]}

    except Exception as e:
        # Catch any unexpected errors at the top level
        yield {
            "error": f"Unexpected error in handler: {str(e)}",
            "type": type(e).__name__,
        }


runpod.serverless.start(
    {
        "handler": async_handler,
        "concurrency_modifier": get_max_concurrency,
        "return_aggregate_stream": True,
    }
)
