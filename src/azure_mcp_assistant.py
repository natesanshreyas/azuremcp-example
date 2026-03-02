from __future__ import annotations

import json
import os
import re
import select
import shlex
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .openai_client import OpenAISettings, chat_completion


class AzureMCPError(Exception):
    pass


@dataclass
class ToolCallRecord:
    name: str
    arguments: Dict[str, Any]
    result_preview: str


@dataclass
class QueryAnswer:
    answer: str
    iterations: int
    tool_calls: List[ToolCallRecord]


class StdioMCPClient:
    def __init__(
        self,
        command: str,
        env: Optional[Dict[str, str]] = None,
        io_timeout_seconds: float = 45.0,
    ):
        self.command = command
        self.env = env or {}
        self.io_timeout_seconds = io_timeout_seconds
        self.proc: Optional[subprocess.Popen[bytes]] = None
        self._next_id = 1

    def __enter__(self) -> "StdioMCPClient":
        argv = shlex.split(self.command)
        if not argv:
            raise AzureMCPError("AZURE_MCP_SERVER_COMMAND is empty")
        merged_env = os.environ.copy()
        merged_env.update(self.env)
        self.proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=merged_env,
        )
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def close(self):
        if not self.proc:
            return
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    def initialize(self):
        self._request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "clientInfo": {"name": "azure-mcp-subscription-assistant", "version": "0.1.0"},
                "capabilities": {},
            },
        )
        self._notify("notifications/initialized", {})

    def list_tools(self) -> List[Dict[str, Any]]:
        result = self._request("tools/list", {})
        tools = result.get("tools")
        return tools if isinstance(tools, list) else []

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("tools/call", {"name": name, "arguments": arguments})

    def _notify(self, method: str, params: Dict[str, Any]):
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def _request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        request_id = self._next_id
        self._next_id += 1
        self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})

        while True:
            message = self._recv()
            if message.get("id") != request_id:
                continue
            if "error" in message:
                raise AzureMCPError(f"MCP {method} error: {json.dumps(message['error'])}")
            result = message.get("result")
            return result if isinstance(result, dict) else {"value": result}

    def _send(self, payload: Dict[str, Any]):
        if not self.proc or not self.proc.stdin:
            raise AzureMCPError("MCP process not started")
        body = json.dumps(payload).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        self.proc.stdin.write(header)
        self.proc.stdin.write(body)
        self.proc.stdin.flush()

    def _recv(self) -> Dict[str, Any]:
        if not self.proc or not self.proc.stdout:
            raise AzureMCPError("MCP process not started")

        headers: Dict[str, str] = {}
        while True:
            self._wait_for_stdout("header")
            line = self.proc.stdout.readline()
            if not line:
                stderr_msg = ""
                if self.proc.stderr:
                    stderr_msg = self.proc.stderr.read().decode("utf-8", errors="ignore")
                hint = ""
                lowered = stderr_msg.lower()
                if "ebadengine" in lowered or "unsupported engine" in lowered:
                    hint = (
                        " Azure MCP requires Node.js >=20. "
                        "Current runtime appears incompatible. "
                        "Upgrade Node and restart the app."
                    )
                raise AzureMCPError(f"MCP process exited unexpectedly. {stderr_msg}{hint}")
            text = line.decode("utf-8", errors="ignore").strip()
            if not text:
                break
            if ":" in text:
                key, value = text.split(":", 1)
                headers[key.strip().lower()] = value.strip()

        content_length = int(headers.get("content-length", "0"))
        if content_length <= 0:
            raise AzureMCPError("Missing Content-Length in MCP response")
        body = self._read_exactly(content_length)
        if not body:
            raise AzureMCPError("Empty MCP response body")
        return json.loads(body.decode("utf-8"))

    def _wait_for_stdout(self, phase: str):
        if not self.proc or not self.proc.stdout:
            raise AzureMCPError("MCP process not started")

        fd = self.proc.stdout.fileno()
        ready, _, _ = select.select([fd], [], [], self.io_timeout_seconds)
        if ready:
            return

        process_state = "running"
        if self.proc.poll() is not None:
            process_state = f"exited ({self.proc.returncode})"

        hint = (
            " Ensure Azure auth is ready (for example run 'az login') and that "
            "AZURE_MCP_SERVER_COMMAND is valid."
        )
        raise AzureMCPError(
            (
                f"Timed out waiting for MCP {phase} after {self.io_timeout_seconds:.0f}s "
                f"while process is {process_state}.{hint}"
            )
        )

    def _read_exactly(self, size: int) -> bytes:
        if not self.proc or not self.proc.stdout:
            raise AzureMCPError("MCP process not started")

        chunks: List[bytes] = []
        remaining = size
        started_at = time.monotonic()

        while remaining > 0:
            elapsed = time.monotonic() - started_at
            if elapsed > self.io_timeout_seconds:
                raise AzureMCPError(
                    f"Timed out reading MCP body after {self.io_timeout_seconds:.0f}s"
                )
            self._wait_for_stdout("body")
            piece = self.proc.stdout.read(remaining)
            if not piece:
                raise AzureMCPError("MCP stream ended while reading response body")
            chunks.append(piece)
            remaining -= len(piece)

        return b"".join(chunks)


def _extract_json_dict(text: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        parsed = json.loads(text[start : end + 1])
        if isinstance(parsed, dict):
            return parsed

    raise AzureMCPError("LLM did not return valid JSON")


def _format_tool_result(result: Dict[str, Any], max_chars: int = 5000) -> str:
    content = result.get("content")
    if isinstance(content, list):
        chunks: List[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if isinstance(item.get("text"), str):
                chunks.append(item["text"])
            elif "json" in item:
                chunks.append(json.dumps(item["json"], ensure_ascii=False))
        output = "\n".join(chunks).strip()
    else:
        output = json.dumps(result, ensure_ascii=False)
    return output[:max_chars] + ("\n...[truncated]" if len(output) > max_chars else "")


def _tool_manifest(tools: List[Dict[str, Any]]) -> str:
    compact = []
    for tool in tools:
        compact.append(
            {
                "name": tool.get("name"),
                "description": tool.get("description", ""),
                "inputSchema": tool.get("inputSchema", {}),
            }
        )
    return json.dumps(compact, ensure_ascii=False)


def _validate_azure_mcp_command(command: str):
    lowered = command.lower()
    if "azure" not in lowered or "mcp" not in lowered:
        raise AzureMCPError(
            "AZURE_MCP_SERVER_COMMAND must point to Azure MCP (example: npx -y @azure/mcp@latest server start)"
        )


def _extract_first_json_object(text: str) -> Dict[str, Any]:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        start = match.start()
        try:
            parsed, _ = decoder.raw_decode(text[start:])
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    raise AzureMCPError("Azure MCP output did not contain valid JSON")


def _run_azmcp_cli(
    command_base: str,
    subcommand: str,
    arguments: Optional[Dict[str, Any]] = None,
    timeout_seconds: int = 60,
) -> Dict[str, Any]:
    argv = shlex.split(command_base)
    if not argv:
        raise AzureMCPError("AZURE_MCP_SERVER_COMMAND is empty")

    if "server" in argv:
        server_idx = argv.index("server")
        argv = argv[:server_idx]

    argv.extend(shlex.split(subcommand))

    args = arguments or {}
    for key, value in args.items():
        opt = str(key).strip()
        if not opt:
            continue
        if not opt.startswith("--"):
            opt = f"--{opt}"

        if isinstance(value, bool):
            if value:
                argv.append(opt)
            continue
        if value is None:
            continue
        if isinstance(value, list):
            for item in value:
                argv.extend([opt, str(item)])
            continue

        argv.extend([opt, str(value)])

    proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_seconds)
    output = (proc.stdout or "") + "\n" + (proc.stderr or "")

    try:
        payload = _extract_first_json_object(proc.stdout or "")
    except AzureMCPError:
        payload = _extract_first_json_object(output)

    if proc.returncode != 0:
        raise AzureMCPError(f"Azure MCP CLI failed ({proc.returncode}): {output[:800]}")

    return payload


def _build_cli_manifest(tools: List[Dict[str, Any]], question: str) -> List[Dict[str, Any]]:
    keywords = [w.lower() for w in re.findall(r"[a-zA-Z0-9_\-]+", question) if len(w) > 2]

    scored: List[tuple[int, Dict[str, Any]]] = []
    for tool in tools:
        command = str(tool.get("command", "")).strip()
        if not command:
            continue

        description = str(tool.get("description", "")).strip()
        text = f"{command} {description}".lower()
        score = 0
        for kw in keywords:
            if kw in text:
                score += 3
        if command.startswith("foundry "):
            score += 2
        if command in {"subscription list", "group list"}:
            score += 2

        options = [
            opt.get("name")
            for opt in (tool.get("option") or [])
            if isinstance(opt, dict) and isinstance(opt.get("name"), str)
        ]
        scored.append(
            (
                score,
                {
                    "command": command,
                    "description": description[:180],
                    "options": options[:20],
                },
            )
        )

    scored.sort(key=lambda item: item[0], reverse=True)

    selected = [entry for _, entry in scored[:80]]
    required = {"subscription list", "group list"}
    for _, entry in scored:
        if entry["command"] in required and entry not in selected:
            selected.append(entry)

    dedup: Dict[str, Dict[str, Any]] = {}
    for entry in selected:
        dedup[entry["command"]] = entry
    return list(dedup.values())


def _query_subscription_via_cli(
    openai_settings: OpenAISettings,
    question: str,
    command_base: str,
    subscription_id: Optional[str],
    max_iterations: int,
) -> QueryAnswer:
    tools_payload = _run_azmcp_cli(command_base, "tools list", timeout_seconds=90)
    tool_entries = tools_payload.get("results")
    if not isinstance(tool_entries, list) or not tool_entries:
        raise AzureMCPError("Azure MCP tools list returned no tools")

    manifest_entries = _build_cli_manifest(tool_entries, question)
    allowed_commands = {entry["command"] for entry in manifest_entries}
    # Map command → set of accepted option names so we can safely inject --subscription
    command_options: Dict[str, List[str]] = {
        entry["command"]: entry.get("options", []) for entry in manifest_entries
    }
    manifest = json.dumps(manifest_entries, ensure_ascii=False)

    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are an Azure operations copilot using Azure MCP CLI commands only. "
                "Return strict JSON with one of these shapes: "
                "{\"action\":\"cli_call\",\"command\":\"group list\",\"arguments\":{},\"reason\":\"...\"} "
                "or {\"action\":\"final\",\"answer\":\"...\"}. "
                "Only choose command values from the provided manifest. "
                "Only include arguments that are listed in the command's 'options'. "
                "Do NOT add --subscription to commands that do not list it in their options."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n"
                f"Subscription hint: {subscription_id or os.getenv('AZURE_SUBSCRIPTION_ID', 'not set')}\n"
                f"Available Azure MCP commands: {manifest}"
            ),
        },
    ]

    trace: List[ToolCallRecord] = []

    for i in range(1, max_iterations + 1):
        decision_raw = chat_completion(openai_settings, messages, temperature=0.0, max_tokens=1200)
        decision = _extract_json_dict(decision_raw)
        action = decision.get("action")

        if action == "final":
            answer = str(decision.get("answer", "")).strip()
            if not answer:
                raise AzureMCPError("LLM returned final without answer")
            return QueryAnswer(answer=answer, iterations=i, tool_calls=trace)

        if action != "cli_call":
            raise AzureMCPError(f"Unsupported LLM action: {action}")

        command = str(decision.get("command", "")).strip()
        if command not in allowed_commands:
            raise AzureMCPError(f"Unknown Azure MCP command requested: {command}")

        args = decision.get("arguments", {})
        if not isinstance(args, dict):
            args = {}

        effective_sub = subscription_id or os.getenv("AZURE_SUBSCRIPTION_ID", "").strip()
        if effective_sub and "--subscription" not in args and "--subscription" in command_options.get(command, []):
            args["--subscription"] = effective_sub

        result = _run_azmcp_cli(command_base, command, arguments=args, timeout_seconds=90)
        preview = _format_tool_result({"content": [{"json": result}]})
        trace.append(ToolCallRecord(name=command, arguments=args, result_preview=preview))

        messages.append(
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "action": "cli_call",
                        "command": command,
                        "arguments": args,
                        "reason": decision.get("reason", ""),
                    },
                    ensure_ascii=False,
                ),
            }
        )
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Azure MCP CLI result ({command}):\n{preview}\n"
                    "If this is sufficient, return final answer; otherwise call another command."
                ),
            }
        )

    raise AzureMCPError(f"No final answer after {max_iterations} iterations")


def query_subscription(
    openai_settings: OpenAISettings,
    question: str,
    subscription_id: Optional[str] = None,
    max_iterations: int = 6,
) -> QueryAnswer:
    if not question.strip():
        raise AzureMCPError("question cannot be empty")

    command = os.getenv("AZURE_MCP_SERVER_COMMAND", "").strip()
    if not command:
        raise AzureMCPError("AZURE_MCP_SERVER_COMMAND is not set")
    _validate_azure_mcp_command(command)

    transport = os.getenv("AZURE_MCP_TRANSPORT", "cli").strip().lower()
    if transport == "stdio":
        try:
            extra_env: Dict[str, str] = {}
            if subscription_id:
                extra_env["AZURE_SUBSCRIPTION_ID"] = subscription_id

            with StdioMCPClient(command, env=extra_env) as mcp:
                mcp.initialize()
                tools = mcp.list_tools()
                tool_names = {t.get("name") for t in tools if t.get("name")}
                manifest = _tool_manifest(tools)

                messages: List[Dict[str, str]] = [
                    {
                        "role": "system",
                        "content": (
                            "You are an Azure operations copilot using Azure MCP tools only. "
                            "Always return strict JSON with one of these shapes: "
                            "{\"action\":\"tool_call\",\"tool\":\"name\",\"arguments\":{...},\"reason\":\"...\"} "
                            "or {\"action\":\"final\",\"answer\":\"...\"}. "
                            "Ground your answer only in MCP tool results."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Question: {question}\n"
                            f"Subscription hint: {subscription_id or os.getenv('AZURE_SUBSCRIPTION_ID', 'not set')}\n"
                            f"Available Azure MCP tools: {manifest}"
                        ),
                    },
                ]

                trace: List[ToolCallRecord] = []

                for i in range(1, max_iterations + 1):
                    decision_raw = chat_completion(
                        openai_settings,
                        messages,
                        temperature=0.0,
                        max_tokens=1200,
                    )
                    decision = _extract_json_dict(decision_raw)
                    action = decision.get("action")

                    if action == "final":
                        answer = str(decision.get("answer", "")).strip()
                        if not answer:
                            raise AzureMCPError("LLM returned final without answer")
                        return QueryAnswer(answer=answer, iterations=i, tool_calls=trace)

                    if action != "tool_call":
                        raise AzureMCPError(f"Unsupported LLM action: {action}")

                    tool_name = str(decision.get("tool", "")).strip()
                    args = decision.get("arguments", {})
                    if not isinstance(args, dict):
                        args = {}
                    if tool_name not in tool_names:
                        raise AzureMCPError(f"Unknown Azure MCP tool requested: {tool_name}")

                    result = mcp.call_tool(tool_name, args)
                    preview = _format_tool_result(result)
                    trace.append(ToolCallRecord(name=tool_name, arguments=args, result_preview=preview))

                    messages.append(
                        {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "action": "tool_call",
                                    "tool": tool_name,
                                    "arguments": args,
                                    "reason": decision.get("reason", ""),
                                },
                                ensure_ascii=False,
                            ),
                        }
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"Azure MCP tool result ({tool_name}):\n{preview}\n"
                                "If this is sufficient, return final answer; otherwise call another tool."
                            ),
                        }
                    )

            raise AzureMCPError(f"No final answer after {max_iterations} iterations")
        except AzureMCPError:
            pass

    return _query_subscription_via_cli(
        openai_settings=openai_settings,
        question=question,
        command_base=command,
        subscription_id=subscription_id,
        max_iterations=max_iterations,
    )
