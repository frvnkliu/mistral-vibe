from __future__ import annotations

from collections.abc import AsyncIterator
import json

import pytest

pytest.importorskip("mistralai_vibe_local_harness.runtime")

from mistralai_vibe_local_harness.protocol import (
    DEPRECATED_RUNTIME_BUILTIN_TOOL_NAMES,
    RUNTIME_BUILTIN_TOOL_NAMES,
    RustBashTimeoutSettings,
    RustContextSettings,
    RustDisabledCompactionPolicy,
    RustDisabledLargeOutputPolicy,
    RustDisabledRuntimeToolFeature,
    RustEnabledRuntimeToolFeature,
    RustFixedImageCostRule,
    RustHarnessCapabilitySet,
    RustHarnessConfig,
    RustHarnessSettings,
    RustModelFacts,
    RustProgrammaticToolSettings,
    RustSystemMessage,
    RustTextContentBlock,
    RustTokenUsage,
    RustToolCallAction,
    RustToolGroupDefinition,
    RustToolMessage,
    RustToolSettings,
    RustTurnSettings,
    RustUnixCommandEnvironment,
)
from mistralai_vibe_local_harness.runtime import (
    InMemoryHarnessRuntime,
    ModelAssistantContentDelta,
    ModelCompletionFinished,
    ModelRequest,
    ModelStreamItem,
    ModelToolCallArgumentsDelta,
    ModelToolCallStartedDelta,
    RuntimeBuiltinToolHandlers,
    ToolExecutionResult,
)


def _harness_config() -> RustHarnessConfig:
    return RustHarnessConfig(
        task_id="test-non-object-arguments",
        settings=RustHarnessSettings(
            turn=RustTurnSettings(),
            context=RustContextSettings(
                compaction=RustDisabledCompactionPolicy(),
                model_facts=RustModelFacts(
                    context_window=128_000,
                    compaction_threshold=64_000,
                    compaction_threshold_max=96_000,
                    input_modalities=["text"],
                    image_cost=RustFixedImageCostRule(tokens=100),
                ),
            ),
            tools=RustToolSettings(
                programmatic=RustProgrammaticToolSettings(
                    max_effects=8,
                    max_operations=16,
                    max_elapsed_ms=10_000,
                    max_sleep_ms=1_000,
                ),
                subagents=RustDisabledRuntimeToolFeature(),
                background_processes=RustEnabledRuntimeToolFeature(),
                command_environment=RustUnixCommandEnvironment(),
                large_output=RustDisabledLargeOutputPolicy(),
                bash_timeout=RustBashTimeoutSettings(
                    default_seconds=30, max_seconds=60
                ),
            ),
        ),
        capabilities=RustHarnessCapabilitySet(
            tool_groups=[
                RustToolGroupDefinition.model_validate({
                    "name": "demo",
                    "description": "Demo tools",
                    "tools": [
                        {
                            "name": "echo",
                            "description": "Echo input",
                            "input_schema": {},
                            "exposure": "direct_and_programmatic",
                        }
                    ],
                })
            ]
        ),
    )


def _runtime_with_tool_call(
    tool_name: str, arguments_json: str, requests: list[ModelRequest]
) -> tuple[InMemoryHarnessRuntime, list[str]]:
    dispatched: list[str] = []

    usage = RustTokenUsage(input_tokens=1, output_tokens=1, total_tokens=2)

    async def model_handler(request: ModelRequest) -> AsyncIterator[ModelStreamItem]:
        requests.append(request)
        if request.iteration == 0:
            yield ModelToolCallStartedDelta(call_id="call_1", name=tool_name)
            yield ModelToolCallArgumentsDelta(call_id="call_1", json=arguments_json)
            yield ModelCompletionFinished(finish_reason="tool_call", usage=usage)
        else:
            yield ModelAssistantContentDelta(
                content=[RustTextContentBlock(text="done")]
            )
            yield ModelCompletionFinished(finish_reason="stop", usage=usage)

    def record_builtin(action: RustToolCallAction) -> ToolExecutionResult:
        dispatched.append(action.action_id)
        return ToolExecutionResult(output={})

    builtin_handlers: RuntimeBuiltinToolHandlers = {
        name: record_builtin
        for name in RUNTIME_BUILTIN_TOOL_NAMES
        if name not in DEPRECATED_RUNTIME_BUILTIN_TOOL_NAMES
    }
    runtime = InMemoryHarnessRuntime(
        _harness_config(),
        initial_history=[],
        model_handler=model_handler,
        runtime_builtin_handlers=builtin_handlers,
        provided_tool_handler=lambda action: (
            dispatched.append(f"{action.call.group_name}.{action.call.tool_name}")
            or ToolExecutionResult(output={})
        ),
    )
    return runtime, dispatched


@pytest.mark.parametrize(
    ("call_expr", "expected_message"),
    [
        (
            'tools.file_system.read_file("notes.txt")',
            'Invalid arguments for tools.file_system.read_file: "notes.txt" is not of type "object"',
        ),
        (
            'tools.process.start("echo ready")',
            'Invalid arguments for tools.process.start: "echo ready" is not of type "object"',
        ),
        (
            "tools.file_system.bash(42)",
            'Invalid arguments for tools.file_system.bash: 42 is not of type "object"',
        ),
        (
            'tools.demo.echo(["item"])',
            'Invalid arguments for tools.demo.echo: ["item"] is not of type "object"',
        ),
    ],
)
@pytest.mark.parametrize("caught", [False, True], ids=["uncaught", "caught"])
@pytest.mark.asyncio
async def test_run_typescript_non_object_tool_arguments_return_tool_error(
    call_expr: str, expected_message: str, caught: bool
) -> None:
    if caught:
        code = (
            "async function main() {\n"
            f"  try {{ await {call_expr}; }}\n"
            "  catch (err) { return err instanceof Error ? err.message : String(err); }\n"
            '  return "unreachable";\n'
            "}"
        )
    else:
        code = f"async function main() {{ return await {call_expr}; }}"

    requests: list[ModelRequest] = []
    runtime, dispatched = _runtime_with_tool_call(
        "run_typescript", json.dumps({"code": code}), requests
    )

    output = await runtime.run_turn("run check")

    assert output == [RustTextContentBlock(text="done")]
    assert dispatched == []
    tool_messages = [
        msg for msg in requests[-1].messages if isinstance(msg, RustToolMessage)
    ]
    assert len(tool_messages) == 1
    assert tool_messages[0].outcome == ("success" if caught else "failure")
    tool_text = "".join(
        block.text
        for block in tool_messages[0].content
        if isinstance(block, RustTextContentBlock)
    )
    if caught:
        assert json.loads(tool_text) == expected_message
    else:
        assert json.dumps(expected_message) in tool_text


@pytest.mark.asyncio
async def test_direct_non_object_arguments_rejected_even_with_open_schema() -> None:
    requests: list[ModelRequest] = []
    runtime, dispatched = _runtime_with_tool_call("echo", json.dumps("hello"), requests)

    output = await runtime.run_turn("call echo")

    assert output == [RustTextContentBlock(text="done")]
    assert dispatched == []
    tool_messages = [
        msg for msg in requests[-1].messages if isinstance(msg, RustToolMessage)
    ]
    assert len(tool_messages) == 1
    assert tool_messages[0].outcome == "failure"
    tool_text = "".join(
        block.text
        for block in tool_messages[0].content
        if isinstance(block, RustTextContentBlock)
    )
    payload = json.loads(tool_text)
    assert payload["result"]["error"]["name"] == "InvalidToolCall"
    assert '"hello" is not of type "object"' in payload["result"]["error"]["message"]

    system_messages = [
        msg for msg in requests[0].messages if isinstance(msg, RustSystemMessage)
    ]
    system_text = "\n".join(
        block.text for msg in system_messages for block in msg.content
    )
    assert "`tools.process.start({ command })`" in system_text
