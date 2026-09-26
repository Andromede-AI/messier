# General AgentBench Description

[General AgentBench](https://github.com/cxcscmu/General-AgentBench) evaluates models with one shared scaffold across search, reasoning, and tool-use benchmarks through a Model Context Protocol (MCP) interface. MESSIER imports tasks and evaluation results from its released [dataset](https://huggingface.co/datasets/cx-cmu/agent_trajectories) for MathHay, BrowseComp, WebVoyager, and MCPBench.

These runs use General AgentBench's scaffold and tool interface, followed by each benchmark's evaluator. They may therefore differ from runs made with the original benchmark interface. For example, the imported WebVoyager runs use a web-search tool rather than browser controls.

| | MathHay | BrowseComp | WebVoyager | MCPBench |
| --- | --- | --- | --- | --- |
| Task and environment | A long-context math question with relevant and distractor documents in the prompt | A fact-finding question and live web search | An online shopping, travel, academic, or other information request with live web search | An instruction and tools exposed by an MCP server |
| Environment state | Not stateful. The supplied documents do not change. | Live search results. Tool calls do not change the web. | Live search results. Tool calls do not change the web. | Data and behavior maintained by the task's MCP server |
| Action space | Answer-submission calls and text responses | Web-search calls and text responses | Web-search calls and text responses | MCP tool calls and text responses |
| Verifier | Accepts an exact numerical match or an equivalent answer accepted by an LLM judge | An LLM judge compares the answer with the gold answer | An LLM judge assesses whether the final answer completes the task | An LLM judge assigns six numerical verifier results. Their mean must be at least 5 out of 10 |
| Gold answer | The expected numerical answer | The expected text answer | None | None |

### MathHay example

```python
Task(
    benchmark="mathhay",
    task_id="mathhay_3s3d_000",
    task_description="Long-Context Documents: ... Calculate the product of Nvidia's closing stock price ...",
    environment_metadata={
        "action_space": {
            "types": ["tool_calls", "text"],
            "tool_registry": {"tools": [{"name": "mathhay__submit_answer", ...}]},
        },
        "environment_state": {"type": "none", "access": "none"},
    },
    task_metadata={"extra_context": {"system_prompt": "You are a helpful AI agent ..."}},
    gold_answer=193488.39484000002,
    verifiers=[{"type": "exact_match", "mode": "final"}],
    source_platform="huggingface",
    data_provider="general_agentbench",
)

Record(
    benchmark="mathhay",
    task_id="mathhay_3s3d_000",
    agent_id="deepseek-r1+general-agentbench",
    agent_model="deepseek-r1",
    agent_scaffold="general-agentbench",
    trial=0,
    result=0,
    metadata={
        "source_reward": 0.0,
        "numerical_match": False,
        "llm_judge": "No",
    },
    source_platform="huggingface",
    data_provider="general_agentbench",
)
```

### BrowseComp example

```python
Task(
    benchmark="browsecomp",
    task_id="2",
    task_description="You should pay attention to the format of your output. ... Question: A video game has been mentioned in two papers ...",
    environment_metadata={
        "action_space": {
            "types": ["tool_calls", "text"],
            "tool_registry": {"tools": [{"name": "search__web_search", ...}]},
        },
        "environment_state": {"type": "live_web", "access": "read_only"},
    },
    task_metadata={"extra_context": {"system_prompt": "You are a helpful AI agent ..."}},
    gold_answer="Michael",
    verifiers=[{"type": "llm_judge", "mode": "final"}],
    source_platform="huggingface",
    data_provider="general_agentbench",
)

Record(
    benchmark="browsecomp",
    task_id="2",
    agent_id="deepseek-r1+general-agentbench",
    agent_model="deepseek-r1",
    agent_scaffold="general-agentbench",
    trial=0,
    result=0,
    metadata={
        "source_reward": 0.0,
        "num_turns": 19,
    },
    source_platform="huggingface",
    data_provider="general_agentbench",
)
```

### WebVoyager example

```python
Task(
    benchmark="webvoyager",
    task_id="136",
    task_description="You should pay attention to the format of your output. ... Question: Search for women's golf polos in m size ...",
    environment_metadata={
        "action_space": {
            "types": ["tool_calls", "text"],
            "tool_registry": {"tools": [{"name": "search__web_search", ...}]},
        },
        "environment_state": {"type": "live_web", "access": "read_only"},
    },
    task_metadata={"extra_context": {"system_prompt": "You are a helpful AI agent ..."}},
    gold_answer=None,
    verifiers=[{"type": "llm_judge", "mode": "final"}],
    source_platform="huggingface",
    data_provider="general_agentbench",
)

Record(
    benchmark="webvoyager",
    task_id="136",
    agent_id="deepseek-r1+general-agentbench",
    agent_model="deepseek-r1",
    agent_scaffold="general-agentbench",
    trial=0,
    result=0,
    metadata={
        "source_reward": 0.0,
        "num_turns": 7,
    },
    source_platform="huggingface",
    data_provider="general_agentbench",
)
```

### MCPBench example

```python
Task(
    benchmark="mcpbench",
    task_id="mcpbench_0",
    task_description="Hey, I'm working on this new dashboard ...\n\nTask specification:\nPerform a comparative audit of search endpoints ...",
    environment_metadata={
        "action_space": {
            "types": ["tool_calls", "text"],
            "tool_registry": {
                "tools": [
                    {"name": "OpenAPI_Explorer__getApiOverview", ...},
                    {"name": "OpenAPI_Explorer__getApiOperation", ...},
                ]
            },
        },
        "environment_state": {"type": "in_memory", "access": "read_write"},
    },
    task_metadata={"extra_context": {"system_prompt": "You are a helpful AI agent ..."}},
    gold_answer=None,
    verifiers=[
        {"type": "llm_judge", "mode": "sequential", "role": "task_fulfillment"},
        {"type": "llm_judge", "mode": "sequential", "role": "grounding"},
        {"type": "llm_judge", "mode": "sequential", "role": "tool_appropriateness"},
        {"type": "llm_judge", "mode": "sequential", "role": "parameter_accuracy"},
        {"type": "llm_judge", "mode": "sequential", "role": "dependency_awareness"},
        {"type": "llm_judge", "mode": "sequential", "role": "parallelism_and_efficiency"},
    ],
    source_platform="huggingface",
    data_provider="general_agentbench",
)

Record(
    benchmark="mcpbench",
    task_id="mcpbench_0",
    agent_id="deepseek-r1+general-agentbench",
    agent_model="deepseek-r1",
    agent_scaffold="general-agentbench",
    trial=0,
    result=0,
    metadata={
        "source_reward": 3.966666666666666,
    },
    source_platform="huggingface",
    data_provider="general_agentbench",
)
```

## Interaction examples

These are the same source trials shown above. Ellipses shorten long documents and tool responses.

### MathHay (`mathhay_3s3d_000`)

**Agent** = **Model:** `deepseek-r1` + **Scaffold:** `general-agentbench`

| | Interaction |
| --- | --- |
| 👤 **User** | [Long-context documents omitted] Calculate the product of Nvidia's closing stock price on May 31, 2024, its Q3 2024 revenue in billions, and Tesla's volatility as of October 29, 2024. |
| 🤖 **Agent** | Returns `{"reasoning": "...", "answer": 1934.8839484}`. |
| 🟥 **Trial result** | `0`, because `1934.8839484` does not match the expected answer `193488.39484000002`. |

### BrowseComp (`2`)

**Agent** = **Model:** `deepseek-r1` + **Scaffold:** `general-agentbench`

| | Interaction |
| --- | --- |
| 👤 **User** | A video game has been mentioned in two papers published between 2020 and 2022. ... What was the first name of the author of the PhD thesis? |
| 🤖 **Agent** | Calls `search__web_search(query="British universities founded as art schools in 19th century")`. |
| 🌐 **Env** | `A search for 'British universities founded as art schools in 19th century' found 9 results: ...` |
| 🤖 **Agent** | Makes eight further web searches and returns `<answer>Laurent</answer>`. |
| 🟥 **Trial result** | `0`, because `Laurent` does not match the gold answer `Michael`. |

### WebVoyager (`136`)

**Agent** = **Model:** `deepseek-r1` + **Scaffold:** `general-agentbench`

| | Interaction |
| --- | --- |
| 👤 **User** | Search for women's golf polos in m size, priced between 50 to 75 dollars, and save the lowest priced among results. |
| 🤖 **Agent** | Calls `search__web_search(query="women's golf polos size M $50 to $75")`. |
| 🌐 **Env** | `A search for 'women's golf polos size M $50 to $75' found 10 results: ...` |
| 🤖 **Agent** | Searches again and returns `The lowest priced among results is $49.99.` |
| 🟥 **Trial result** | `0`, because the source evaluator does not accept the response. |

### MCPBench (`mcpbench_0`)

**Agent** = **Model:** `deepseek-r1` + **Scaffold:** `general-agentbench`

| | Interaction |
| --- | --- |
| 👤 **User** | Hey, I'm working on this new dashboard that pulls search results from three different services - one for AI stuff, one for code hosting, and one for edge networking - and I'm scratching my head over how each handles pagination. ... |
| 🤖 **Agent** | Calls `OpenAPI_Explorer__getApiOverview` for `openai`, `github`, and `cloudflare`. |
| 🌐 **Env** | Returns the OpenAI and GitHub API overviews, then `Error: The SLOP found is too large to process with this MCP. Please try a different OpenAPI.` |
| 🤖 **Agent** | Calls `OpenAPI_Explorer__getApiOperation` for selected operations, then returns `Based on the API documentation analysis, here's a detailed breakdown of pagination approaches for each service: ...` |
| 🟥 **Trial result** | `0`, because the mean of the six verifier results is `3.9667`, below the threshold of `5`. |

## References

- Imported results: [General AgentBench dataset](https://huggingface.co/datasets/cx-cmu/agent_trajectories)
- Data provider: [General AgentBench](https://github.com/cxcscmu/General-AgentBench)
- Original benchmarks: [MathHay](https://arxiv.org/abs/2410.04698), [BrowseComp](https://openai.com/index/browsecomp/), [WebVoyager](https://github.com/MinorJerry/WebVoyager), and [MCP-Bench](https://github.com/Accenture/mcp-bench)
