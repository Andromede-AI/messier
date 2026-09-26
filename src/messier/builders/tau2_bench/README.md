# Tau2-bench Description

[Tau2-bench](https://github.com/sierra-research/tau2-bench) evaluates conversational agents in environments where both the agent and a user can communicate and take actions. MESSIER imports task definitions, trial results, verifier results, and trajectories for airline, retail, telecom, and banking knowledge from seven Sierra leaderboard submissions.


| | Airline | Retail | Telecom | Banking knowledge |
| --- | --- | --- | --- | --- |
| Task and environment | Assist an airline customer using a flight-booking system | Assist a retail customer using an order and account system | Assist a telecom customer using an account and service system | Answer a bank customer using policy documents and customer-service tools |
| Environment state | Users, flights, reservations, payments, baggage, and insurance that tools can change | Users, products, orders, payments, returns, exchanges, and refunds that tools can change | Customer accounts, phone lines, plans, devices, bills, and support information that tools can change | Bank documents and task-specific customer data. Customer-side tools can change applications, referrals, or transactions |
| Action space | Airline service tool calls and text responses | Retail service tool calls and text responses | Telecom service tool calls and text responses | Retrieval or customer-service tool calls and text responses. Some configurations also provide shell access for document search |
| Verifier | Checks the final booking state and required communication | Checks the final retail state and, where required, response content | Checks assertions about the final state and, for some tasks, the action sequence | Checks the final state, action sequence, or response content specified by the task |
| Scoring rule | All verifier results listed by the task must equal `1` | All verifier results listed by the task must equal `1` | All verifier results listed by the task must equal `1` | All verifier results listed by the task must equal `1` |
| Gold answer | Evaluation criteria containing the expected state and communication | Evaluation criteria containing the expected state and response assertions | Evaluation criteria containing the expected state and actions | Evaluation criteria containing the expected state, actions, or response assertions |


```python
Task(
    benchmark="tau2-bench",
    task_id="airline.46",
    environment_id="airline",
    environment_description=(
        "An airline customer-service system containing users, flights, "
        "reservations, payments, baggage, and travel insurance. ..."
    ),
    task_description=(
        "You want to get a refund for the insurance you purchased for your "
        "flight but you don't want to cancel the flight itself. ..."
    ),
    environment_metadata={
        "action_space": {
            "types": ["tool_calls", "text"],
            "description": (
                "Tool calls for reading and updating the airline system, "
                "together with text responses to the customer."
            ),
        },
        "environment_state": {
            "type": "in_memory",
            "access": "read_write",
            "ref": "tau2-bench://domain/airline",
        },
    },
    gold_answer={
        "nl_assertions": ["Agent does not cancel insurance or offer a refund."],
        "reward_basis": ["DB", "COMMUNICATE"],
        "...": "...",
    },
    verifiers=[
        {"type": "exact_match", "mode": "final", "name": "db"},
        {"type": "script", "mode": "sequential", "name": "communicate"},
    ],
    scoring_rule="all_pass",
    source_platform="github",
)

Record(
    benchmark="tau2-bench",
    task_id="airline.46",
    agent_id="gpt-5-2 (high)+tau2-bench",
    agent_model="gpt-5-2",
    model_reasoning_effort="high",
    agent_scaffold="tau2-bench",
    trial=1,
    result=1,
    metadata={
        "source_reward": 1.0,
        "reward_breakdown": {"DB": 1.0, "COMMUNICATE": 1.0},
        "termination_reason": "user_stop",
        "...": "...",
    },
    source_platform="github",
)
```

## Interaction examples

### Airline (`airline.46`) example

**Agent** = **Model:** `gpt-5-2` + **Reasoning effort:** `high` + **Scaffold:** `tau2-bench`

| | Interaction |
| --- | --- |
| 👤 **User** | Requests a refund for travel insurance without cancelling the flight. |
| 🤖 **Agent** | Calls `transfer_to_human_agents(...)` because the available policy and tools do not permit removing the insurance separately. |
| 🌐 **Env** | Returns `Transfer successful`. |
| 🤖 **Agent** | Tells the customer that the transfer is in progress. |
| 👤 **User** | Returns `###TRANSFER###`. |
| 🟥 **Trial result** | `1`, because both the final database and communication verifier results equal `1`. |

### Banking knowledge (`banking_knowledge.task_002`) example

**Agent** = **Model:** `qwen-3-5-397b-a17b` + **Reasoning effort:** `enabled` + **Scaffold:** `tau2-bench`

| | Interaction |
| --- | --- |
| 👤 **User** | Asks which Rho-Bank card offers the highest cash back and what its annual fee is. |
| 🤖 **Agent** | Calls `KB_search(query="credit card cash back rewards highest rate annual fee")`. |
| 🌐 **Env** | Returns matching bank documents, including the Platinum Rewards Card's cash-back rate and annual fee. |
| 🤖 **Agent** | Recommends the Platinum Rewards Card. |
| 👤 **User** | Calls `apply_for_credit_card(card_type="Platinum Rewards Card", ...)`. |
| 🌐 **Env** | Confirms that the application was submitted. |
| 🟥 **Trial result** | `1`, because the final application state matches the expected state. |

## References

- Imported results: [Tau2-bench leaderboard submissions](https://github.com/sierra-research/tau2-bench/tree/main/web/leaderboard/public/submissions)
- Original benchmark: [Tau2-bench](https://github.com/sierra-research/tau2-bench) and its [paper](https://arxiv.org/abs/2506.07982)
- Banking knowledge domain: [Tau-Knowledge](https://arxiv.org/abs/2603.04370)
