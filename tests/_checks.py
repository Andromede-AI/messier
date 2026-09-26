from collections import Counter, defaultdict
from datetime import date
import json
from pydantic import ValidationError
from messier.io import contains_credentials
from messier.models import Record, RecordType


class RecordAudit:
    def __init__(self, tasks, verifier_keys):
        self.tasks = tasks
        self.verifier_keys = verifier_keys
        self.invalid_count = 0
        self.invalid_samples = []
        self.record_keys = set()
        self.duplicate_keys = []
        self.missing_tasks = []
        self.missing_verifiers = []
        self.task_mismatches = []
        self.malformed_agents = []
        self.agent_identities = {}
        self.identity_mismatches = []
        self.model_dates = {}
        self.model_date_mismatches = []
        self.invalid_model_dates = []
        self.exposed_credentials = []
        self.direct_verifier_results = []
        self.cells = defaultdict(lambda: {"trial_seen": False, "trial_result": None, "verifiers": []})
        self.record_counts = Counter()
        self.trial_counts = Counter()
        self.trial_sums = Counter()
        self.trial_keys = set()

    def visit(self, row):
        benchmark = row.get("benchmark")
        task_id = row.get("task_id")
        task_key = (benchmark, task_id)
        if contains_credentials(json.dumps(row)) and len(self.exposed_credentials) < 3:
            self.exposed_credentials.append(task_key)

        try:
            Record.model_validate(row)
        except ValidationError as error:
            self.invalid_count += 1
            if len(self.invalid_samples) < 3:
                self.invalid_samples.append(
                    {
                        "benchmark": benchmark,
                        "task_id": task_id,
                        "errors": error.errors()[:3],
                    }
                )

        record_key = (
            benchmark,
            row.get("agent_id"),
            task_id,
            row.get("verifier_id"),
            row.get("record_type"),
            row.get("trial", 0),
        )
        if record_key in self.record_keys:
            if len(self.duplicate_keys) < 3:
                self.duplicate_keys.append(record_key)
        else:
            self.record_keys.add(record_key)

        task = self.tasks.get(task_key)
        if task is None and len(self.missing_tasks) < 3:
            self.missing_tasks.append(task_key)
        elif task is not None and row.get("benchmark_group") != task.get("benchmark_group") and len(self.task_mismatches) < 3:
            self.task_mismatches.append(task_key)

        verifier_id = row.get("verifier_id")
        if verifier_id and (benchmark, task_id, verifier_id) not in self.verifier_keys and len(self.missing_verifiers) < 3:
            self.missing_verifiers.append((benchmark, task_id, verifier_id))

        agent_id = row.get("agent_id") or ""
        model = row.get("agent_model")
        if (not agent_id or agent_id.strip() != agent_id or not model) and len(self.malformed_agents) < 3:
            self.malformed_agents.append((benchmark, task_id, agent_id))

        identity = (model, row.get("model_reasoning_effort"), row.get("agent_scaffold"))
        previous_identity = self.agent_identities.setdefault(agent_id, identity)
        if previous_identity != identity and len(self.identity_mismatches) < 3:
            self.identity_mismatches.append((agent_id, previous_identity, identity))

        model_date = row.get("model_date")
        if model_date:
            try:
                date.fromisoformat(model_date)
            except (TypeError, ValueError):
                if len(self.invalid_model_dates) < 3:
                    self.invalid_model_dates.append((model, model_date))

            previous_date = self.model_dates.setdefault(model, model_date)
            if previous_date != model_date and len(self.model_date_mismatches) < 3:
                self.model_date_mismatches.append((model, previous_date, model_date))

        if task_key in self.tasks:
            self.record_counts[task_key] += 1

        record_type = row.get("record_type")
        result = row.get("result")
        if task_key in self.tasks and record_type == RecordType.TRIAL_RESULT:
            self.trial_keys.add((benchmark, task_id, agent_id, row.get("trial")))
            if result is not None:
                self.trial_counts[task_key] += 1
                self.trial_sums[task_key] += int(result)

        if task_key not in self.tasks or record_type == RecordType.SOURCE_SUMMARY:
            return

        scoring_rule = self.tasks[task_key].get("scoring_rule")
        if record_type == RecordType.VERIFIER_RESULT and scoring_rule == "direct":
            if len(self.direct_verifier_results) < 3:
                self.direct_verifier_results.append(record_key)
            return
        if scoring_rule == "direct":
            return

        cell_key = (benchmark, task_id, agent_id, row.get("trial"))
        cell = self.cells[cell_key]
        if record_type == RecordType.TRIAL_RESULT:
            cell["trial_seen"] = True
            cell["trial_result"] = result
        elif record_type == RecordType.VERIFIER_RESULT:
            cell["verifiers"].append(result)

    def finalize(self):
        missing_trials = []
        invalid_all_pass = []
        all_pass_mismatches = []

        for key, cell in self.cells.items():
            verifier_results = cell["verifiers"]
            if verifier_results and not cell["trial_seen"]:
                if len(missing_trials) < 3:
                    missing_trials.append(key)
                continue

            task = self.tasks[(key[0], key[1])]
            if task.get("scoring_rule") != "all_pass" or not verifier_results:
                continue

            if any(result is not None and not 0 <= result <= 1 for result in verifier_results):
                if len(invalid_all_pass) < 3:
                    invalid_all_pass.append(key)

            complete = len(verifier_results) == task.get("n_verifiers") and None not in verifier_results
            if complete and cell["trial_result"] is not None:
                expected = int(all(result == 1 for result in verifier_results))
                if cell["trial_result"] != expected and len(all_pass_mismatches) < 3:
                    all_pass_mismatches.append(key)

        record_count_mismatches = []
        mean_mismatches = []
        saturation_mismatches = []
        unrecorded_mismatches = []

        for key, task in self.tasks.items():
            record_count = self.record_counts.get(key, 0)
            stored_count = task.get("n_records") or 0
            if record_count != stored_count and len(record_count_mismatches) < 3:
                record_count_mismatches.append((key, stored_count, record_count))

            stored_mean = task.get("mean_trial_result")
            trial_count = self.trial_counts.get(key, 0)
            if trial_count:
                mean = self.trial_sums.get(key, 0) / trial_count
                if (stored_mean is None or abs(mean - stored_mean) > 1e-6) and len(mean_mismatches) < 3:
                    mean_mismatches.append((key, stored_mean, mean))

            expected_saturated = stored_mean in {0.0, 1.0}
            if bool(task.get("is_saturated")) != expected_saturated and len(saturation_mismatches) < 3:
                saturation_mismatches.append(key)

            expected_unrecorded = record_count == 0
            if bool(task.get("is_unrecorded")) != expected_unrecorded and len(unrecorded_mismatches) < 3:
                unrecorded_mismatches.append(key)

        return {
            "invalid_count": self.invalid_count,
            "invalid_samples": self.invalid_samples,
            "duplicate_keys": self.duplicate_keys,
            "missing_tasks": self.missing_tasks,
            "missing_verifiers": self.missing_verifiers,
            "task_mismatches": self.task_mismatches,
            "malformed_agents": self.malformed_agents,
            "identity_mismatches": self.identity_mismatches,
            "model_date_mismatches": self.model_date_mismatches,
            "invalid_model_dates": self.invalid_model_dates,
            "exposed_credentials": self.exposed_credentials,
            "direct_verifier_results": self.direct_verifier_results,
            "missing_trials": missing_trials,
            "invalid_all_pass": invalid_all_pass,
            "all_pass_mismatches": all_pass_mismatches,
            "record_count_mismatches": record_count_mismatches,
            "mean_mismatches": mean_mismatches,
            "saturation_mismatches": saturation_mismatches,
            "unrecorded_mismatches": unrecorded_mismatches,
            "trial_keys": self.trial_keys,
        }
