def test_record_schema(record_audit):
    assert record_audit["invalid_count"] == 0, (
        f"invalid records: {record_audit['invalid_samples']}"
    )
    assert not record_audit["exposed_credentials"], (
        f"records contain possible credentials: {record_audit['exposed_credentials']}"
    )


def test_record_keys(record_audit):
    assert not record_audit["duplicate_keys"], (
        f"duplicate record keys: {record_audit['duplicate_keys']}"
    )


def test_record_references(record_audit):
    assert not record_audit["missing_tasks"], (
        f"records reference missing tasks: {record_audit['missing_tasks']}"
    )
    assert not record_audit["missing_verifiers"], (
        f"records reference missing verifiers: {record_audit['missing_verifiers']}"
    )
    assert not record_audit["task_mismatches"], (
        f"record fields disagree with their tasks: {record_audit['task_mismatches']}"
    )


def test_agent_identity(record_audit):
    assert not record_audit["malformed_agents"], (
        f"malformed agents: {record_audit['malformed_agents']}"
    )
    assert not record_audit["identity_mismatches"], (
        f"agent IDs map to different configurations: {record_audit['identity_mismatches']}"
    )
    assert not record_audit["model_date_mismatches"], (
        f"models have conflicting release dates: {record_audit['model_date_mismatches']}"
    )


def test_model_dates(record_audit):
    assert not record_audit["invalid_model_dates"], (
        f"invalid model dates: {record_audit['invalid_model_dates']}"
    )


def test_scoring_semantics(record_audit):
    assert not record_audit["direct_verifier_results"], (
        f"direct results are stored twice: {record_audit['direct_verifier_results']}"
    )
    assert not record_audit["missing_trials"], (
        f"verifier results lack trial results: {record_audit['missing_trials']}"
    )
    assert not record_audit["invalid_all_pass"], (
        f"all-pass verifier results fall outside [0, 1]: {record_audit['invalid_all_pass']}"
    )
    assert not record_audit["all_pass_mismatches"], (
        f"all-pass results disagree: {record_audit['all_pass_mismatches']}"
    )


def test_derived_task_values(record_audit):
    assert not record_audit["record_count_mismatches"], (
        f"task record counts disagree: {record_audit['record_count_mismatches']}"
    )
    assert not record_audit["mean_mismatches"], (
        f"task means disagree: {record_audit['mean_mismatches']}"
    )
    assert not record_audit["saturation_mismatches"], (
        f"task saturation flags disagree: {record_audit['saturation_mismatches']}"
    )
    assert not record_audit["unrecorded_mismatches"], (
        f"task unrecorded flags disagree: {record_audit['unrecorded_mismatches']}"
    )
