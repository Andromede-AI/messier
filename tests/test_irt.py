import numpy as np
import pandas as pd
import pyro
from messier import IRT

RESULTS = pd.DataFrame(
    {
        "benchmark": ["test"] * 6,
        "task_id": ["easy", "hard"] * 3,
        "agent_id": ["strong", "strong", "middle", "middle", "weak", "weak"],
        "result": [1, 1, 1, 0, 0, 0],
    }
)


def test_fit_centers_uncertainty_and_saves(tmp_path):
    irt = IRT(model_type="1pl", epochs=20).fit(RESULTS)
    raw_std = pyro.param("scale_theta_raw").detach().cpu().numpy()
    raw_variance = raw_std**2
    n_agents = len(irt.agents)
    expected_std = np.sqrt(raw_variance * (1 - 2 / n_agents) + raw_variance.sum() / n_agents**2)

    assert np.isclose(irt.theta.mean(), 0)
    assert np.allclose(irt.theta_posterior_std, expected_std)

    irt.save(tmp_path)
    loaded = IRT.load(tmp_path)
    assert np.allclose(loaded.theta, irt.theta)
    assert np.allclose(loaded.theta_posterior_std, irt.theta_posterior_std)


def test_2pl_estimates_positive_discrimination():
    irt = IRT(model_type="2pl", epochs=20).fit(RESULTS)
    assert np.all(irt.alpha > 0)
    assert irt.log_alpha_posterior_std is not None


def test_refit_theta_keeps_beta_fixed():
    irt = IRT(model_type="1pl", epochs=100, lr=0.05)
    irt.task_ids = ["test::task"]
    irt.beta = np.array([0.0])
    results = pd.DataFrame(
        {
            "benchmark": ["test"] * 10,
            "task_id": ["task"] * 10,
            "agent_id": ["agent"] * 10,
            "result": [1] * 10,
        }
    )

    refitted = irt.refit_theta(results)
    assert refitted.loc[0, "theta"] > 0
