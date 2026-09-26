"""
Estimate agent ability and task difficulty from results of 0 or 1.

The input contains one row for each observed agent and task result. The 1PL
model assigns an ability value, ``theta``, to every agent and a difficulty
value, ``beta``, to every task. It predicts a result of 1 with probability
``sigmoid(theta - beta)``. An agent and task with equal values therefore have
a 50 percent success probability. Increasing ability raises that probability,
while increasing difficulty lowers it.

The 2PL model also assigns each task a positive discrimination value,
``alpha``. Its probability is ``sigmoid(alpha * (theta - beta))``. A larger
``alpha`` means that a small difference between ability and difficulty causes
a larger change in the predicted probability.

In ordinary PyTorch training, ``theta`` and ``beta`` could be learned as
individual parameters by minimizing binary prediction error. Pyro instead
represents every unknown value with a probability distribution. Before seeing
the results, ``theta``, ``beta``, and the logarithm of ``alpha`` are assumed to
follow zero-centered Normal distributions. Their population standard
deviations have Half-Normal priors and are learned from the results, allowing
the spread of each parameter to adapt to the corpus.

After seeing the results, the desired distribution for each unknown value is
called its posterior. Computing it exactly is impractical here, so ``guide()``
uses a Normal distribution with a learnable mean and standard deviation as an
approximation. SVI repeatedly samples values from these approximations, checks
how well they predict the observed results through ``model()``, and updates
their means and standard deviations using gradient descent.

The learned means are reported as ``theta`` and ``beta``. In 2PL, ``alpha`` is
the exponential of the learned mean for its logarithm. Posterior standard
deviations describe approximate uncertainty, with larger values indicating
less certainty.
"""

import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import torch
import torch.distributions.constraints as constraints
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam
from .models import IRTConfig, IRTModelType


class IRTModel:
    """
    Define the IRT probability model and posterior approximation.
    """

    def __init__(
        self,
        n_agents: int, # theta per agent
        n_tasks: int, # beta per task
        model_type: IRTModelType, # 1pl or 2pl
    ):
        self.n_agents = n_agents
        self.n_tasks = n_tasks
        self.model_type = model_type

    def model(self, agent_index, task_index, results):
        """
        Define the IRT priors and likelihood for the observed results.

        Pyro uses this function as the probabilistic model. It samples agent
        abilities and task parameters, then uses their indexes to assign a
        Bernoulli probability to each observed result of 0 or 1.

        :param agent_index: Agent position for each observed result.
        :param task_index: Task position for each observed result.
        :param results: Observed results (binary).
        """
        # scalar tensors inherit the result tensor's data type and device
        zero = results.new_tensor(0.0)
        one = results.new_tensor(1.0)

        theta_population_std = pyro.sample(name="sigma_theta", fn=dist.HalfNormal(one))
        beta_population_std = pyro.sample(name="sigma_b", fn=dist.HalfNormal(one))
        if self.model_type == IRTModelType.TWO_PARAM:
            log_alpha_population_std = pyro.sample(name="sigma_log_a", fn=dist.HalfNormal(one))

        # sample one independent ability value per agent
        with pyro.plate("agents", self.n_agents):
            theta = pyro.sample(name="theta_raw", fn=dist.Normal(zero, theta_population_std))

        # sample one independent difficulty value per task
        with pyro.plate("tasks", self.n_tasks):
            beta = pyro.sample(name="b", fn=dist.Normal(zero, beta_population_std))
            # 2PL has an additional discrimination parameter per task, log(a)
            if self.model_type == IRTModelType.TWO_PARAM:
                log_alpha = pyro.sample(name="log_a", fn=dist.Normal(zero, log_alpha_population_std))

        theta = theta - theta.mean()
        logits = theta[agent_index] - beta[task_index]
        if self.model_type == IRTModelType.TWO_PARAM:
            logits = log_alpha.exp()[task_index] * logits

        with pyro.plate("records", results.size(0)):
            pyro.sample(
                name="y",
                fn=dist.Bernoulli(logits=logits),
                obs=results
            )

    def guide(self, _agent_index, _task_index, results):
        """
        Define the learnable distributions used to approximate the posterior.

        Pyro's SVI fits the posterior mean and standard deviation of these
        Normal distributions against the priors and observations defined in
        :meth:`model`.

        :param _agent_index: Unused agent positions accepted because SVI passes
            the model arguments to both functions.
        :param _task_index: Unused task positions accepted for the same reason.
        :param results: Observed results of 0 or 1.
        """
        device = results.device

        theta_population_std = pyro.param(
            name="sigma_theta_q",
            init_tensor=torch.tensor(1.0, device=device),
            constraint=constraints.positive
        )
        beta_population_std = pyro.param(
            name="sigma_b_q",
            init_tensor=torch.tensor(1.0, device=device),
            constraint=constraints.positive
        )
        pyro.sample(name="sigma_theta", fn=dist.Delta(theta_population_std))
        pyro.sample(name="sigma_b", fn=dist.Delta(beta_population_std))

        if self.model_type == IRTModelType.TWO_PARAM:
            log_alpha_population_std = pyro.param(
                name="sigma_log_a_q",
                init_tensor=torch.tensor(1.0, device=device),
                constraint=constraints.positive
            )
            pyro.sample(name="sigma_log_a", fn=dist.Delta(log_alpha_population_std))

        # define one learned Normal approximation for each agent's ability
        theta_posterior_mean = pyro.param(name="loc_theta_raw", init_tensor=torch.zeros(self.n_agents, device=device))
        theta_posterior_std = pyro.param(
            name="scale_theta_raw",
            init_tensor=torch.ones(self.n_agents, device=device),
            constraint=constraints.positive # > 0
        )

        with pyro.plate("agents", self.n_agents):
            pyro.sample(name="theta_raw", fn=dist.Normal(theta_posterior_mean, theta_posterior_std))

        # define one learned Normal approximation for each task's difficulty
        beta_posterior_mean = pyro.param(name="loc_b", init_tensor=torch.zeros(self.n_tasks, device=device))
        beta_posterior_std = pyro.param(
            name="scale_b",
            init_tensor=torch.ones(self.n_tasks, device=device),
            constraint=constraints.positive
        )

        # 2PL also approximates each task's log-discrimination
        if self.model_type == IRTModelType.TWO_PARAM:
            log_alpha_posterior_mean = pyro.param(name="loc_log_a", init_tensor=torch.zeros(self.n_tasks, device=device))
            log_alpha_posterior_std = pyro.param(
                name="scale_log_a",
                init_tensor=torch.ones(self.n_tasks, device=device),
                constraint=constraints.positive
            )

        with pyro.plate("tasks", self.n_tasks):
            pyro.sample(name="b", fn=dist.Normal(beta_posterior_mean, beta_posterior_std))
            if self.model_type == IRTModelType.TWO_PARAM:
                pyro.sample(name="log_a", fn=dist.Normal(log_alpha_posterior_mean, log_alpha_posterior_std))


class IRT:
    """
    Bayesian 1PL or 2PL IRT over agents and tasks.

    Each agent ID, representing one model configuration and scaffold, gets one ability parameter.
    `data` is a DataFrame with: benchmark, task_id, agent_id, result.
    """

    def __init__(self, config: IRTConfig | None = None, **kwargs):
        self.config = config if config is not None else IRTConfig(**kwargs)
        self.task_ids = self.agents = None
        self.beta = self.theta = self.alpha = None
        self.beta_posterior_std = self.theta_posterior_std = self.log_alpha_posterior_std = None

    def fit(self, records):
        """
        Fit the configured IRT model.

        :param records: Data frame with ``benchmark``, ``task_id``, ``agent_id``,
                        and binary ``result`` columns.
        :return: This fitted instance.
        """
        indexed = self._index(records)
        print(
            f"joint fit ({self.config.model_type.value}) on "
            f"{len(indexed['results'])} trial results "
            f"({len(indexed['agents'])} agents, {len(indexed['tasks'])} tasks)"
        )

        self.theta, self.beta, self.alpha, self.theta_posterior_std, self.beta_posterior_std, self.log_alpha_posterior_std = self._train(indexed)
        self.task_ids = indexed["tasks"]
        self.agents = indexed["agents"]
        return self

    def refit_theta(
        self,
        records_subset: pd.DataFrame,
        epochs: int | None = None,
        lr: float | None = None,
    ) -> pd.DataFrame:
        """
        Estimate agent ability from a subset while keeping task difficulty fixed.

        The task difficulty values, ``beta``, come from the original model and
        are not updated. Only agent ability, ``theta``, is estimated again.

        :param records_subset: Results from the requested subset.
        :param epochs: Number of optimization epochs. Defaults to the config.
        :param lr: Learning rate. Defaults to the config.
        :return: One row per agent with columns ``agent_id`` and ``theta``.
        """
        if self.beta is None or self.task_ids is None:
            raise RuntimeError("must call fit() before refit_theta()")
        if self.config.model_type != IRTModelType.ONE_PARAM:
            raise ValueError("refit_theta supports 1PL fits only")

        epochs = epochs or self.config.epochs
        lr = lr or self.config.lr

        data = records_subset.copy()
        data["task_key"] = data["benchmark"] + "::" + data["task_id"].astype(str)
        fitted_tasks = set(self.task_ids)
        data = data[data["task_key"].isin(fitted_tasks)].reset_index(drop=True)
        if data.empty:
            raise ValueError("subset has no records for tasks in the fitted model")

        task_to_index = {task: index for index, task in enumerate(self.task_ids)}
        agents = sorted(data["agent_id"].unique())
        agent_to_index = {agent: index for index, agent in enumerate(agents)}
        n_agents = len(agents)

        beta = torch.tensor(self.beta, dtype=torch.float32)
        agent_index = torch.tensor(data["agent_id"].map(agent_to_index).values, dtype=torch.long)
        task_index = torch.tensor(data["task_key"].map(task_to_index).values, dtype=torch.long)
        results = torch.tensor(data["result"].values, dtype=torch.float)

        pyro.set_rng_seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        pyro.clear_param_store()

        # beta remains fixed, so this model and guide define theta only
        def model(agent_index, task_index, results):
            zero = results.new_tensor(0.0)
            one = results.new_tensor(1.0)
            theta_population_std = pyro.sample(name="sigma_theta", fn=dist.HalfNormal(one))
            with pyro.plate("agents", n_agents):
                theta = pyro.sample(name="theta_raw", fn=dist.Normal(zero, theta_population_std))

            logits = theta[agent_index] - beta[task_index]
            with pyro.plate("records", results.size(0)):
                pyro.sample(
                    name="y",
                    fn=dist.Bernoulli(logits=logits),
                    obs=results
                )

        def guide(_agent_index, _task_index, results):
            device = results.device
            theta_population_std = pyro.param(
                name="sigma_theta_q",
                init_tensor=torch.tensor(1.0, device=device),
                constraint=constraints.positive
            )
            pyro.sample(name="sigma_theta", fn=dist.Delta(theta_population_std))

            theta_posterior_mean = pyro.param(name="loc_theta_raw", init_tensor=torch.zeros(n_agents, device=device))
            theta_posterior_std = pyro.param(
                name="scale_theta_raw",
                init_tensor=torch.ones(n_agents, device=device),
                constraint=constraints.positive
            )
            with pyro.plate("agents", n_agents):
                pyro.sample(name="theta_raw", fn=dist.Normal(theta_posterior_mean, theta_posterior_std))

        inference = SVI(
            model,
            guide,
            ClippedAdam({"lr": lr, "betas": (0.9, 0.999), "clip_norm": 5.0}),
            loss=Trace_ELBO()
        )
        print(
            f"refit_theta on {len(data):,} results "
            f"({n_agents} agents, {data['task_key'].nunique():,} tasks, beta fixed)"
        )
        for epoch in range(1, epochs + 1):
            loss = inference.step(agent_index, task_index, results)
            if epoch == 1 or epoch % 500 == 0 or epoch == epochs:
                print(
                    f"  epoch {epoch:5d}  "
                    f"loss/trial_result={loss / len(results):.4f}"
                )

        theta = pyro.param("loc_theta_raw").detach().cpu().numpy()
        return pd.DataFrame({"agent_id": agents, "theta": theta})

    def task_difficulty(self):
        """
        Return the fitted difficulty (beta) and uncertainty (beta_posterior_std) for each task.
        """
        rows = [task.split("::", 1) for task in self.task_ids]
        difficulties = pd.DataFrame(rows, columns=["benchmark", "task_id"])
        difficulties["beta"] = self.beta
        difficulties["beta_posterior_std"] = self.beta_posterior_std
        return difficulties

    def save(self, path):
        """
        Save the fit.

        :param path: Destination directory.
        :return: Destination path.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        state = {
            "config": asdict(self.config),
            "tasks": self.task_ids,
            "agents": self.agents,
            "beta": self.beta.tolist() if self.beta is not None else None,
            "theta": self.theta.tolist() if self.theta is not None else None,
            "alpha": self.alpha.tolist() if self.alpha is not None else None,
            "beta_posterior_std": self.beta_posterior_std.tolist() if self.beta_posterior_std is not None else None,
            "theta_posterior_std": self.theta_posterior_std.tolist() if self.theta_posterior_std is not None else None,
            "log_alpha_posterior_std": self.log_alpha_posterior_std.tolist() if self.log_alpha_posterior_std is not None else None,
        }
        (path / "state.json").write_text(json.dumps(state, indent=2))
        return path

    @classmethod
    def load(cls, path):
        """
        Load parameters from a saved state.

        :param path: Directory containing ``state.json``.
        :return: Loaded IRT instance.
        """
        state = json.loads((Path(path) / "state.json").read_text())
        instance = cls.__new__(cls)
        instance.config = IRTConfig(**state["config"])
        instance.task_ids = state.get("tasks", state.get("items"))
        instance.agents = state["agents"]
        instance.beta = np.asarray(state["beta"]) if state["beta"] is not None else None
        instance.theta = np.asarray(state["theta"]) if state["theta"] is not None else None
        instance.alpha = np.asarray(state["alpha"]) if state["alpha"] is not None else None
        beta_posterior_std = state.get("beta_posterior_std", state.get("beta_scale"))
        theta_posterior_std = state.get("theta_posterior_std", state.get("theta_scale"))
        log_alpha_posterior_std = state.get("log_alpha_posterior_std", state.get("log_alpha_scale"))
        instance.beta_posterior_std = np.asarray(beta_posterior_std) if beta_posterior_std is not None else None
        instance.theta_posterior_std = np.asarray(theta_posterior_std) if theta_posterior_std is not None else None
        instance.log_alpha_posterior_std = np.asarray(log_alpha_posterior_std) if log_alpha_posterior_std is not None else None
        return instance

    def _index(self, records):
        """
        Prepare data for PyTorch.

        PyTorch uses integer indexes. This method assigns an integer to each
        agent and task, then returns the agent, task, and result for every input
        row. The benchmark name is included in the task ID so tasks from
        different benchmarks cannot share an index.

        :param records: Data frame containing binary trial results.
        :return: Agent and task names together with their integer indexes and results.
        """

        required = ["benchmark", "task_id", "agent_id", "result"]
        missing = [column for column in required if column not in records]
        if missing:
            raise ValueError(f"records missing required columns: {missing}")

        if ("record_type" in records and (records["record_type"] != "trial_result").any()):
            raise ValueError("IRT accepts only the final result for each trial. Use messier.trial_records")

        data = records.dropna(subset=required).copy()
        if data.empty:
            raise ValueError("no non-null trial results to fit")
        if not data["result"].isin([0, 1]).all():
            raise ValueError("result must contain only 0, 1, or null")

        data["task_key"] = data["benchmark"] + "::" + data["task_id"]
        agents = sorted(data["agent_id"].unique())
        tasks = sorted(data["task_key"].unique())

        agent_to_index = {agent: index for index, agent in enumerate(agents)}
        task_to_index = {task: index for index, task in enumerate(tasks)}

        return {
            "agents": agents,
            "tasks": tasks,
            "agent_index": torch.tensor(data["agent_id"].map(agent_to_index).values, dtype=torch.long),
            "task_index": torch.tensor(data["task_key"].map(task_to_index).values, dtype=torch.long),
            "results": torch.tensor(data["result"].values, dtype=torch.float),
        }

    def _train(self, indexed):
        config = self.config
        pyro.set_rng_seed(config.seed)
        torch.manual_seed(config.seed)
        pyro.clear_param_store()

        n_agents = len(indexed["agents"])
        probabilistic_model = IRTModel(
            n_agents=n_agents,
            n_tasks=len(indexed["tasks"]),
            model_type=config.model_type,
        )
        svi = SVI(
            model=probabilistic_model.model,
            guide=probabilistic_model.guide,
            optim=ClippedAdam({"lr": config.lr, "betas": (0.9, 0.999), "clip_norm": 5.0}),
            loss=Trace_ELBO()
        )
        n_trial_results = indexed["results"].size(0)
        for epoch in range(1, config.epochs + 1):
            loss = svi.step(
                indexed["agent_index"],
                indexed["task_index"],
                indexed["results"],
            )
            loss_per_result = loss / n_trial_results

            # print periodically
            if epoch == 1 or epoch % 500 == 0 or epoch == config.epochs:
                print(
                    f"  epoch {epoch:5d}  "
                    f"loss/trial_result={loss_per_result:.4f}"
                )

        beta = pyro.param("loc_b").detach()
        beta_posterior_std = pyro.param("scale_b").detach().cpu().numpy()

        theta = pyro.param("loc_theta_raw").detach()
        theta_raw_posterior_std = pyro.param("scale_theta_raw").detach()
        theta_raw_variance = theta_raw_posterior_std.square()
        theta_posterior_variance = theta_raw_variance * (1 - 2 / n_agents) + theta_raw_variance.sum() / n_agents**2
        theta_posterior_std = theta_posterior_variance.clamp_min(0).sqrt().cpu().numpy()

        if config.model_type == IRTModelType.TWO_PARAM:
            alpha = pyro.param("loc_log_a").detach().exp().cpu().numpy()
            log_alpha_posterior_std = pyro.param("scale_log_a").detach().cpu().numpy()
        else:
            alpha = np.ones(len(indexed["tasks"]), dtype=np.float32)
            log_alpha_posterior_std = None

        return (
            (theta - theta.mean()).cpu().numpy(),
            beta.cpu().numpy(),
            alpha,
            theta_posterior_std,
            beta_posterior_std,
            log_alpha_posterior_std,
        )
