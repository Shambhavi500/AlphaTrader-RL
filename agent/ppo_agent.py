"""
AlphaTrader-RL | Part 3: PPO Agent Wrapper
Uses Stable-Baselines3 PPO with MlpPolicy.
"""
import os
import numpy as np
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from environment.trading_env import TradingEnv

os.makedirs("logs/tensorboard", exist_ok=True)
os.makedirs("models", exist_ok=True)

logging.basicConfig(
    filename="logs/ppo_agent.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("PPOAgent")


def _make_env(env: TradingEnv, monitor_path: str = "logs/monitor"):
    """Wraps a TradingEnv in Monitor + DummyVecEnv as required by SB3."""
    os.makedirs(monitor_path, exist_ok=True)

    def _init():
        return Monitor(env, filename=os.path.join(monitor_path, "monitor.csv"))

    return DummyVecEnv([_init])


class PPOAgent:
    """
    Stable-Baselines3 PPO wrapper tuned for financial RL.

    Parameters
    ----------
    env : TradingEnv
        A freshly-constructed (or reset) trading environment.
    learning_rate : float
    total_timesteps : int
        Default timesteps used when ``train()`` is called without an argument.
    tensorboard_log : str
        Directory for TensorBoard event files.
    verbose : int
        SB3 verbosity (0=silent, 1=info).
    """

    DEFAULT_POLICY_KWARGS = dict(net_arch=[128, 128])

    def __init__(
        self,
        env: TradingEnv,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        ent_coef: float = 0.01,
        clip_range: float = 0.2,
        tensorboard_log: str = "logs/tensorboard",
        verbose: int = 1,
    ):
        self._base_env = env
        self._vec_env = _make_env(env)

        # Only pass tensorboard_log when a path is given AND the package is available.
        tb_log = None
        if tensorboard_log:
            try:
                import tensorboard  # noqa: F401
                os.makedirs(tensorboard_log, exist_ok=True)
                tb_log = tensorboard_log
            except ImportError:
                logger.warning(
                    "tensorboard not installed — TensorBoard logging disabled. "
                    "Install with: pip install tensorboard"
                )

        self.model = PPO(
            policy="MlpPolicy",
            env=self._vec_env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            clip_range=clip_range,
            policy_kwargs=self.DEFAULT_POLICY_KWARGS,
            tensorboard_log=tb_log,
            verbose=verbose,
        )
        logger.info("PPOAgent initialised — obs_space=%s  tb_log=%s", env.observation_space.shape, tb_log)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, total_timesteps: int = 200_000, tb_log_name: str = "PPO_AlphaTrader"):
        """Train the agent for ``total_timesteps`` environment steps."""
        logger.info("Training started: total_timesteps=%d", total_timesteps)
        self.model.learn(
            total_timesteps=total_timesteps,
            tb_log_name=tb_log_name,
            reset_num_timesteps=False,
        )
        logger.info("Training complete.")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, env: TradingEnv, n_episodes: int = 3) -> float:
        """
        Run ``n_episodes`` deterministic episodes on *env*.

        Returns
        -------
        float
            Mean Sharpe ratio across episodes (from ``env.summary()``).
        """
        sharpe_scores = []
        for ep in range(n_episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = env.step(int(action))
                done = terminated or truncated

            summary = env.summary()
            sharpe = summary.get("sharpe_ratio", 0.0)
            sharpe_scores.append(sharpe)

            print(
                f"  [Ep {ep + 1}/{n_episodes}] "
                f"Return={summary['total_return_pct']:+.2f}%  "
                f"MaxDD={summary['max_drawdown_pct']:.2f}%  "
                f"Sharpe={sharpe:.3f}  "
                f"FinalValue=₹{summary['final_portfolio_value']:,.2f}"
            )
            logger.info("Episode %d summary: %s", ep + 1, summary)

        mean_sharpe = float(np.mean(sharpe_scores))
        logger.info("Evaluation mean Sharpe=%.4f over %d episodes", mean_sharpe, n_episodes)
        return mean_sharpe

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str = "models/best_model"):
        """Save the model to *path* (.zip appended by SB3 automatically)."""
        self.model.save(path)
        logger.info("Model saved to %s.zip", path)
        print(f"  [✓] Model saved → {path}.zip")

    @classmethod
    def load(cls, path: str, env: TradingEnv) -> "PPOAgent":
        """Load a previously saved model from *path*."""
        instance = object.__new__(cls)
        instance._base_env = env
        instance._vec_env = _make_env(env)
        instance.model = PPO.load(path, env=instance._vec_env)
        logger.info("Model loaded from %s", path)
        return instance

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def get_config(self) -> dict:
        """Return a dict of the agent's key hyperparameters."""
        return {
            "policy": "MlpPolicy",
            "learning_rate": self.model.learning_rate,
            "n_steps": self.model.n_steps,
            "batch_size": self.model.batch_size,
            "n_epochs": self.model.n_epochs,
            "gamma": self.model.gamma,
            "ent_coef": self.model.ent_coef,
            "clip_range": self.model.clip_range,
            "net_arch": self.DEFAULT_POLICY_KWARGS["net_arch"],
        }
