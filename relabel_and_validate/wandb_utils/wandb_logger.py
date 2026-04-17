"""
Lightweight WandB logging utilities for LPQLD training.
"""
import wandb


def wandb_log(metrics: dict, step: int = None):
    """Log a dict of metrics to WandB."""
    if wandb.run is not None:
        wandb.log(metrics, step=step)


def wandb_hyperparam_log(args):
    """Log all hyperparameters from an argparse Namespace to WandB config."""
    if wandb.run is not None:
        wandb.config.update(vars(args), allow_val_change=True)


def wandb_terminal_log(text: str, step: int = None):
    """Log a plain-text string as a WandB table row (useful for terminal output)."""
    if wandb.run is not None:
        wandb.log({"terminal": wandb.Html(f"<pre>{text}</pre>")}, step=step)
