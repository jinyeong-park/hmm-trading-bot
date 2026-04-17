"""
main.py — Entry point for regime-trader.

Modes (controlled by CLI args or config):
  live    : Connect to Alpaca, run the trading loop indefinitely.
  paper   : Same as live but forces paper_trading=True.
  backtest: Run walk-forward backtest and print performance summary.
  stress  : Run stress-test scenarios and print results.

Usage:
  python main.py --mode paper
  python main.py --mode backtest
  python main.py --mode stress
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NoReturn

import yaml
from dotenv import load_dotenv


def _load_config(config_path: str = "config/settings.yaml") -> dict:
    """
    Load and return the YAML settings file as a dict.

    Parameters
    ----------
    config_path : str

    Returns
    -------
    dict
    """
    raise NotImplementedError


def _parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Attributes: mode (str), config (str), symbols (list[str] | None)
    """
    parser = argparse.ArgumentParser(
        description="regime-trader: HMM-based regime allocation bot"
    )
    parser.add_argument(
        "--mode",
        choices=["live", "paper", "backtest", "stress"],
        default="paper",
        help="Execution mode (default: paper)",
    )
    parser.add_argument(
        "--config",
        default="config/settings.yaml",
        help="Path to settings YAML file",
    )
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=None,
        help="Override symbol list (space-separated tickers)",
    )
    return parser.parse_args()


def run_live(config: dict) -> NoReturn:
    """
    Main live / paper trading loop.

    Parameters
    ----------
    config : dict
        Loaded settings.yaml as a dict.
    """
    raise NotImplementedError


def run_backtest(config: dict) -> None:
    """
    Execute walk-forward backtest and print a performance summary.

    Parameters
    ----------
    config : dict
    """
    raise NotImplementedError


def run_stress(config: dict) -> None:
    """
    Execute stress-test scenarios and print results.

    Parameters
    ----------
    config : dict
    """
    raise NotImplementedError


def main() -> None:
    """Bootstrap environment, config, logging, then dispatch to the chosen mode."""
    # Load .env before anything else
    load_dotenv()

    args = _parse_args()

    try:
        config = _load_config(args.config)
    except NotImplementedError:
        # Config loading not yet implemented — use empty dict for skeleton run
        config = {}

    # Override symbols from CLI if provided
    if args.symbols:
        config.setdefault("broker", {})["symbols"] = args.symbols

    # Force paper mode
    if args.mode == "paper":
        config.setdefault("broker", {})["paper_trading"] = True

    dispatch = {
        "live": run_live,
        "paper": run_live,
        "backtest": run_backtest,
        "stress": run_stress,
    }

    try:
        dispatch[args.mode](config)
    except NotImplementedError:
        print(f"[regime-trader] Mode '{args.mode}' is not yet implemented.")
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n[regime-trader] Shutting down.")
        sys.exit(0)


if __name__ == "__main__":
    main()
