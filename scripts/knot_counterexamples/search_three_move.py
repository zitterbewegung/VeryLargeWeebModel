#!/usr/bin/env python3
"""Wrapper for 3-move bounded search."""

from search_n_move import main


if __name__ == "__main__":
    raise SystemExit(
        main(
            default_move_size=3,
            conjecture_name="Montesinos-Nakanishi (3-move)",
        )
    )
