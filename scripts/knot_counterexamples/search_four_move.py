#!/usr/bin/env python3
"""Wrapper for 4-move bounded search."""

from search_n_move import main


if __name__ == "__main__":
    raise SystemExit(
        main(
            default_move_size=4,
            conjecture_name="Nakanishi (4-move)",
        )
    )
