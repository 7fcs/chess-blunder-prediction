"""
Audit a Lichess PGN file.

The script reads the PGN one line at a time so it does not need to load
the entire dataset into memory. It summarizes game headers, clock data,
evaluation markers, ratings, and bot games.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import median


# PGN header example:
# [WhiteElo "1500"]
HEADER_PATTERN = re.compile(r'^\[([A-Za-z0-9_]+)\s+"(.*)"\]$')

# Lichess annotation examples:
# [%clk 0:05:32]
# [%eval 0.35]
CLOCK_PATTERN = re.compile(r"\[%clk\s+([^\]]+)\]")
EVAL_PATTERN = re.compile(r"\[%eval\s+([^\]]+)\]")


def parse_arguments() -> argparse.Namespace:
    """Read command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create a summary profile of a Lichess PGN file."
    )

    parser.add_argument(
        "pgn_path",
        type=Path,
        help="Path to the PGN file to audit.",
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Folder where the audit files will be saved.",
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=10,
        help="Number of game headers to save in the CSV sample.",
    )

    return parser.parse_args()


def is_numeric_rating(value: str | None) -> bool:
    """Return True if a PGN rating is a usable integer."""
    if value is None:
        return False

    try:
        rating = int(value)
    except ValueError:
        return False

    return rating > 0


def is_bot_game(headers: dict[str, str]) -> bool:
    """
    Return True when either player appears to be a bot.

    Lichess PGNs commonly identify bots through a BOT title or a
    username containing '[BOT]'.
    """
    for key in ("White", "Black", "WhiteTitle", "BlackTitle"):
        value = headers.get(key, "").upper()

        if value == "BOT" or "[BOT]" in value or value.endswith(" BOT"):
            return True

    return False


def finish_game(
    game: dict[str, object] | None,
    games: list[dict[str, object]],
) -> None:
    """Store the current game after its data has been collected."""
    if game is None:
        return

    headers = game["headers"]
    assert isinstance(headers, dict)

    games.append(
        {
            "headers": headers,
            "clock_markers": game["clock_markers"],
            "eval_markers": game["eval_markers"],
        }
    )


def read_games(
    pgn_path: Path,
    sample_size: int,
) -> list[dict[str, object]]:
    """
    Read games from the PGN file.

    A new game starts when a line begins with '[Event '.
    This approach is intentionally simple and works with a truncated
    sample whose final game may be incomplete.
    """
    games: list[dict[str, object]] = []
    current_game: dict[str, object] | None = None

    with pgn_path.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as pgn_file:
        for line in pgn_file:
            if line.startswith("[Event "):
                finish_game(current_game, games)

                current_game = {
                    "headers": {},
                    "clock_markers": 0,
                    "eval_markers": 0,
                }

            if current_game is None:
                continue

            clock_count = len(CLOCK_PATTERN.findall(line))
            eval_count = len(EVAL_PATTERN.findall(line))

            current_game["clock_markers"] += clock_count
            current_game["eval_markers"] += eval_count

            header_match = HEADER_PATTERN.match(line.strip())

            if header_match:
                key, value = header_match.groups()
                headers = current_game["headers"]
                assert isinstance(headers, dict)
                headers[key] = value

    # Include the final game, even if the sample ends before it finishes.
    finish_game(current_game, games)

    return games


def build_profile(
    pgn_path: Path,
    games: list[dict[str, object]],
) -> dict[str, object]:
    """Create a JSON-compatible summary of the PGN data."""
    total_games = len(games)

    clock_counts = [
        int(game["clock_markers"])
        for game in games
    ]

    eval_counts = [
        int(game["eval_markers"])
        for game in games
    ]

    games_with_clock = sum(count > 0 for count in clock_counts)
    games_with_eval = sum(count > 0 for count in eval_counts)

    ratings: list[int] = []

    bot_games = 0
    results: dict[str, int] = {}
    time_controls: dict[str, int] = {}

    for game in games:
        headers = game["headers"]
        assert isinstance(headers, dict)

        for rating_key in ("WhiteElo", "BlackElo"):
            rating = headers.get(rating_key)

            if is_numeric_rating(rating):
                assert rating is not None
                ratings.append(int(rating))

        if is_bot_game(headers):
            bot_games += 1

        result = headers.get("Result", "Unknown")
        results[result] = results.get(result, 0) + 1

        time_control = headers.get("TimeControl", "Unknown")
        time_controls[time_control] = (
            time_controls.get(time_control, 0) + 1
        )

    profile: dict[str, object] = {
        "source_file": str(pgn_path),
        "source_size_bytes": pgn_path.stat().st_size,
        "games_seen": total_games,
        "games_with_clock_data": games_with_clock,
        "clock_coverage": (
            games_with_clock / total_games
            if total_games > 0
            else 0.0
        ),
        "games_with_evaluation_markers": games_with_eval,
        "total_clock_markers": sum(clock_counts),
        "total_evaluation_markers": sum(eval_counts),
        "median_clock_markers_per_game": (
            median(clock_counts)
            if clock_counts
            else 0
        ),
        "median_clock_markers_among_games_with_clock": (
            median(
                [
                    count
                    for count in clock_counts
                    if count > 0
                ]
            )
            if games_with_clock > 0
            else 0
        ),
        "bot_games": bot_games,
        "rating_count": len(ratings),
        "rating_min": min(ratings) if ratings else None,
        "rating_max": max(ratings) if ratings else None,
        "result_counts": results,
        "time_control_counts": time_controls,
    }

    return profile


def write_header_sample(
    output_path: Path,
    games: list[dict[str, object]],
    sample_size: int,
) -> None:
    """Write a small CSV containing the first game headers."""
    fieldnames = [
        "game_number",
        "Event",
        "Site",
        "Date",
        "Round",
        "White",
        "Black",
        "Result",
        "WhiteElo",
        "BlackElo",
        "TimeControl",
        "Termination",
        "clock_markers",
        "evaluation_markers",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for game_number, game in enumerate(
            games[:sample_size],
            start=1,
        ):
            headers = game["headers"]
            assert isinstance(headers, dict)

            writer.writerow(
                {
                    "game_number": game_number,
                    "Event": headers.get("Event", ""),
                    "Site": headers.get("Site", ""),
                    "Date": headers.get("Date", ""),
                    "Round": headers.get("Round", ""),
                    "White": headers.get("White", ""),
                    "Black": headers.get("Black", ""),
                    "Result": headers.get("Result", ""),
                    "WhiteElo": headers.get("WhiteElo", ""),
                    "BlackElo": headers.get("BlackElo", ""),
                    "TimeControl": headers.get("TimeControl", ""),
                    "Termination": headers.get("Termination", ""),
                    "clock_markers": game["clock_markers"],
                    "evaluation_markers": game["eval_markers"],
                }
            )


def main() -> None:
    """Run the PGN audit."""
    args = parse_arguments()

    if not args.pgn_path.exists():
        raise FileNotFoundError(
            f"PGN file not found: {args.pgn_path}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading: {args.pgn_path}")

    games = read_games(
        pgn_path=args.pgn_path,
        sample_size=args.sample_size,
    )

    profile = build_profile(
        pgn_path=args.pgn_path,
        games=games,
    )

    profile_path = args.out_dir / "dataset_profile.json"
    sample_path = args.out_dir / "header_sample.csv"

    with profile_path.open(
        "w",
        encoding="utf-8",
    ) as json_file:
        json.dump(
            profile,
            json_file,
            indent=2,
        )
        json_file.write("\n")

    write_header_sample(
        output_path=sample_path,
        games=games,
        sample_size=args.sample_size,
    )

    print(f"Games found: {profile['games_seen']}")
    print(f"Games with clock data: {profile['games_with_clock_data']}")
    print(f"Rating range: {profile['rating_min']} to {profile['rating_max']}")
    print(f"Saved: {profile_path}")
    print(f"Saved: {sample_path}")


if __name__ == "__main__":
    main()