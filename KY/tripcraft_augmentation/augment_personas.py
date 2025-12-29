#!/usr/bin/env python3
"""Augment TripCraft personas with companion profiles via structured output (OpenAI Responses API).

For each JSON in the TripCraft outputs (3/5/7 day folders), this script:
- Parses core trip info into `initial_info`.
- Generates group personas (count = `people_number`, min 1) with role/profile
  structure using OpenAI Responses API (structured output parsing).
- Writes results into mirrored files under KY/tripcraft_augmentation/output as
  `group_personas`, preserving the original `persona` string.
- Processes files with a 4-worker thread pool and shows progress with tqdm.

Requires:
- OPENAI_API_KEY in your environment (loaded from .env automatically).
- The OpenAI Python SDK and python-dotenv (both already present here).

Usage:
    python KY/tripcraft_augmentation/augment_personas.py
"""

from __future__ import annotations

import concurrent.futures
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# Handle both script and module execution import paths.
try:
    from structured_output import OursInputOutput
except ImportError:  # pragma: no cover - fallback for package import
    try:
        from KY.tripcraft_augmentation.structured_output import OursInputOutput
    except ImportError:
        import sys
        sys.path.append(str(Path(__file__).resolve().parent))
        from structured_output import OursInputOutput


REPO_ROOT = Path(__file__).resolve().parents[2]
TRIPCRAFT_ROOT = REPO_ROOT / "benchmarks" / "TripCraft" / "tripcraft"
TARGET_SUBDIRS = ["tripcraft_3day", "tripcraft_5day", "tripcraft_7day"]
OUTPUT_ROOT = Path(__file__).resolve().parent / "tripcraft_augmentation_output"
OURS_INPUT_ROOT = Path(__file__).resolve().parent / "ours_input_json"


def build_prompts(
    trip: Dict[str, Any],
    base_persona: str | None,
) -> Tuple[str, str]:
    """Build system and user prompts for the chat completion call."""
    system_prompt = (
        "You are generating concise travel group personas in English.\n"
        "- Produce exactly the requested count.\n"
        "- Use people_number from Initial Info as the persona count.\n"
        "- Avoid duplicates; vary motivations, budgets, and location tastes.\n"
        "- Names must be English.\n"
        "- Include structured_requirement for each persona. Each required field must have value, alpha, beta (alpha importance 0-1; beta = 1 - alpha flexibility).\n"
        "- max_budget should reflect the trip budget; other required fields can be None when unknown but try to select 2-3 non-null enums among house_rule/room_type/cuisine/transportation/event/attraction.\n"
        "- Optional must contain 3+ entries; each with preference_name, value, alpha, beta, description (~50 chars explaining meaning and units/enum).\n"
        "- Optional values can be booleans, strings, or numbers (not limited to 0-1); choose meaningful keys relevant to the persona."
    )

    def fmt_list(value: Any) -> str:
        if isinstance(value, list):
            return ", ".join(str(v) for v in value)
        return str(value)

    user_payload_md = [
        "## Instructions",
        "- goal: Create group personas (including the requester and companions) who would enjoy this itinerary. Answer must be in English.",
        "",
        "## Output Schema",
        "- format: Return JSON with key 'group_personas' as a list of objects.",
        "- profile_schema:",
        "  - name: English name",
        "  - role: Short title describing their contribution to the trip",
        "  - profile:",
        "    - values: List of 3 values/concerns (nouns)",
        "    - focus_dimensions: List of 3 focal dimensions they care about",
        "    - profile_text: 1 sentence background/angle",
        "    - traveler_type: Defines how a traveler approaches their journey—whether they seek relaxation in cozy spots or adrenaline-pumping adventures.",
        "    - purpose_of_travel: Captures trip motivation. Examples: to unwind, explore cultures etc.",
        "    - spending_preference: Reflects the traveler's budget and style, from luxurious indulgence to cost-conscious experiences.",
        "    - location_preference: Highlights preferred environments, such as beaches, mountains, cities, or wildlife-rich forests.",
        "- structured_requirement:",
        "  - required_constraints: each has value, alpha (0-1), beta (0-1, usually 1-alpha). Do NOT include description for required.",
        "      - max_budget: The total budget of the trip.",
        "      - house_rule: Room rules include “No parties”, “No smoking”, “No children under 10”, “No pets”, and “No visitors”.",
        "      - room_type: Room types include “Entire Room”, “Private Room”, “Shared Room”, and “No Shared Room”.",
        "      - cuisine: Cuisines include “Chinese”, “American”, “Italian”, “Mexican”, “Indian”, “Mediterranean”, and “French”.",
        "      - transportation: Transportation options include “No flight” and “No self-driving”.",
        "      - event: Event Types include “Sports”, “Arts & Theatre”, “Music”, and “Film”.",
        "      - attraction: Each attraction belongs to one or more of 15 predefined categories: Boat Tours & Water Sports, Casinos & Gambling, Classes & Workshops, Concerts & Shows, Food & Drink, Fun & Games, Museums, Nature & Parks, Nightlife, Outdoor Activities, Shopping, Sights & Landmarks, Spas & Wellness, Water & Amusement Parks, Zoos & Aquariums.",
        "  - optional_preferences: list; add 3+ entries, each with preference_name, value, alpha, beta, description (~50 chars explaining preference meaning/units/enum). Values may be boolean, string, or numeric.",
        "",
        "## Few-shot Example",
        "- example_group_personas:",
        "  - name: Alice",
        "  - role: Fiscally Prudent",
        "  - profile:",
        "    - values: [\"Value for money\", \"Shared costs\", \"Cost transparency\"]",
        "    - focus_dimensions: [\"Total spend\", \"Avoid upsells\", \"Hidden transport fees\"]",
        "    - profile_text: \"Early 30s financial consultant; sums flights+stays+local transit first.\"",
        "    - traveler_type: Adventure Seeker",
        "    - purpose_of_travel: Cultural Exploration",
        "    - spending_preference: Luxury Traveler",
        "    - location_preference: Beaches",
        "  - structured_requirement:",
        "    - required_constraints:",
        "      - max_budget: {value: 3000, alpha: 0.8, beta: 0.2}",
        "      - house_rule: {value: \"No parties\", alpha: 0.2, beta: 0.8}",
        "      - cuisine: {value: \"Mediterranean\", alpha: 0.5, beta: 0.5}",
        "      - room_type: {value: \"Private Room\", alpha: 0.4, beta: 0.6}",
        "      - transportation: {value: None, alpha: 0.0, beta: 1.0}",
        "      - event: {value: \"Music\", alpha: 0.3, beta: 0.7}",
        "      - attraction: {value: \"Museums\", alpha: 0.6, beta: 0.4}",
        "    - optional_preferences:",
        "      - preference_name: avoid_hidden_fees",
        "        value: True",
        "        alpha: 0.9",
        "        beta: 0.1",
        "        description: \"Avoid hidden resort or cleaning fees.\"",
        "      - preference_name: prefer_walkable_areas",
        "        value: \"walkable neighborhoods\"",
        "        alpha: 0.6",
        "        beta: 0.4",
        "        description: \"Stay near areas easy to walk between sights.\"",
        "      - preference_name: favor_local_eateries",
        "        value: \"local restaurants\"",
        "        alpha: 0.5",
        "        beta: 0.5",
        "        description: \"Prefer local spots over chain restaurants.\"",
        "",
        "## Initial Info",
        f"- org: {trip.get('org')}",
        f"- dest: {trip.get('dest')}",
        f"- days: {trip.get('days')}",
        f"- visiting_city_number: {trip.get('visiting_city_number')}",
        f"- date: {fmt_list(trip.get('date'))}",
        f"- people_number: {trip.get('people_number')}",
        "",
        "## Persona (someone who would like this trip)",
        f"- persona: {base_persona or '-'}",
        "",
        "## Output",
        "Output: ",
    ]

    md_text = "\n".join(user_payload_md)
    user_payload = f'"""\n{md_text}\n"""'

    return system_prompt, user_payload


def generate_personas(
    client: OpenAI,
    trip: Dict[str, Any],
    base_persona: str | None,
) -> List[Dict]:
    """Call OpenAI Responses API to synthesize group personas with schema validation."""
    system_prompt, user_payload = build_prompts(trip, base_persona)

    response = client.responses.parse(
        model="gpt-4.1",
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload},
        ],
        temperature=0.8,
        text_format=OursInputOutput,
    )

    payload: OursInputOutput = response.output_parsed
    return payload.group_personas


def augment_file(client: OpenAI, path: Path) -> Path:
    with path.open() as f:
        data = json.load(f)

    people_number = int(data.get("people_number") or 1)
    base_persona = data.get("persona") if isinstance(data.get("persona"), str) else None

    group_personas = generate_personas(client, data, base_persona)
    group_personas_dicts = [
        gp.model_dump() if hasattr(gp, "model_dump") else gp for gp in group_personas
    ]

    data["group_personas"] = group_personas_dicts

    # Reorder so persona/group_personas sit immediately after people_number.
    keys = list(data.keys())
    for k in ["persona", "group_personas"]:
        if k in keys:
            keys.remove(k)
    insert_pos = keys.index("people_number") + 1 if "people_number" in keys else 0
    reordered_keys = keys[:insert_pos] + [k for k in ["persona", "group_personas"] if k in data] + keys[insert_pos:]
    data = {k: data[k] for k in reordered_keys}

    relative_path = path.relative_to(TRIPCRAFT_ROOT)
    out_path = OUTPUT_ROOT / relative_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # Also write to ours_input_json with condensed fields.
    ours_payload = {
        "initial_info": {
            "org": data.get("org"),
            "dest": data.get("dest"),
            "days": data.get("days"),
            "visiting_city_number": data.get("visiting_city_number"),
            "date": data.get("date"),
            "people_number": data.get("people_number"),
        },
        "group_personas": group_personas_dicts,
    }

    ours_relative = path.relative_to(TRIPCRAFT_ROOT)
    ours_path = OURS_INPUT_ROOT / ours_relative
    ours_path.parent.mkdir(parents=True, exist_ok=True)
    with ours_path.open("w", encoding="utf-8") as f:
        json.dump(ours_payload, f, ensure_ascii=False, indent=2)

    return out_path


def main() -> None:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is missing. Set it in .env or the environment.")

    client = OpenAI()

    files: List[Path] = []
    for subdir in TARGET_SUBDIRS:
        dir_path = TRIPCRAFT_ROOT / subdir
        files.extend(sorted(dir_path.glob("*.json")))

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    OURS_INPUT_ROOT.mkdir(parents=True, exist_ok=True)

    def worker(file_path: Path) -> Path:
        return augment_file(client, file_path)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(worker, path) for path in files]
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Augmenting personas",
        ):
            # Propagate any error immediately to stop the run.
            future.result()


if __name__ == "__main__":
    main()
