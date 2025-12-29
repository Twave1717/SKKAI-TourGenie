"""Prompt builder for LLM-based alpha survey."""

from __future__ import annotations

from typing import Any, Dict


def build_alpha_survey_prompt(persona: Dict[str, Any], trip_context: Dict[str, Any]) -> str:
    """Build a prompt for LLM to conduct alpha survey on a persona.

    Args:
        persona: Stravl persona profile with decoded preferences
        trip_context: Trip context (people, days, budget, season, etc.)

    Returns:
        Prompt string for LLM
    """
    # Extract key persona attributes
    ref_id = persona.get("ref_id", "unknown")
    age = persona.get("age_range", "unknown")
    travel_freq = persona.get("travel_frequency", "unknown")
    budget_tier = persona.get("budget_tier", "unknown")
    season_pref = persona.get("season_preference", "unknown")
    activity = persona.get("activity_level", "unknown")
    safety = persona.get("safety_conscious", "unknown")
    popularity = persona.get("destination_popularity", "unknown")
    experiences = persona.get("travel_experiences", [])
    scenery = persona.get("scenery_preferences", [])

    # Extract trip context
    people = trip_context.get("people_number", 2)
    days = trip_context.get("days", 3)
    budget = trip_context.get("budget_anchor")
    cities = trip_context.get("org", "Unknown")

    # --- Dynamic Budget Calculation (The Anchor Logic) ---
    # Calculate budget anchors for dynamic price tier generation
    if budget and budget > 0:
        # Daily budget per group (total group spending per day)
        daily_budget_total = budget / days

        # Daily budget per person
        daily_budget_pp = daily_budget_total / people

        # Category-specific anchors (Experience-based heuristic allocation)
        # - Accommodation: ~40% of daily group budget
        base_hotel = daily_budget_total * 0.40

        # - Dining: ~15% of daily per-person budget (per meal)
        base_meal = daily_budget_pp * 0.15

        # Multipliers for 5 tiers (Economy -> Luxury)
        # Range: 0.8x ~ 1.5x
        # Tier 1 (Economy):   0.8x (Lower bound)
        # Tier 2 (Budget):    0.9x (Frugal)
        # Tier 3 (Moderate):  1.0x (Standard Allocation - The Anchor)
        # Tier 4 (Upscale):   1.25x (Overspending)
        # Tier 5 (Luxury):    1.5x (Upper bound - Splurging)

        def fmt_price(val):
            return f"${int(val)}"

        # Dynamic Hotel Options
        h_opt1 = f"Economy (Under {fmt_price(base_hotel * 0.8)})"
        h_opt2 = f"Budget ({fmt_price(base_hotel * 0.8)} - {fmt_price(base_hotel * 0.9)})"
        h_opt3 = f"Moderate ({fmt_price(base_hotel * 0.9)} - {fmt_price(base_hotel * 1.1)})"
        h_opt4 = f"Upscale ({fmt_price(base_hotel * 1.1)} - {fmt_price(base_hotel * 1.3)})"
        h_opt5 = f"Luxury (Over {fmt_price(base_hotel * 1.3)})"

        # Dynamic Dining Options (Per Person Per Meal)
        r_opt1 = f"Economy (Under {fmt_price(base_meal * 0.8)})"
        r_opt2 = f"Budget ({fmt_price(base_meal * 0.8)} - {fmt_price(base_meal * 0.9)})"
        r_opt3 = f"Moderate ({fmt_price(base_meal * 0.9)} - {fmt_price(base_meal * 1.1)})"
        r_opt4 = f"Upscale ({fmt_price(base_meal * 1.1)} - {fmt_price(base_meal * 1.3)})"
        r_opt5 = f"Luxury (Over {fmt_price(base_meal * 1.3)})"
    else:
        # Fallback to generic options if budget is not available
        h_opt1 = "Economy (Under $50)"
        h_opt2 = "Budget ($50-100)"
        h_opt3 = "Moderate ($100-200)"
        h_opt4 = "Upscale ($200-400)"
        h_opt5 = "Luxury (Over $400)"

        r_opt1 = "Economy (Under $10)"
        r_opt2 = "Budget ($10-25)"
        r_opt3 = "Moderate ($25-50)"
        r_opt4 = "Upscale ($50-100)"
        r_opt5 = "Luxury (Over $100)"

    prompt = f"""# Context: Your Previous Travel Survey

You are **{ref_id}**, a traveler who has already completed the **Stravl Travel Preference Survey**.
You are now being asked to fill out an **additional detailed survey** for a specific trip to help optimize travel planning.

---

## Part 1: Your Completed Stravl Survey

*You previously filled out these questions. Use these as the foundation for your detailed preferences below.*

**Q1. What is your age range?**
→ Answer: {age}

**Q2. How often do you travel?**
→ Answer: {travel_freq}

**Q3. What is your typical travel budget level?**
→ Answer: {budget_tier}

**Q4. Which season do you prefer to travel?**
→ Answer: {season_pref}

**Q5. What is your activity level while traveling?**
→ Answer: {activity}

**Q6. How safety-conscious are you?**
→ Answer: {safety}

**Q7. What type of destinations do you prefer?**
→ Answer: {popularity}

**Q8. What travel experiences do you enjoy?**
→ Answer: {', '.join(experiences) if experiences else 'None specified'}

**Q9. What scenery do you prefer?**
→ Answer: {', '.join(scenery) if scenery else 'None specified'}

---

## Part 2: Current Trip Context

You are planning a trip with these details:
- **Destination**: {cities}
- **Duration**: {days} days
- **Budget Anchor (Reference)**: ${budget}
- **Group Size**: {people} people

**IMPORTANT**: The budget shown above is a **reference anchor** used to calculate price tier ranges in the questions below.
- Your **actual spending** can be higher or lower depending on your budget tier preference (Frugal/Comfort/Luxury).
- **Frugal/Budget travelers** typically spend ~0.8x the anchor (below the reference)
- **Comfort travelers** typically spend ~1.0x the anchor (at the reference)
- **Luxury travelers** typically spend ~1.5x the anchor (above the reference)
- The price ranges shown in questions are dynamically calculated based on this anchor.

---

## Part 3: Your Ongoing Survey

For **ALL 20 fields** below, please provide:
1. **Value**: Your preferred value or specific constraint
2. **Importance Score (0-10)**: How rigid you are about this preference
   - **9-10 (Must Have)**: Deal breaker. You will reject the plan if violated.
   - **7-8 (Should Have)**: Strong preference. You will argue for this but may compromise.
   - **4-6 (Could Have)**: Nice to have. You prefer this but will easily concede.
   - **0-3 (Indifferent)**: You truly do not care.
3. **Reason**: Explain *why* in **10-15 words** based on your Stravl survey answers

### Survey Questions:

### Section 1: Accommodations (8 questions)

**Q1-1. What is your preferred price tier per night?**
*Based on the group's total budget, here are the estimated ranges:*
- Options:
  - {h_opt1}
  - {h_opt2}
  - {h_opt3}
  - {h_opt4}
  - {h_opt5}
  - Flexible / No preference

**Q1-2. What is your MINIMUM acceptable hotel star rating?**
- Options (0-5 stars):
  - No minimum (Any star is fine)
  - 2 Stars (Basic)
  - 3 Stars (Comfort)
  - 4 Stars (Quality)
  - 5 Stars (Luxury only)

**Q1-3. What room type do you require?**
- Options:
  - Entire home/apt (Privacy critical)
  - Private room (Standard)
  - Shared room (Budget saver)
  - Any / Flexible

**Q1-4. Are there specific house rules you require or must avoid?**
- Options (select all that apply):
  - Must be Non-smoking
  - Must allow Pets
  - Must forbid Pets (Allergy)
  - Must allow Parties
  - Must forbid Children under 10
  - Must forbid Visitors
  - No specific requirements

**Q1-5. How flexible are you about "Minimum Nights" policies?**
- Options:
  - Very flexible (I can stay 3+ nights in one place)
  - Somewhat flexible (Max 2 nights per place)
  - Not flexible (I move every day / 1 night max)
  - No preference

**Q1-6. What minimum room capacity do you need for your group of {people}?**
- Options:
  - Exactly {people} people (Cost efficient)
  - {people}+ people (Need extra space)
  - No preference (As long as we fit)

**Q1-7. What is your minimum acceptable review score (out of 5.0)?**
- Options:
  - 4.5+ (Excellent)
  - 4.0+ (Very Good)
  - 3.5+ (Good)
  - 3.0+ (Okay)
  - Any score is fine

**Q1-8. Do you prefer consistent pricing across all bookings or are you comfortable with price variations?**
- Options:
  - Consistent (similar prices for all accommodations)
  - Moderate variation (some ups and downs acceptable)
  - High variation OK (I prioritize best value, even if prices vary)
  - No preference

### Section 2: Restaurants (6 questions)

**Q2-1. What is your preferred dining budget tier per person?**
*Estimated per-meal ranges:*
- Options:
  - {r_opt1}
  - {r_opt2}
  - {r_opt3}
  - {r_opt4}
  - {r_opt5}
  - Flexible

**Q2-2. What is your MINIMUM acceptable restaurant rating (out of 5.0)?**
- Options:
  - 4.5+ (Gourmet)
  - 4.0+ (Reliable)
  - 3.5+ (Decent)
  - Any rating (Food is food)

**Q2-3. What cuisines are MANDATORY or BANNED for you?**
- Options:
  - Mandatory: [List cuisines you MUST eat]
  - Banned: [List cuisines you REFUSE to eat]
  - No strict restrictions

**Q2-4. What dining atmosphere do you prefer?**
- Options:
  - Casual / Street Food
  - Family-friendly / Quiet
  - Lively / Bar-style
  - Romantic / Fine Dining
  - No preference

**Q2-5. How important is location convenience for dining?**
- Options:
  - Must be walking distance (<10 min)
  - Short drive/transit ok (<20 min)
  - Willing to travel for good food
  - No preference

**Q2-6. How many meals per day do you plan to eat out at restaurants?**
- Options:
  - 1 meal/day (mostly self-catering or hotel breakfast)
  - 2 meals/day (typical dining out)
  - 3 meals/day (all meals at restaurants)
  - Flexible (depends on the day)

### Section 3: Flights (5 questions)

**Q3-1. What is your preferred ticket price tier?**
- Options:
  - Super Saver (Cheapest possible)
  - Economy Standard (Average)
  - Premium Economy (Comfort)
  - Business/First (Luxury)
  - Flexible

**Q3-2. What is your tolerance for layovers/stops?**
- Options:
  - Direct flights ONLY
  - Direct preferred, 1 stop acceptable
  - Multiple stops acceptable (for savings)
  - No preference

**Q3-3. What is your preferred departure time window?**
- Options:
  - Morning (06:00 - 12:00)
  - Afternoon (12:00 - 18:00)
  - Evening/Night (18:00+)
  - Flexible

**Q3-4. What is your preferred arrival time window?**
- Options:
  - Morning (Arrive early to play)
  - Afternoon (Check-in time)
  - Evening/Night (Sleep immediately)
  - Flexible

**Q3-5. How important is baggage allowance included in the price?**
- Options:
  - Must include checked bag
  - Carry-on only is fine
  - No preference

### Section 4: Inter-city Travel (1 question)

**Q4-1. Maximum travel duration you can tolerate between cities?**
- Options:
  - Under 2 hours
  - 2 - 4 hours
  - 4+ hours is fine
  - No preference

---

## Critical Guidelines

**You must stay consistent with your Part 1 Stravl answers. Follow these core rules:**

1. **Budget Tier Consistency (Q3: "{budget_tier}")**
   - "Frugal/Budget" → ALL price questions importance ≥ 8, choose Economy/Budget tiers consistently across accommodations, dining, and flights
   - "Comfort" → Price importance 4-6, choose Moderate tiers consistently
   - "Luxury" → Price importance ≤ 3, choose Upscale/Luxury tiers + Quality (ratings/reviews) importance ≥ 7
   - **Be consistent**: If you're frugal, choose budget options for BOTH hotels AND restaurants. If luxury, choose luxury for BOTH.

2. **Group Size Logic (Group: {people} people)**
   - If group > 4 → Q1-6 (room capacity) importance MUST be ≥ 8 (hard constraint)
   - If group ≤ 2 → Q1-6 importance should be ≤ 3 (not critical)

3. **Importance Score Distribution**
   - 9-10 = Deal breaker (reject plan if violated)
   - 7-8 = Strong preference (argue but may compromise)
   - 4-6 = Nice to have (easily concede)
   - 0-3 = Indifferent (don't care)
   - **Do NOT make everything 9-10** - most travelers are flexible on SOME dimensions

---

**Now, complete ALL 20 questions above as yourself ({ref_id}), staying consistent with your Stravl survey answers.**"""

    return prompt
