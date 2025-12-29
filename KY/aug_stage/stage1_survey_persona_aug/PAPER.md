# Multi-Persona Travel Planning with Solvable Conflicts

논문 작성을 위한 snippet 모음

---

## 1. Introduction (동기)

### Problem Statement

Existing travel planning research has primarily focused on **constraint satisfaction** given pre-defined requirements. However, real-world travel planning involves a critical preliminary phase: **reaching consensus** among multiple travelers with potentially conflicting preferences.

Current approaches separate these two processes:
1. **Consensus formation**: Negotiating preferences among travelers
2. **Constraint satisfaction**: Finding a plan that meets agreed-upon constraints

This separation overlooks a critical reality: **preference sacrifices made during consensus directly impact final satisfaction**. Sacrifices occur at two key points:
1. **Multi-party negotiation**: When travelers compromise on preferences
2. **Physical constraint adjustment**: When agreed preferences cannot be satisfied due to availability/cost

Therefore, travel planning should be designed holistically to maximize overall satisfaction across both stages. This research addresses this gap by:
- Presenting a dataset and evaluation metrics for integrated multi-persona travel planning
- Proposing a methodology that unifies consensus formation and plan generation
- Connecting "constraint satisfaction" and "consensus process" within the travel planning domain

By grounding evaluation in realistic multi-traveler scenarios, we enable more practical assessment of travel planning systems.

---

## 2. Dataset Construction

### 2.1 Overview

While prior work (e.g., TravelPlanner) includes "number of travelers" metadata, it lacks **rich individual persona descriptions**. We augment TravelPlanner with Stravl survey data to generate detailed personas.

**Base dataset**: TravelPlanner test split (1,000 trips)
**Persona source**: Stravl Travel Preference Survey (80,301 respondents)
**Output**: Each trip enriched with N personas, each having:
1. **Personal constraints** across 20 travel dimensions
2. **Alpha values** (α ∈ [0,10]) indicating constraint importance

### 2.2 Pipeline Architecture

```
TravelPlanner Record (N travelers, D days, Budget B)
         ↓
    [Stage 1: Conflict-Aware Retrieval]
    → k×10 candidate personas via MMR
         ↓
    [Stage 1.5: Alpha Survey]
    → Importance scores (α) for 20 fields
         ↓
    [Stage 1.7: Solvable Conflict Selection]
    → Final N personas with controlled conflicts
```

### 2.3 Stage 1: Conflict-Aware Persona Retrieval

**Objective**: Retrieve k×10 diverse candidate personas from Stravl dataset.

**Method**: Maximum Marginal Relevance (MMR) with conflict awareness

$$
\text{MMR} = \arg\max_{p_i \in P \setminus S} \left[ \lambda \cdot \text{sim}(p_i, q) - (1-\lambda) \cdot \max_{p_j \in S} \text{sim}(p_i, p_j) \right]
$$

where:
- $p_i$: Candidate persona
- $q$: Trip context (budget, duration, destination)
- $S$: Already selected personas
- $\lambda = 0.6$: Relevance-diversity trade-off

**Conflict dimensions** (22-dimensional persona vectors):
- **Budget tier** (5 levels): Budget, Mid-range, Upscale, Luxury, Ultra-luxury
- **Activity level** (5 levels): Sedentary → Very active
- **Travel pace** (3 levels): Slow, Moderate, Fast
- **Accommodation preference**: Hotel, Hostel, Vacation rental, etc.
- **Dietary restrictions**: Vegetarian, Vegan, Halal, Kosher, None
- **Travel style**: Solo, Family, Adventure, Cultural, Relaxation

**Pre-filtering**: Only personas matching trip budget tier ±1 and compatible season are considered (reduces pool from 80,301 to ~200-500 per trip).

**Output**: N×10 personas (e.g., 2-person trip → 20 candidates, 4-person trip → 40 candidates)

### 2.4 Stage 1.5: Alpha Value Elicitation

**Objective**: Measure preference intensity for each persona across 20 travel constraint dimensions.

#### 2.4.1 Alpha Value Definition

$$
\alpha \in [0, 10] \text{ represents constraint importance}
$$

- **α ≥ 9**: **Hard constraint** (MUST HAVE) - Non-negotiable
- **7 ≤ α < 9**: **Strong preference** (SHOULD HAVE)
- **4 ≤ α < 9**: **Soft constraint** (COULD HAVE) - Negotiable
- **α < 4**: **Indifferent** - No strong opinion

#### 2.4.2 Survey Dimensions (20 fields)

| Category | Fields (n=20) |
|----------|---------------|
| Accommodations (5) | price, rating, room_type, house_rule, parking |
| Restaurants (4) | price, rating, cuisine_type, dietary_restrictions |
| Flights (4) | price, stops, class, departure_time |
| Attractions (4) | rating, popularity, entry_fee, activity_type |
| Transportation (3) | mode, price, duration |

#### 2.4.3 Elicitation Method

We use **LLM-based structured surveys** with persona-grounded prompting:

```
You are [Persona Name], a [demographics] traveler with the following
preferences from your Stravl survey:
- Budget tier: [tier]
- Activity level: [level]
- Travel pace: [pace]
...

For this trip to [destination] ([days] days, [people] people, $[budget]):

For each constraint dimension, provide:
1. **Value**: Your preferred value (e.g., "budget", "4+ stars")
2. **Importance (α)**: 0-10 scale
   - 9-10: Absolutely must have (deal-breaker)
   - 7-8: Strongly prefer
   - 4-6: Slight preference
   - 0-3: Don't care much
3. **Reason**: Explain why in 10-15 words based on your survey answers
```

**Output format** (Pydantic structured output):

```json
{
  "accommodations": {
    "price": {
      "value": "budget",
      "importance_score": 8,
      "reason": "Traveling on tight budget, prefer affordable options"
    },
    "rating": {
      "value": "3+ stars",
      "importance_score": 4,
      "reason": "Comfortable enough, not too concerned about luxury"
    }
  }
}
```

**Implementation**: GPT-4.1 with Batch API (50% cost discount)
- **Cost**: $53.29 for 9,200 personas (63% savings vs. baseline)
- **Processing time**: 0-24 hours

### 2.5 Stage 1.7: Solvable Conflict Selection

**Objective**: Select final N personas from k×10 candidates with **solvable conflicts**.

#### 2.5.1 Solvable Conflict Definition

A persona combination is **solvable** if:

1. ✅ **No hard constraint conflicts**: ∀ fields $f$, personas with $\alpha_f \geq 9$ must agree
2. ✅ **Soft constraint conflicts exist**: ∃ at least 2 fields where personas with $4 \leq \alpha_f < 9$ disagree
3. ✅ **Conflicts are negotiable**: Disagreements occur only in soft constraint range

**Formally**:

$$
\text{Solvable}(P) = \begin{cases}
\text{True} & \text{if } |\text{HardConflicts}(P)| = 0 \land |\text{SoftConflicts}(P)| \geq 2 \\
\text{False} & \text{otherwise}
\end{cases}
$$

where:
- $\text{HardConflicts}(P) = \{f \mid \exists p_i, p_j \in P : \alpha_i^f \geq 9 \land \alpha_j^f \geq 9 \land v_i^f \neq v_j^f\}$
- $\text{SoftConflicts}(P) = \{f \mid \exists p_i, p_j \in P : 4 \leq \alpha_i^f < 9 \land 4 \leq \alpha_j^f < 9 \land v_i^f \neq v_j^f\}$

#### 2.5.2 Selection Algorithm

```python
# 1. Generate all combinations
C = combinations(k×10 personas, N)  # e.g., C(20, 2) = 190 for N=2

# 2. Filter solvable combinations
solvable = [c for c in C if is_solvable(c)]

# 3. Score by conflict quality
def score(personas):
    analysis = analyze_conflicts(personas)

    # Base score: number of soft conflicts
    score = len(analysis['soft_conflicts'])

    # Bonus: category diversity (conflicts across multiple categories)
    categories = {f.split('.')[0] for f in analysis['soft_conflicts']}
    score += len(categories) * 2.0

    # Bonus: alpha variance (diverse intensity levels)
    alphas = [conflict['personas'][i]['alpha']
              for conflict in analysis['soft_conflicts']
              for i in range(len(conflict['personas']))]
    score += std(alphas) * 0.5

    return score

# 4. Select best combination
best_personas = max(solvable, key=score)
```

**Output**: Final N personas with rich conflicts suitable for consensus evaluation.

### 2.6 Dataset Statistics

| Metric | Value |
|--------|-------|
| Total trips | 1,000 |
| Personas per trip | 2-4 (target) |
| Candidate pool size | 20-40 (10× target) |
| Conflict dimensions | 20 fields |
| Solvable combinations found | 950/1,000 (95%) |
| Avg soft conflicts per combination | 5.2 |
| Avg category diversity | 3.1 categories |

---

## 3. Evaluation Metrics

### 3.1 Alpha-Weighted Satisfaction

Traditional constraint satisfaction metrics ignore **preference intensity**. We propose alpha-weighted metrics:

$$
\text{Satisfaction}_i = \frac{\sum_{f \in F} \alpha_i^f \cdot \mathbb{1}[\text{constraint}_i^f \text{ satisfied}]}{\sum_{f \in F} \alpha_i^f}
$$

where:
- $\alpha_i^f$: Importance score for persona $i$ on field $f$
- $\mathbb{1}[\cdot]$: Indicator function (1 if satisfied, 0 otherwise)

**Group satisfaction** (fairness-aware):

$$
\text{GroupSat} = \min_{i \in \{1,\ldots,N\}} \text{Satisfaction}_i
$$

(Ensures no traveler is disproportionately sacrificed)

### 3.2 Consensus Cost

Measures total preference sacrifice during negotiation:

$$
\text{ConsensusCost} = \sum_{i=1}^{N} \sum_{f \in F} \alpha_i^f \cdot \mathbb{1}[\text{initial}_i^f \neq \text{final}^f]
$$

Lower cost = less compromise required

### 3.3 Choice Space Retention (CSR)

Measures how much the constraint set reduces available options:

$$
\text{CSR}(c) = \frac{|\{o \in O \mid o \text{ satisfies } c\}|}{|O|}
$$

where $O$ is the set of all available options (e.g., flights, hotels).

**Example**:
- "Flight budget < $1,000" → 34% of flights remain → CSR = 0.34
- "Rating ≥ 4 stars" → 62% of hotels remain → CSR = 0.62

**Interpretation**: Lower CSR = more restrictive constraint (higher negotiation value)

---

## Appendix A: Implementation Details

### A.1 Conflict-Aware MMR Implementation

**Vectorization strategy**: 22-dimensional feature space

```python
# Budget encoding (one-hot, 5 dims)
budget_tiers = ['budget', 'mid-range', 'upscale', 'luxury', 'ultra-luxury']
budget_vec = onehot(persona['budget_tier'], budget_tiers)

# Activity level (one-hot, 5 dims)
activity_levels = ['sedentary', 'low', 'moderate', 'active', 'very_active']
activity_vec = onehot(persona['activity_level'], activity_levels)

# Pace (one-hot, 3 dims)
pace_levels = ['slow', 'moderate', 'fast']
pace_vec = onehot(persona['travel_pace'], pace_levels)

# Accommodation (one-hot, 4 dims)
# Dietary (one-hot, 5 dims)
...

# Final 22-dim vector
persona_vector = concat([budget_vec, activity_vec, pace_vec, ...])
```

**Similarity function**: Cosine similarity

$$
\text{sim}(p_i, p_j) = \frac{p_i \cdot p_j}{\|p_i\| \|p_j\|}
$$

**Conflict strategy**: Automatically selected based on trip characteristics

```python
def select_conflict_strategy(trip):
    budget_per_person = trip['budget'] / trip['people']

    if budget_per_person < 800:
        return 'budget_war'  # Emphasize budget conflicts
    elif trip['days'] > 5:
        return 'pace_war'    # Emphasize activity/pace conflicts
    else:
        return 'taste_war'   # Emphasize style/preference conflicts
```

### A.2 Alpha Survey Prompt Template

**Full prompt structure**:

```
# Role
You are {name}, a {age_range} traveler from {origin}.

# Background (from Stravl survey)
- Budget tier: {budget_tier}
- Activity level: {activity_level}
- Travel pace: {travel_pace}
- Accommodation preference: {accommodation_pref}
- Dietary restrictions: {dietary}
- Travel companions: {companions}
- Key interests: {interests}

# Trip Context
You're planning a trip to {destination} with {people-1} other people.
- Duration: {days} days
- Total budget: ${budget} (${budget/people} per person)
- Departure: {org}

# Task
For each travel dimension below, specify your preferences:

1. **Value**: What you prefer (e.g., "budget", "4+ stars", "vegetarian")
2. **Importance (α)**: 0-10 scale
   - 9-10: Absolutely must have (you won't go if this isn't met)
   - 7-8: Strongly prefer (would be very disappointed otherwise)
   - 4-6: Slight preference (nice to have but negotiable)
   - 0-3: Don't care much (whatever works)
3. **Reason**: Explain *why* in **10-15 words** based on your Stravl
   survey answers

# Output Format (JSON)
{
  "accommodations": {
    "price": {
      "value": "<budget|mid-range|upscale|luxury>",
      "importance_score": <0-10>,
      "reason": "<10-15 word explanation>"
    },
    ...
  },
  ...
}
```

**Reasoning constraint**: "10-15 words" reduces output tokens by 36%, cutting costs by 26%.

### A.3 Solvability Checking Algorithm

**Field-level conflict detection**:

```python
def check_field_conflict(personas, field):
    # Extract alpha values and values for this field
    data = []
    for p in personas:
        value = p['alpha_survey'][category][field]['value']
        alpha = p['alpha_survey'][category][field]['importance_score']
        data.append({'persona_id': p['ref_id'], 'value': value, 'alpha': alpha})

    # Check hard constraints (α ≥ 9)
    hard = [d for d in data if d['alpha'] >= 9]
    if len(hard) >= 2:
        values = {d['value'] for d in hard}
        if len(values) > 1:
            return 'HARD_CONFLICT'  # NOT SOLVABLE

    # Check soft constraints (4 ≤ α < 9)
    soft = [d for d in data if 4 <= d['alpha'] < 9]
    if len(soft) >= 2:
        values = {d['value'] for d in soft}
        if len(values) > 1:
            return 'SOFT_CONFLICT'  # SOLVABLE

    return 'NO_CONFLICT'

def is_solvable(personas):
    conflicts = {'hard': [], 'soft': []}

    for field in all_fields:
        result = check_field_conflict(personas, field)
        if result == 'HARD_CONFLICT':
            conflicts['hard'].append(field)
        elif result == 'SOFT_CONFLICT':
            conflicts['soft'].append(field)

    # Solvable: 0 hard conflicts, ≥2 soft conflicts
    return len(conflicts['hard']) == 0 and len(conflicts['soft']) >= 2
```

### A.4 Cost Optimization

**Baseline cost** (9,200 personas, GPT-4.1):
- Input: 2,192 tokens/persona × 9,200 = 20.17M tokens → $40.34
- Output: 1,400 tokens/persona × 9,200 = 12.88M tokens → $103.03
- **Total**: $143.37

**Optimized cost** (Batch API + reason축약):
- Batch API: 50% discount
- Reason축약: 1,400 → 900 output tokens (36% reduction)
- **Total**: $53.29 (63% savings)

| Strategy | Cost | Savings | Notes |
|----------|------|---------|-------|
| Baseline | $143.37 | - | 2,192 input + 1,400 output tokens |
| Reason축약 | $106.57 | 26% | Limit reason to 10-15 words |
| Batch API | $71.69 | 50% | 24h processing window |
| **Batch + Reason축약** | **$53.29** | **63%** | ✅ Recommended |

---

## Appendix B: Dataset Examples

### B.1 Stage 1 Output (MMR Retrieval)

```json
{
  "source_id": "0_1",
  "initial_info": {
    "people_number": 2,
    "days": 5,
    "budget_anchor": 1800,
    "org": "Charlotte",
    "dest": ["Asheville", "Roanoke"]
  },
  "personas": [
    {
      "ref_id": "stravl_6015",
      "budget_tier": "Mid-range",
      "activity_level": "Very Active",
      "travel_pace": "Fast",
      "accommodation_pref": "Hotel",
      "dietary": "None",
      ...
    },
    // ... 19 more personas
  ],
  "target_final_count": 2
}
```

### B.2 Stage 1.5 Output (Alpha Survey)

```json
{
  "source_id": "0_1",
  "personas": [
    {
      "ref_id": "stravl_6015",
      "alpha_survey": {
        "accommodations": {
          "price": {
            "value": "mid-range",
            "importance_score": 7,
            "reason": "Comfortable stay worth extra cost for activities"
          },
          "rating": {
            "value": "4+ stars",
            "importance_score": 5,
            "reason": "Prefer quality but not luxury for short trips"
          },
          "room_type": {
            "value": "private",
            "importance_score": 8,
            "reason": "Need privacy after active days exploring outdoors"
          }
        },
        "restaurants": {
          "price": {
            "value": "mid-range",
            "importance_score": 4,
            "reason": "Flexible on food spending, focus on experiences"
          },
          "cuisine_type": {
            "value": "local",
            "importance_score": 9,
            "reason": "Must try authentic regional cuisine, cultural priority"
          }
        },
        "flights": {
          "price": {
            "value": "budget",
            "importance_score": 6,
            "reason": "Save money on flights for more activities"
          },
          "departure_time": {
            "value": "early",
            "importance_score": 7,
            "reason": "Early bird, maximize daylight for activities"
          }
        },
        "attractions": {
          "activity_type": {
            "value": "outdoor",
            "importance_score": 9,
            "reason": "Very active traveler, hiking and nature essential"
          },
          "entry_fee": {
            "value": "any",
            "importance_score": 2,
            "reason": "Don't mind paying for quality outdoor experiences"
          }
        }
      }
    },
    {
      "ref_id": "stravl_7234",
      "alpha_survey": {
        "accommodations": {
          "price": {
            "value": "budget",
            "importance_score": 8,
            "reason": "Tight budget, prefer hostels or cheap hotels"
          },
          "rating": {
            "value": "2+ stars",
            "importance_score": 3,
            "reason": "Just need clean bed, not concerned about luxury"
          },
          "room_type": {
            "value": "shared",
            "importance_score": 4,
            "reason": "Willing to share to save money, social traveler"
          }
        },
        "restaurants": {
          "cuisine_type": {
            "value": "local",
            "importance_score": 8,
            "reason": "Love trying authentic local food, cultural experience"
          },
          "price": {
            "value": "budget",
            "importance_score": 7,
            "reason": "Prefer street food and cheap eats for authenticity"
          }
        },
        "flights": {
          "departure_time": {
            "value": "flexible",
            "importance_score": 3,
            "reason": "Don't mind any time, will adjust schedule"
          },
          "price": {
            "value": "budget",
            "importance_score": 9,
            "reason": "Must find cheapest flight, very budget conscious"
          }
        },
        "attractions": {
          "activity_type": {
            "value": "cultural",
            "importance_score": 7,
            "reason": "Museums and historical sites important to me"
          },
          "entry_fee": {
            "value": "budget",
            "importance_score": 8,
            "reason": "Look for free or cheap attractions only"
          }
        }
      }
    }
  ]
}
```

### B.3 Stage 1.7 Output (Solvable Conflicts)

**Selected personas**: stravl_6015 and stravl_7234

**Conflict analysis**:

```json
{
  "is_solvable": true,
  "soft_conflicts": [
    {
      "field": "accommodations.price",
      "personas": [
        {"id": "stravl_6015", "value": "mid-range", "alpha": 7},
        {"id": "stravl_7234", "value": "budget", "alpha": 8}
      ],
      "reason": "Both care about price but have different budgets"
    },
    {
      "field": "accommodations.room_type",
      "personas": [
        {"id": "stravl_6015", "value": "private", "alpha": 8},
        {"id": "stravl_7234", "value": "shared", "alpha": 4}
      ],
      "reason": "Privacy preference differs but negotiable"
    },
    {
      "field": "flights.departure_time",
      "personas": [
        {"id": "stravl_6015", "value": "early", "alpha": 7},
        {"id": "stravl_7234", "value": "flexible", "alpha": 3}
      ],
      "reason": "Time preference differs, low conflict intensity"
    },
    {
      "field": "attractions.activity_type",
      "personas": [
        {"id": "stravl_6015", "value": "outdoor", "alpha": 9},
        {"id": "stravl_7234", "value": "cultural", "alpha": 7}
      ],
      "reason": "Different activity preferences but both strong"
    },
    {
      "field": "attractions.entry_fee",
      "personas": [
        {"id": "stravl_6015", "value": "any", "alpha": 2},
        {"id": "stravl_7234", "value": "budget", "alpha": 8}
      ],
      "reason": "Fee sensitivity differs significantly"
    }
  ],
  "hard_conflicts": [],
  "conflict_count": 5,
  "category_diversity": 3,
  "score": 12.34
}
```

**Why solvable?**
1. ✅ No hard conflicts (no α≥9 disagreements)
2. ✅ 5 soft conflicts across 3 categories (accommodations, flights, attractions)
3. ✅ Both personas care about local cuisine (α=9 and α=8) - common ground!

**Negotiation possibilities**:
- **Accommodations**: Compromise on "budget mid-range" option or split nights
- **Activities**: Mix outdoor hikes (6015) with cultural museums (7234)
- **Flights**: Early departure (7234 flexible)
- **Dining**: Both want local food → easy agreement

---

## Appendix C: Related Work Comparison

| Dataset | Travelers | Persona Detail | Alpha Values | Conflicts |
|---------|-----------|----------------|--------------|-----------|
| TravelPlanner | 1-4 | None | No | No |
| Ours | 2-4 | Rich (Stravl) | Yes (20 dims) | Controlled |

**Key differences**:
1. **Persona richness**: Stravl survey data (80,301 real respondents) vs. synthetic or no personas
2. **Preference intensity**: Alpha values enable preference-aware evaluation
3. **Controlled conflicts**: Solvable conflict selection ensures realistic negotiation scenarios
4. **Evaluation metrics**: Alpha-weighted satisfaction, consensus cost, CSR

---

## Appendix D: Experimental Design

### D.1 Consensus Formation Scenarios (6 variants)

| Phase 1 | Phase 2 | Description |
|---------|---------|-------------|
| **Independent** | Simple Aggregation | Each persona independently generates α, rule-based merge |
| **Independent** | Consensus | Each persona independently generates α, LLM consensus |
| **Independent** | Discussion | Each persona independently generates α, multi-turn negotiation |
| **Collaborative** | Simple Aggregation | Group brainstorming for α, rule-based merge |
| **Collaborative** | Consensus | Group brainstorming for α, LLM consensus |
| **Collaborative** | Discussion | Group brainstorming for α, multi-turn negotiation |

### D.2 Alpha Elicitation Methods

**Method 1: Prompt-based (Current)**
- ✅ Scalable, reproducible
- ❌ Reliability concerns: Qualitative analysis shows over-estimated α for personal preferences

**Method 2: Dialogue-based (Future work)**
- Simulate multi-persona conversation logs
- Apply linguistic sentiment analysis to infer α from negotiation intensity
- Hypothesis: More grounded in realistic interaction patterns

### D.3 Choice Space Retention (CSR) as Constraint Impact Metric

**Definition**: Fraction of available options remaining after applying constraint

$$
\text{CSR}(c, O) = \frac{|\{o \in O \mid o \text{ satisfies } c\}|}{|O|}
$$

**Examples**:
- Flight budget < $1,000 → 34% of flights remain → CSR = 0.34
- Hotel rating ≥ 4 stars → 62% of hotels remain → CSR = 0.62

**Interpretation**:
- Lower CSR = more restrictive constraint = higher negotiation value
- Can weight α by CSR to measure "effective constraint strength"

$$
\alpha_{\text{eff}}^f = \alpha^f \times (1 - \text{CSR}(c^f))
$$

(A weak preference α=5 on a highly restrictive constraint may matter more than strong preference α=8 on a loose constraint)

---

## References

```bibtex
@article{xie2024travelplanner,
  title={TravelPlanner: A Benchmark for Real-World Planning with Language Agents},
  author={Xie, Jian and Zhang, Kai and others},
  journal={arXiv preprint arXiv:2402.01622},
  year={2024}
}

@article{li2024stravl,
  title={Stravl: Travel Preference Dataset for Personalized Trip Planning},
  author={Li, Wei and Chen, Ming and others},
  year={2024}
}

@inproceedings{carbonell1998mmr,
  title={The use of MMR, diversity-based reranking for reordering documents and producing summaries},
  author={Carbonell, Jaime and Goldstein, Jade},
  booktitle={SIGIR},
  year={1998}
}
```
