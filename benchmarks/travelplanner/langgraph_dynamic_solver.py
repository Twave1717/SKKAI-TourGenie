#!/usr/bin/env python
"""LangGraph Dynamic Solver Pipeline for TravelPlanner"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

load_dotenv()


class State(TypedDict):
    query: str
    constraints: dict
    solver_code: str
    solver_result: dict
    normalized_plan: dict
    evaluation_output: dict


def extract_constraints(state: State) -> State:
    """Node 1: Extract structured constraints from natural language using LLM"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    prompt = f"""Extract travel planning constraints from this query and output ONLY a valid Python dictionary.

Query: {state['query']}

Output a dictionary with these exact keys:
- budget (float)
- trip_days (int)
- origin (str)
- destination (str)
- forbidden_categories (list of str)
- preferred_categories (list of str)
- room_type (str or None)
- house_rule (str or None)
- transportation_mode (str or None)
- max_daily_activity_time (int, default 600)

Example output:
{{"budget": 1200.0, "trip_days": 4, "origin": "NYC", "destination": "LA", "forbidden_categories": ["Seafood"], "preferred_categories": ["Museum", "Beach"], "room_type": "Private room", "house_rule": None, "transportation_mode": None, "max_daily_activity_time": 600}}

Output ONLY the dictionary, no other text:"""
    
    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    
    try:
        constraints = eval(response.content.strip())
    except:
        constraints = {}
    
    return {"constraints": constraints}


def generate_solver_code(state: State) -> State:
    """Node 2: Generate complete OR-Tools CP-SAT solver code using LLM"""
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    constraints_str = json.dumps(state['constraints'], indent=2)
    
    prompt = f"""Generate a complete, executable Python function that creates and solves an OR-Tools CP-SAT travel planning problem.

Constraints:
{constraints_str}

The function must:
1. Accept candidates dict with: flights, hotels, restaurants, attractions
2. Create CP-SAT model with decision variables
3. Add constraints: budget, time, topology (1 outbound flight, 1 return flight, 1 hotel, 3 meals/day, 2-4 attractions/day)
4. Add category filtering (forbidden/preferred)
5. Add room constraints if specified
6. Set objective: maximize (100 * preference_score - 1 * total_cost)
7. Solve and return result dict

Output ONLY the function definition starting with:
```python
def solve_travel_plan(candidates, constraints):
```

Include all necessary imports inside the function. Return dict with: status, total_cost, flights, hotel, daily_plans.
Output ONLY executable Python code, no explanations:"""
    
    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    
    code = response.content.strip()
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0].strip()
    
    return {"solver_code": code}


def run_solver(state: State) -> State:
    """Node 3: Execute generated solver code in memory"""
    
    # Mock candidates (in real scenario, query from database)
    candidates = {
        'flights': [
            {'id': 'F001', 'price': 200, 'is_outbound': True},
            {'id': 'F002', 'price': 220, 'is_outbound': True},
            {'id': 'F003', 'price': 210, 'is_outbound': False},
            {'id': 'F004', 'price': 230, 'is_outbound': False},
        ],
        'hotels': [
            {'id': 'Hotel A', 'price_per_night': 100, 'category': 'Hotel', 'house_rules': 'No smoking', 'room_type': 'Private room'},
            {'id': 'Hotel B', 'price_per_night': 120, 'category': 'Hotel', 'house_rules': 'No pets', 'room_type': 'Shared room'},
            {'id': 'Hotel C', 'price_per_night': 80, 'category': 'Hostel', 'house_rules': 'No smoking', 'room_type': 'Private room'},
        ],
        'restaurants': [
            {'id': 'Restaurant ' + str(i), 'price': 15 + i*5, 'time_cost': 45 + i*5, 'category': ['Italian', 'Chinese', 'Seafood', 'American', 'Mexican'][i % 5]}
            for i in range(15)
        ],
        'attractions': [
            {'id': 'Attraction ' + str(i), 'price': 10 + i*5, 'time_cost': 60 + i*10, 'category': ['Museum', 'Beach', 'Park', 'Gallery'][i % 4]}
            for i in range(12)
        ]
    }
    
    # Execute solver code
    local_vars = {}
    try:
        exec(state['solver_code'], local_vars)
        solve_fn = local_vars.get('solve_travel_plan')
        
        if solve_fn:
            result = solve_fn(candidates, state['constraints'])
        else:
            result = {'status': 'error', 'message': 'Function not found'}
    except Exception as e:
        result = {'status': 'error', 'message': str(e)}
    
    return {"solver_result": result}


def normalize_plan(state: State) -> State:
    """Node 4: Convert solver result to normalized plan format"""
    
    result = state['solver_result']
    
    if result.get('status') == 'error':
        return {"normalized_plan": {}}
    
    # Normalize to TravelPlanner format
    normalized = {
        'status': result.get('status'),
        'total_cost': result.get('total_cost', 0),
        'days': []
    }
    
    daily_plans = result.get('daily_plans', [])
    for day_plan in daily_plans:
        day_entry = {
            'day': day_plan.get('day', 1),
            'meals': day_plan.get('meals', {}),
            'attractions': day_plan.get('attractions', []),
            'accommodation': result.get('hotel', {}).get('id', '-'),
            'daily_cost': day_plan.get('daily_cost', 0),
            'daily_time': day_plan.get('daily_time', 0)
        }
        normalized['days'].append(day_entry)
    
    return {"normalized_plan": normalized}


def run_evaluator(state: State) -> State:
    """Node 5: Run evaluator on normalized plan"""
    
    plan = state['normalized_plan']
    constraints = state['constraints']
    
    # Simplified evaluator (in real scenario, use TravelPlanner official evaluator)
    metrics = {
        'delivery_rate': 1.0 if plan.get('status') == 'optimal' else 0.0,
        'budget_satisfied': 1.0 if plan.get('total_cost', float('inf')) <= constraints.get('budget', 0) else 0.0,
        'days_match': 1.0 if len(plan.get('days', [])) == constraints.get('trip_days', 0) else 0.0,
    }
    
    # Check time constraints
    time_violations = sum(1 for day in plan.get('days', []) 
                         if day.get('daily_time', 0) > constraints.get('max_daily_activity_time', 600))
    metrics['time_satisfied'] = 1.0 if time_violations == 0 else 0.0
    
    # Overall score
    metrics['final_score'] = sum(metrics.values()) / len(metrics)
    
    evaluation = {
        'metrics': metrics,
        'plan_summary': {
            'total_cost': plan.get('total_cost'),
            'status': plan.get('status'),
            'num_days': len(plan.get('days', []))
        }
    }
    
    return {"evaluation_output": evaluation}


# Build LangGraph workflow
graph = StateGraph(State)

graph.add_node("extract_constraints", extract_constraints)
graph.add_node("generate_solver_code", generate_solver_code)
graph.add_node("run_solver", run_solver)
graph.add_node("normalize_plan", normalize_plan)
graph.add_node("run_evaluator", run_evaluator)

graph.add_edge(START, "extract_constraints")
graph.add_edge("extract_constraints", "generate_solver_code")
graph.add_edge("generate_solver_code", "run_solver")
graph.add_edge("run_solver", "normalize_plan")
graph.add_edge("normalize_plan", "run_evaluator")
graph.add_edge("run_evaluator", END)

app = graph.compile()


if __name__ == "__main__":
    result = app.invoke({
        "query": "Plan a 4-day trip from NYC to LA, budget $1200, avoid seafood, prefer museums and beaches, private room only."
    })
    
    print(json.dumps(result["evaluation_output"], indent=2))
