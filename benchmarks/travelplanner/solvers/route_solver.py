"""
TravelRouteSolver: OR-Tools CP-SAT based trip optimizer.

This solver takes candidate travel items (flights, hotels, restaurants, attractions)
and user constraints, then outputs an optimal day-by-day itinerary.

Key Features:
- HARD constraints: Budget, Time, Category exclusions, Selection counts
- SOFT constraints: Cost minimization, Preference matching
- Uses Google OR-Tools CP-SAT solver for efficient constraint satisfaction
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from ortools.sat.python import cp_model


class TravelRouteSolver:
    """
    Optimization-based travel itinerary planner using CP-SAT.
    
    This class formulates the trip planning problem as a Mixed Integer Linear Program (MILP)
    with hard and soft constraints, then solves it using Google OR-Tools.
    """
    
    def __init__(self, candidates: Dict[str, List[Dict]], constraints: Dict[str, Any]):
        """
        Initialize the solver with candidate items and constraints.
        
        Args:
            candidates: Dictionary containing:
                - flights: List[{'id': str, 'price': float, 'is_outbound': bool}]
                - hotels: List[{'id': str, 'price_per_night': float, 'category': str}]
                - restaurants: List[{'id': str, 'price': float, 'time_cost': int, 'category': str}]
                - attractions: List[{'id': str, 'price': float, 'time_cost': int, 'category': str}]
            constraints: Dictionary containing:
                - budget: float (Maximum total cost)
                - trip_days: int (Number of days, e.g., 3)
                - max_daily_activity_time: int (Maximum minutes per day, e.g., 600)
                - forbidden_categories: List[str] (HARD: Categories to exclude)
                - preferred_categories: List[str] (SOFT: Preferred categories for bonus)
                - room_constraints: Dict (HARD/SOFT: House rule, room type)
                - transportation_mode: str (HARD: Transportation constraint if specified)
        """
        self.candidates = candidates
        self.constraints = constraints
        
        # Extract data
        self.flights = candidates.get('flights', [])
        self.hotels = candidates.get('hotels', [])
        self.restaurants = candidates.get('restaurants', [])
        self.origin_restaurants = candidates.get('origin_restaurants', [])  # Day 1 breakfast
        self.attractions = candidates.get('attractions', [])
        
        # Extract constraints
        self.budget = constraints.get('budget', float('inf'))
        self.trip_days = constraints.get('trip_days', 3)
        self.max_daily_time = constraints.get('max_daily_activity_time', 600)
        self.forbidden_categories = set(constraints.get('forbidden_categories', []))
        self.preferred_categories = set(constraints.get('preferred_categories', []))
        self.room_constraints = constraints.get('room_constraints', {})
        self.transportation_mode = constraints.get('transportation_mode')
        
        # Initialize model
        self.model = cp_model.CpModel()
        self.vars = {}  # Store all decision variables
        
        # Weights for objective function
        self.W_PREFERENCE = 100  # Prioritize preference matching
        self.W_COST = 1  # Secondary: minimize cost
        self.PREFERENCE_BONUS = 10  # Points for preferred category
        self.BASE_VISIT_SCORE = 1  # Points for any valid visit
        
    def _create_variables(self) -> None:
        """Create all binary decision variables for the optimization problem."""
        
        # ========== DECISION VARIABLES ==========
        
        # Flight variables: x_flight[i] = 1 if flight i is selected
        self.vars['flights'] = {}
        for i, flight in enumerate(self.flights):
            var_name = f"flight_{i}_{flight['id']}"
            self.vars['flights'][i] = self.model.NewBoolVar(var_name)
        
        # Hotel variables: x_hotel[i] = 1 if hotel i is selected (for whole trip)
        self.vars['hotels'] = {}
        for i, hotel in enumerate(self.hotels):
            var_name = f"hotel_{i}_{hotel['id']}"
            self.vars['hotels'][i] = self.model.NewBoolVar(var_name)
        
        # Restaurant variables: x_meal[day][meal_type][i] = 1 if restaurant i is selected
        # meal_type: 'breakfast', 'lunch', 'dinner'
        # Day 1 breakfast uses origin_restaurants, all others use dest restaurants
        self.vars['meals'] = {}
        meal_types = ['breakfast', 'lunch', 'dinner']
        for day in range(self.trip_days):
            self.vars['meals'][day] = {}
            for meal_type in meal_types:
                self.vars['meals'][day][meal_type] = {}
                
                # Day 1 breakfast: use origin city restaurants
                if day == 0 and meal_type == 'breakfast' and self.origin_restaurants:
                    for i, restaurant in enumerate(self.origin_restaurants):
                        var_name = f"meal_d{day}_{meal_type}_origin_{i}_{restaurant['id']}"
                        self.vars['meals'][day][meal_type][i] = self.model.NewBoolVar(var_name)
                else:
                    # All other meals: use destination city restaurants
                    for i, restaurant in enumerate(self.restaurants):
                        var_name = f"meal_d{day}_{meal_type}_{i}_{restaurant['id']}"
                        self.vars['meals'][day][meal_type][i] = self.model.NewBoolVar(var_name)
        
        # Attraction variables: x_attr[day][i] = 1 if attraction i is selected on day
        self.vars['attractions'] = {}
        for day in range(self.trip_days):
            self.vars['attractions'][day] = {}
            for i, attraction in enumerate(self.attractions):
                var_name = f"attr_d{day}_{i}_{attraction['id']}"
                self.vars['attractions'][day][i] = self.model.NewBoolVar(var_name)
    
    def _add_hard_constraints(self) -> None:
        """Add all hard constraints that MUST be satisfied."""
        
        # ========== HARD CONSTRAINT 1: Topology Rules ==========
        
        # Must select exactly 1 outbound flight
        outbound_flights = [
            self.vars['flights'][i] 
            for i, f in enumerate(self.flights) 
            if f.get('is_outbound', False)
        ]
        if outbound_flights:
            self.model.Add(sum(outbound_flights) == 1)
        
        # Must select exactly 1 return flight
        return_flights = [
            self.vars['flights'][i] 
            for i, f in enumerate(self.flights) 
            if not f.get('is_outbound', False)
        ]
        if return_flights:
            self.model.Add(sum(return_flights) == 1)
        
        # Must select exactly 1 hotel for the entire trip
        if self.hotels:
            self.model.Add(sum(self.vars['hotels'].values()) == 1)
        
        # Must select exactly 1 restaurant per meal type per day (3 meals/day)
        for day in range(self.trip_days):
            for meal_type in ['breakfast', 'lunch', 'dinner']:
                meal_vars = list(self.vars['meals'][day][meal_type].values())
                if meal_vars:
                    self.model.Add(sum(meal_vars) == 1)
        
        # Must select between 2 to 4 attractions per day
        for day in range(self.trip_days):
            attr_vars = list(self.vars['attractions'][day].values())
            if attr_vars:
                self.model.Add(sum(attr_vars) >= 2)
                self.model.Add(sum(attr_vars) <= 4)
        
        # ========== HARD CONSTRAINT 2: Budget Rule ==========
        
        total_cost_terms = []
        
        # Flight costs
        for i, flight in enumerate(self.flights):
            cost = int(flight.get('price', 0) * 100)  # Scale to avoid floating point
            total_cost_terms.append(self.vars['flights'][i] * cost)
        
        # Hotel costs (price_per_night * trip_days)
        for i, hotel in enumerate(self.hotels):
            cost = int(hotel.get('price_per_night', 0) * self.trip_days * 100)
            total_cost_terms.append(self.vars['hotels'][i] * cost)
        
        # Restaurant costs
        for day in range(self.trip_days):
            for meal_type in ['breakfast', 'lunch', 'dinner']:
                for i, restaurant in enumerate(self.restaurants):
                    cost = int(restaurant.get('price', 0) * 100)
                    total_cost_terms.append(self.vars['meals'][day][meal_type][i] * cost)
        
        # Attraction costs
        for day in range(self.trip_days):
            for i, attraction in enumerate(self.attractions):
                cost = int(attraction.get('price', 0) * 100)
                total_cost_terms.append(self.vars['attractions'][day][i] * cost)
        
        # Total cost must not exceed budget (scaled by 100)
        budget_scaled = int(self.budget * 100)
        self.model.Add(sum(total_cost_terms) <= budget_scaled)
        
        # ========== HARD CONSTRAINT 3: Time Rule ==========
        
        # For each day: Sum(time_cost of restaurants + attractions) <= max_daily_activity_time
        for day in range(self.trip_days):
            daily_time_terms = []
            
            # Restaurant time costs
            for meal_type in ['breakfast', 'lunch', 'dinner']:
                for i, restaurant in enumerate(self.restaurants):
                    time_cost = restaurant.get('time_cost', 0)
                    daily_time_terms.append(self.vars['meals'][day][meal_type][i] * time_cost)
            
            # Attraction time costs
            for i, attraction in enumerate(self.attractions):
                time_cost = attraction.get('time_cost', 0)
                daily_time_terms.append(self.vars['attractions'][day][i] * time_cost)
            
            self.model.Add(sum(daily_time_terms) <= self.max_daily_time)
        
        # ========== HARD CONSTRAINT 4: Category Exclusion Rule ==========
        
        # If item category is in forbidden_categories, force x[i] = 0
        
        # Hotels
        for i, hotel in enumerate(self.hotels):
            if hotel.get('category', '') in self.forbidden_categories:
                self.model.Add(self.vars['hotels'][i] == 0)
        
        # Restaurants
        for day in range(self.trip_days):
            for meal_type in ['breakfast', 'lunch', 'dinner']:
                for i, restaurant in enumerate(self.restaurants):
                    if restaurant.get('category', '') in self.forbidden_categories:
                        self.model.Add(self.vars['meals'][day][meal_type][i] == 0)
        
        # Attractions
        for day in range(self.trip_days):
            for i, attraction in enumerate(self.attractions):
                if attraction.get('category', '') in self.forbidden_categories:
                    self.model.Add(self.vars['attractions'][day][i] == 0)
        
        # ========== HARD CONSTRAINT 5: Uniqueness Rule ==========
        
        # Same restaurant cannot be visited more than once across entire trip
        for i in range(len(self.restaurants)):
            usage_across_days = []
            for day in range(self.trip_days):
                for meal_type in ['breakfast', 'lunch', 'dinner']:
                    if i in self.vars['meals'][day][meal_type]:
                        usage_across_days.append(self.vars['meals'][day][meal_type][i])
            if usage_across_days:
                self.model.Add(sum(usage_across_days) <= 1)
        
        # Same attraction cannot be visited more than once across entire trip
        for i in range(len(self.attractions)):
            usage_across_days = []
            for day in range(self.trip_days):
                if i in self.vars['attractions'][day]:
                    usage_across_days.append(self.vars['attractions'][day][i])
            if usage_across_days:
                self.model.Add(sum(usage_across_days) <= 1)
        
        # ========== HARD CONSTRAINT 6: Room Rule Constraints ==========
        
        # If house_rule is specified, enforce it
        house_rule = self.room_constraints.get('house_rule')
        if house_rule:
            for i, hotel in enumerate(self.hotels):
                hotel_rules = hotel.get('house_rules', '')
                # If hotel doesn't match required rule, exclude it
                if house_rule not in ['', '-', None]:
                    # Simple matching - hotel must contain the required rule
                    if house_rule.lower() not in str(hotel_rules).lower():
                        self.model.Add(self.vars['hotels'][i] == 0)
        
        # ========== HARD CONSTRAINT 7: Transportation Mode Constraints ==========
        
        # If transportation mode constraint is specified
        if self.transportation_mode:
            transport_lower = str(self.transportation_mode).lower()
            
            # Handle 'no flight' constraint
            if 'no flight' in transport_lower or 'no flights' in transport_lower:
                # Disable all flight selections
                for i in range(len(self.flights)):
                    self.model.Add(self.vars['flights'][i] == 0)
    
    def _set_objective(self) -> None:
        """
        Set the objective function to maximize value.
        
        Objective = (W_PREFERENCE × PreferenceScore) - (W_COST × TotalCost)
        
        SOFT CONSTRAINTS:
        - PreferenceScore: Reward items in preferred_categories
        - TotalCost: Minimize total spending (within budget)
        """
        
        preference_terms = []
        cost_terms = []
        
        # ========== SOFT CONSTRAINT: Preference Matching ==========
        
        # Restaurants: +10 points if preferred, +1 otherwise
        for day in range(self.trip_days):
            for meal_type in ['breakfast', 'lunch', 'dinner']:
                # Day 1 breakfast uses origin restaurants
                if day == 0 and meal_type == 'breakfast' and self.origin_restaurants:
                    for i, restaurant in enumerate(self.origin_restaurants):
                        if i in self.vars['meals'][day][meal_type]:
                            category = restaurant.get('category', '')
                            score = self.PREFERENCE_BONUS if category in self.preferred_categories else self.BASE_VISIT_SCORE
                            preference_terms.append(self.vars['meals'][day][meal_type][i] * score)
                else:
                    # All other meals use destination restaurants
                    for i, restaurant in enumerate(self.restaurants):
                        if i in self.vars['meals'][day][meal_type]:
                            category = restaurant.get('category', '')
                            score = self.PREFERENCE_BONUS if category in self.preferred_categories else self.BASE_VISIT_SCORE
                            preference_terms.append(self.vars['meals'][day][meal_type][i] * score)
        
        # Attractions: +10 points if preferred, +1 otherwise
        for day in range(self.trip_days):
            for i, attraction in enumerate(self.attractions):
                category = attraction.get('category', '')
                if category in self.preferred_categories:
                    score = self.PREFERENCE_BONUS
                else:
                    score = self.BASE_VISIT_SCORE
                preference_terms.append(self.vars['attractions'][day][i] * score)
        
        # Hotels: +20 points for matching room type preference
        room_type_pref = self.room_constraints.get('room_type')
        if room_type_pref and room_type_pref not in ['', '-', None]:
            for i, hotel in enumerate(self.hotels):
                hotel_room_type = hotel.get('room_type', '')
                if room_type_pref.lower() in str(hotel_room_type).lower():
                    preference_terms.append(self.vars['hotels'][i] * 20)  # Bonus for matching room type
        
        # ========== SOFT CONSTRAINT: Cost Minimization ==========
        
        # Same as budget constraint, but used in objective to minimize spending
        
        # Flight costs
        for i, flight in enumerate(self.flights):
            cost = int(flight.get('price', 0) * 100)
            cost_terms.append(self.vars['flights'][i] * cost)
        
        # Hotel costs
        for i, hotel in enumerate(self.hotels):
            cost = int(hotel.get('price_per_night', 0) * self.trip_days * 100)
            cost_terms.append(self.vars['hotels'][i] * cost)
        
        # Restaurant costs
        for day in range(self.trip_days):
            for meal_type in ['breakfast', 'lunch', 'dinner']:
                # Day 1 breakfast uses origin restaurants
                if day == 0 and meal_type == 'breakfast' and self.origin_restaurants:
                    for i, restaurant in enumerate(self.origin_restaurants):
                        if i in self.vars['meals'][day][meal_type]:
                            cost = int(restaurant.get('price', 0) * 100)
                            cost_terms.append(self.vars['meals'][day][meal_type][i] * cost)
                else:
                    # All other meals use destination restaurants
                    for i, restaurant in enumerate(self.restaurants):
                        if i in self.vars['meals'][day][meal_type]:
                            cost = int(restaurant.get('price', 0) * 100)
                            cost_terms.append(self.vars['meals'][day][meal_type][i] * cost)
        
        # Attraction costs
        for day in range(self.trip_days):
            for i, attraction in enumerate(self.attractions):
                cost = int(attraction.get('price', 0) * 100)
                cost_terms.append(self.vars['attractions'][day][i] * cost)
        
        # ========== Combine into Objective Function ==========
        
        # Objective = W_PREFERENCE × PreferenceScore - W_COST × TotalCost
        # We want to MAXIMIZE this value
        preference_score = sum(preference_terms) if preference_terms else 0
        total_cost = sum(cost_terms) if cost_terms else 0
        
        objective_expr = (self.W_PREFERENCE * preference_score) - (self.W_COST * total_cost)
        self.model.Maximize(objective_expr)
    
    def _extract_solution(self, solver: cp_model.CpSolver) -> Dict[str, Any]:
        """
        Extract the solution and format as a structured JSON itinerary.
        
        Returns:
            Dictionary with day-by-day itinerary including selected items and total cost.
        """
        
        itinerary = {
            'status': 'optimal',
            'total_cost': 0.0,
            'flights': {
                'outbound': None,
                'return': None
            },
            'hotel': None,
            'daily_plans': []
        }
        
        # Extract selected flights
        for i, flight in enumerate(self.flights):
            if solver.Value(self.vars['flights'][i]) == 1:
                flight_info = {
                    'id': flight['id'],
                    'price': flight['price']
                }
                if flight.get('is_outbound', False):
                    itinerary['flights']['outbound'] = flight_info
                else:
                    itinerary['flights']['return'] = flight_info
                itinerary['total_cost'] += flight['price']
        
        # Extract selected hotel
        for i, hotel in enumerate(self.hotels):
            if solver.Value(self.vars['hotels'][i]) == 1:
                total_hotel_cost = hotel['price_per_night'] * self.trip_days
                itinerary['hotel'] = {
                    'id': hotel['id'],
                    'category': hotel.get('category', ''),
                    'price_per_night': hotel['price_per_night'],
                    'nights': self.trip_days,
                    'total_cost': total_hotel_cost
                }
                itinerary['total_cost'] += total_hotel_cost
                break
        
        # Extract daily plans
        for day in range(self.trip_days):
            day_plan = {
                'day': day + 1,
                'meals': {
                    'breakfast': None,
                    'lunch': None,
                    'dinner': None
                },
                'attractions': [],
                'daily_cost': 0.0,
                'daily_time': 0
            }
            
            # Extract meals
            for meal_type in ['breakfast', 'lunch', 'dinner']:
                # Day 1 breakfast uses origin restaurants
                if day == 0 and meal_type == 'breakfast' and self.origin_restaurants:
                    for i, restaurant in enumerate(self.origin_restaurants):
                        if i in self.vars['meals'][day][meal_type] and solver.Value(self.vars['meals'][day][meal_type][i]) == 1:
                            meal_info = {
                                'id': restaurant['id'],
                                'category': restaurant.get('category', ''),
                                'price': restaurant['price'],
                                'time_cost': restaurant['time_cost']
                            }
                            day_plan['meals'][meal_type] = meal_info
                            day_plan['daily_cost'] += restaurant['price']
                            day_plan['daily_time'] += restaurant['time_cost']
                            itinerary['total_cost'] += restaurant['price']
                            break
                else:
                    # All other meals use destination restaurants
                    for i, restaurant in enumerate(self.restaurants):
                        if i in self.vars['meals'][day][meal_type] and solver.Value(self.vars['meals'][day][meal_type][i]) == 1:
                            meal_info = {
                                'id': restaurant['id'],
                                'category': restaurant.get('category', ''),
                                'price': restaurant['price'],
                                'time_cost': restaurant['time_cost']
                            }
                            day_plan['meals'][meal_type] = meal_info
                            day_plan['daily_cost'] += restaurant['price']
                            day_plan['daily_time'] += restaurant['time_cost']
                            itinerary['total_cost'] += restaurant['price']
                            break
            
            # Extract attractions
            for i, attraction in enumerate(self.attractions):
                if solver.Value(self.vars['attractions'][day][i]) == 1:
                    attr_info = {
                        'id': attraction['id'],
                        'category': attraction.get('category', ''),
                        'price': attraction['price'],
                        'time_cost': attraction['time_cost']
                    }
                    day_plan['attractions'].append(attr_info)
                    day_plan['daily_cost'] += attraction['price']
                    day_plan['daily_time'] += attraction['time_cost']
                    itinerary['total_cost'] += attraction['price']
            
            itinerary['daily_plans'].append(day_plan)
        
        # Round total cost to 2 decimal places
        itinerary['total_cost'] = round(itinerary['total_cost'], 2)
        
        return itinerary
    
    def solve(self, time_limit_seconds: int = 30) -> Optional[Dict[str, Any]]:
        """
        Solve the optimization problem and return the itinerary.
        
        Args:
            time_limit_seconds: Maximum time to spend solving (default: 30)
        
        Returns:
            Dictionary with complete itinerary if solution found, None otherwise.
            Format:
            {
                'status': 'optimal' | 'feasible' | 'infeasible',
                'total_cost': float,
                'flights': {'outbound': {...}, 'return': {...}},
                'hotel': {...},
                'daily_plans': [
                    {
                        'day': 1,
                        'meals': {'breakfast': {...}, 'lunch': {...}, 'dinner': {...}},
                        'attractions': [{...}, {...}],
                        'daily_cost': float,
                        'daily_time': int
                    },
                    ...
                ]
            }
        """
        
        # Step 1: Create decision variables
        self._create_variables()
        
        # Step 2: Add hard constraints
        self._add_hard_constraints()
        
        # Step 3: Set objective function (soft constraints)
        self._set_objective()
        
        # Step 4: Solve the model
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit_seconds
        solver.parameters.log_search_progress = False  # Set to True for debugging
        
        status = solver.Solve(self.model)
        
        # Step 5: Process results
        if status == cp_model.OPTIMAL:
            print("[TravelRouteSolver] Found OPTIMAL solution")
            return self._extract_solution(solver)
        
        elif status == cp_model.FEASIBLE:
            print("[TravelRouteSolver] Found FEASIBLE solution (may not be optimal)")
            result = self._extract_solution(solver)
            result['status'] = 'feasible'
            return result
        
        elif status == cp_model.INFEASIBLE:
            print("[TravelRouteSolver] Problem is INFEASIBLE")
            print("Possible reasons:")
            print("  - Budget too tight (cannot afford minimum required items)")
            print("  - Time constraint too restrictive (cannot fit activities in daily limit)")
            print("  - Forbidden categories exclude too many options (not enough valid choices)")
            print("  - Not enough candidates to satisfy selection requirements (e.g., need 3 meals but only 2 restaurants available)")
            return None
        
        elif status == cp_model.MODEL_INVALID:
            print("[TravelRouteSolver] Model is INVALID - check constraint definitions")
            return None
        
        else:
            print(f"[TravelRouteSolver] Unknown status: {status}")
            return None


# ========== EXAMPLE USAGE ==========

if __name__ == "__main__":
    # Sample input data
    sample_candidates = {
        'flights': [
            {'id': 'F001', 'price': 200, 'is_outbound': True},
            {'id': 'F002', 'price': 250, 'is_outbound': True},
            {'id': 'F003', 'price': 210, 'is_outbound': False},
            {'id': 'F004', 'price': 230, 'is_outbound': False},
        ],
        'hotels': [
            {'id': 'H001', 'price_per_night': 100, 'category': 'Hotel'},
            {'id': 'H002', 'price_per_night': 150, 'category': 'Resort'},
            {'id': 'H003', 'price_per_night': 50, 'category': 'Hostel'},
        ],
        'restaurants': [
            {'id': 'R001', 'price': 20, 'time_cost': 60, 'category': 'Italian'},
            {'id': 'R002', 'price': 15, 'time_cost': 45, 'category': 'Chinese'},
            {'id': 'R003', 'price': 25, 'time_cost': 60, 'category': 'Seafood'},
            {'id': 'R004', 'price': 18, 'time_cost': 50, 'category': 'Italian'},
            {'id': 'R005', 'price': 22, 'time_cost': 55, 'category': 'American'},
            {'id': 'R006', 'price': 12, 'time_cost': 40, 'category': 'FastFood'},
            {'id': 'R007', 'price': 30, 'time_cost': 70, 'category': 'Fine Dining'},
            {'id': 'R008', 'price': 16, 'time_cost': 45, 'category': 'Japanese'},
            {'id': 'R009', 'price': 19, 'time_cost': 50, 'category': 'Mexican'},
            {'id': 'R010', 'price': 14, 'time_cost': 40, 'category': 'Cafe'},
        ],
        'attractions': [
            {'id': 'A001', 'price': 30, 'time_cost': 120, 'category': 'Museum'},
            {'id': 'A002', 'price': 25, 'time_cost': 90, 'category': 'Park'},
            {'id': 'A003', 'price': 40, 'time_cost': 150, 'category': 'Museum'},
            {'id': 'A004', 'price': 15, 'time_cost': 60, 'category': 'Beach'},
            {'id': 'A005', 'price': 35, 'time_cost': 100, 'category': 'Gallery'},
            {'id': 'A006', 'price': 20, 'time_cost': 80, 'category': 'Park'},
            {'id': 'A007', 'price': 50, 'time_cost': 180, 'category': 'ThemePark'},
            {'id': 'A008', 'price': 10, 'time_cost': 45, 'category': 'Market'},
        ]
    }
    
    sample_constraints = {
        'budget': 1500,
        'trip_days': 3,
        'max_daily_activity_time': 600,  # 10 hours in minutes
        'forbidden_categories': ['Seafood', 'Hostel'],
        'preferred_categories': ['Museum', 'Italian']
    }
    
    # Create and solve
    solver = TravelRouteSolver(sample_candidates, sample_constraints)
    result = solver.solve(time_limit_seconds=30)
    
    if result:
        import json
        print("\n" + "="*60)
        print("OPTIMAL ITINERARY FOUND")
        print("="*60)
        print(json.dumps(result, indent=2))
    else:
        print("\nNo feasible solution found. Try relaxing constraints.")
