from pandas import DataFrame
from benchmarks.travelplanner.official.path_utils import resolve_database_path


class Cities:
    def __init__(self ,path=None) -> None:
        self.path = path or resolve_database_path("background", "citySet_with_states.txt")
        self.load_data()
        print("Cities loaded.")

    def load_data(self):
        cityStateMapping = open(self.path, "r").read().strip().split("\n")
        self.data = {}
        for unit in cityStateMapping:
            city, state = unit.split("\t")
            if state not in self.data:
                self.data[state] = [city]
            else:
                self.data[state].append(city)
    
    def run(self, state) -> dict:
        if state not in self.data:
            return ValueError("Invalid State")
        else:
            return self.data[state]
