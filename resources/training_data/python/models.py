class State:

    def __init__(self, name, estimated_cost):
        self.name = name
        self.heuristic = 0
        self.estimated_cost = estimated_cost
        self.neighbours = []

    def add_neighbours(self, neighbours, ):
        for state in neighbours:
            self.neighbours.append({
                "state": state[0],
                "cost": state[1],
            })

    def __str__(self):
        return self.name

    def __repr__(self):
        return "{}: {}".format(self.name, self.heuristic)
