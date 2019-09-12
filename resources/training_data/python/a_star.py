def choose_state(frontier):
    lower_cost = frontier[0]
    for node in frontier:
        if node['cost'] < lower_cost['cost']:
            lower_cost = node
    frontier.remove(lower_cost)
    return lower_cost


def search(initial_state, goal):
    frontier = [{
        'state': initial_state,
        'cost': 0,
        'parent': None
    }]
    explored = set()

    while True:
        # print(frontier)
        if len(frontier) <= 0:
            return False

        chosen = choose_state(frontier)
        explored.add(chosen['state'])

        if chosen['state'] == goal:
            return chosen

        for neighbour in chosen['state'].neighbours:
            path_cost = neighbour['cost'] + chosen['cost'] + neighbour['state'].estimated_cost

            if neighbour['state'] in explored:
                continue

            else:
                flag_found_state = False
                for node in frontier:
                    if neighbour['state'] == node['state']:
                        flag_found_state = True

                        if node['cost'] > path_cost:
                            frontier.remove(node)
                            frontier.append({
                                'state': neighbour['state'],
                                'cost': path_cost,
                                'parent': chosen
                            })
                            neighbour['state'].heuristic = path_cost

                if not flag_found_state:
                    frontier.append({
                        'state': neighbour['state'],
                        'cost': path_cost,
                        'parent': chosen
                    })
                    neighbour['state'].heuristic = path_cost
