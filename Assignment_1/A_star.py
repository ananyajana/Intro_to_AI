from Heap import PriorityQueue

def reconstruct_path(parent, start, goal):
    current = goal
    path = [current]
    while current != start:
        current = parent[current]
        path.append(current)
    path.reverse()
    return path


def heuristic(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return abs(x1 - x2) + abs(y1 - y2)

def heuristic_new(g_value, goal, next):
	if (next not in g_value == False):
		return g_value[goal]-g_value[next]
	else:
		return heuristic(goal, next)

	
def a_star_search(graph, start, goal):
    frontier = PriorityQueue()
    frontier.put(start, 0)
    parent = {start: None}
    g_value = {start: 0}
    c_value = 10202

    while not frontier.empty():
        current = frontier.get()

        if current == goal:
            break
 
        for next in graph.neighbors(current):
            new_cost = g_value[current] + graph.cost(current, next)
            if next not in g_value or new_cost < g_value[next]:
                g_value[next] = new_cost
                priority = new_cost + heuristic(goal, next)
                # priority function c*f(s) - g(s) to break the ties on same f value in favor or larger g_value
                #priority = 500 * (new_cost + heuristic(goal, next)) - new_cost
                
                # priority function  = {c*f(s) + g(s)} to break the ties on same f value in favor or smaller g_value
                #priority = 500 * (new_cost + heuristic(goal, next)) + new_cost
                
                frontier.put(next, priority)
                parent[next] = current

    return parent, g_value

def adaptive_a_star(graph, start, goal, g_value):
    frontier = PriorityQueue()
    frontier.put(start, 0)
    parent = {start: None}
    g_value_new = {start: 0}
    c_value = 10202

    while not frontier.empty():
        current = frontier.get()

        if current == goal:
            break
 
        for next in graph.neighbors(current):
            new_cost = g_value_new[current] + graph.cost(current, next)
            if next not in g_value_new or new_cost < g_value_new[next]:
                g_value_new[next] = new_cost
                priority = new_cost + heuristic_new(g_value, goal, next)
                # priority function c*f(s) - g(s) to break the ties on same f value in favor or larger g_value
                #priority = 500 * (new_cost + heuristic(goal, next)) - new_cost
                
                # priority function  = {c*f(s) + g(s)} to break the ties on same f value in favor or smaller g_value
                #priority = 500 * (new_cost + heuristic(goal, next)) + new_cost
                
                frontier.put(next, priority)
                parent[next] = current

    return parent, g_value_new