from Graph import *
import random;


def draw_tile(graph, id, style, width):
    r = "."
    if 'number' in style and id in style['number']:
        r = "%d" % style['number'][id]
    if 'point_to' in style and style['point_to'].get(id, None) is not None:
        (x1, y1) = id
        (x2, y2) = style['point_to'][id]
        if x2 == x1 + 1:
            r = "\u2192"
        if x2 == x1 - 1:
            r = "\u2190"
        if y2 == y1 + 1:
            r = "\u2193"
        if y2 == y1 - 1:
            r = "\u2191"
    if 'start' in style and id == style['start']:
        r = "A"
    if 'goal' in style and id == style['goal']:
        r = "Z"
    if 'path' in style and id in style['path']:
        r = "@"
    if id in graph.walls:
        r = "#" * width
    return r


def draw_grid(graph, width=2, **style):
    for y in range(graph.height):
        for x in range(graph.width):
            print("%%-%ds" % width % draw_tile(graph, (x, y), style, width), end="")
        print()

diagram4 = GridWithWeights(101, 101)
count = (.3*101*101)
a = []
print (count)
i=0
while(i<count):
	k = random.randint(0,100)
	l = random.randint(0,100)
	id = (k,l)
	if (id not in a):
		a.append(id)
		i = i+1
	

diagram4.walls = a
#print(a)