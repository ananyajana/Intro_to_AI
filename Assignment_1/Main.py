from A_star import *
from Graph import *
from Tools import *

def a_star_search_example():
	ks = random.randint(0,100)
	ls = random.randint(0,100)
	while((ks, ls) not in diagram4.walls == False):
		ls = random.randint(0,100)
	kg = random.randint(0,100)
	lg = random.randint(0,100)
	while(lg == ls and kg == ks ):
		lg = random.randint(0,100)
	while((kg, lg) not in diagram4.walls == False):
		lg = random.randint(0,100)
		while(lg == ls and kg == ks ):
			lg = random.randint(0,100)
	came_from, cost_so_far = a_star_search(diagram4, (ks, ls), (kg, lg))
	came_from_adaptive, cost_so_far_adaptive = adaptive_a_star(diagram4, (ks, ls), (kg, lg), cost_so_far)
	draw_grid(diagram4, width=1, point_to=came_from, start=(ks, ls), goal=(kg, lg))
	print('Adaptive A Structure')
	draw_grid(diagram4, width=1, point_to=came_from_adaptive, start=(ks, ls),  goal=(kg, lg))

if __name__ == '__main__':
    #draw_square_grid_example()
    draw_grid(diagram4, width=1)
    print('A Star on 101x101 grid')
    a_star_search_example()
