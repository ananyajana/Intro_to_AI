import numpy;
import random;
import xlsxwriter;

count = int(0.3*101*101)
print (count)
#print(grid)
for x in range(1,51):
	grid = numpy.ones((101,101))
	i=0
	while(i<count):
		k = random.randint(0,100)
		l = random.randint(0,100)
		if(grid[k][l]== 1):
			grid[k][l] = 0
			i= i+1
	#print (grid)
	
	filename = 'grid'+str(x)+'.xlsx'
	workbook = xlsxwriter.Workbook(filename)
	worksheet = workbook.add_worksheet()

	row = 0

	for col, data in enumerate(grid):
		worksheet.write_column(row, col, data)

	workbook.close()