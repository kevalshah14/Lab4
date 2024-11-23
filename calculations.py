import math
x = -1.968
y = -2.061
x_a = -0.06
y_b = 0.08

# Calculate absolute distances
ans_r = abs(x - y)
ans_a = abs(x_a - y_b)

one_robot_unit = (ans_r / ans_a)

print( one_robot_unit )
top_righ_robot = (-174.793,-1.968,-0.90)