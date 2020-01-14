import numpy as np
import random

from pprint import pprint

df_x = [
	[1, 1, 1],
	[1, 0, 1],
	[0, 0, 1],
	[1, 1, 0]
]

df_y = [1, 1, 0, 1]

def main():
	thetas = init_thetaas()
	final_classes = []
	print()
	for xi, yi in zip(df_x, df_y):
		print('xi:', xi, 'yi:', yi)
		A = [xi]
		A[-1].insert(0, 1) #insert bias as zero element
		A = forward(A, thetas)
		# output layer calculation
		res = sum(val * weight[0] for val, weight in zip(A[-1], thetas[-1]))
		A.append([res])
		print('A:')
		pprint(A)
		deltas = backprop(A, yi, thetas)[1:]
		deltas[-1] = [deltas[-1]]
		print('deltas')
		pprint(deltas)
		thetas = update_thetas(thetas, A, deltas)
		final_classes.append(activ(res))
		break


def update_thetas(thetas, A, deltas):
	A_ = A[1:]
	# print('A_:')
	# pprint(A_)
	new_thetas = []
	for theta_n in thetas:
		new_theta = []
		for theta_n_i, a_i, delta_i in zip(theta_n, A_, deltas):
			# print()
			# print(theta_n_i, a_i, delta_i)
			temp = []
			if len(delta_i) != 1:
				for theta_n_ij, a_ij, delta_ij in zip(theta_n_i, a_i, delta_i):
					temp.append(theta_n_ij + a_ij * delta_ij)
			else:
				for theta_n_i_j in theta_n_i:
					temp.append(theta_n_ij + a_i[0] * delta_i[0])
			new_theta.append(temp)
		new_thetas.append(new_theta)

	print('new_thetas:')
	for el in new_thetas:
		pprint(el)
		print()		


def backprop(A, yi, thetas):
	deltas = [A[-1][0] - yi]
	for ai, theta_n in zip(A[:-1][::-1], thetas[::-1]):
		# print()
		next_delta = deltas[0]
		# print('prev_delta')
		# print(prev_delta)
		interm = [] #means intermediate result
		if len(theta_n[0]) == 1:
			# print("1")
			for theta_n_i in theta_n:
				interm.append(theta_n_i[0] * next_delta)
		else:
			# print("theta_n:")
			# print(theta_n)
			# print('next_delta')
			# print(next_delta)
			# print()
			next_delta = next_delta[1:]
			summa = 0
			for j in range(len(theta_n[0])): # it should be matrix transposing
				for next_delta_i, theta_n_i in zip(next_delta, theta_n):
					summa += next_delta_i * theta_n_i[j]
				interm.append(summa)
		# print('interm:' ,interm)
		g_z = []
		for ai_j in ai:
			g_z.append(ai_j * (1 - ai_j))
		# print('g_z:', g_z)
		delta_i = []
		for interm_i, g_z_i in zip(interm, g_z):
			delta_i.append(interm_i * g_z_i)
		# print('delta(i)', delta_i)
		deltas.insert(0, delta_i)
	return deltas



def forward(A, thetas):
	pprint(thetas)
	for theta in thetas[:-1]:
			prev_layer_vals = A[-1]
			res = [1] #add bias
			for theta_n, val, i in zip(theta, prev_layer_vals, range(3)):
				# print('theta_n')
				# print(theta_n)
				a_i = sum([val * weight for weight in theta_n])
				# print("a_i")
				# print(a_i)
				res.append(a_i)
			A.append(res)
	return A


def init_thetaas():
	random.seed(42)
	theta1 = []
	for i in range(3):
		row = [round(random.random(), 3) for _ in range(4)]
		theta1.append(row)
	#
	print('theta1')
	for el in theta1:
		print(el)
	#
	theta2 = []
	for _ in range(3):
		row = [round(random.random(), 3) for _ in range(4)]
		theta2.append(row)
	print('theta2')
	pprint(theta2)

	theta3 = []
	for _ in range(4):
		row = [round(random.random(), 3)]
		theta3.append(row)
	print('theta3')
	pprint(theta3)
	return theta1, theta2, theta3

def activ(x):
	return 0.5 if x >= 1 else 0 


	

if __name__ == '__main__':
	main()
