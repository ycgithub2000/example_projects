import random
import math 
from BaseAI_3 import BaseAI

# class PlayerAI(BaseAI):
#     def getMove(self, grid):
#     	# Selects a random move and returns it
#     	moveset = grid.getAvailableMoves()
#     	return random.choice(moveset)[0] if moveset else None

def terminalTest(grid, isPlayer):
	if isPlayer:
		return not grid.canMove()  # Player cannot move anymore
	else: # computer's turn
		return len(grid.getAvailableCells()) == 0  # no empty cell available

def eval(grid):
	return grid.getMaxTile()

def getChildren(grid, isPlayer, possibleNewTiles):
	if isPlayer:
		move_and_grid = grid.getAvailableMoves()  # List[(int, Grid)]
		return [x[1] for x in move_and_grid]
	else: # computer's turn
		pos_list = grid.getAvailableCells()  # list of positions for empty cells
		return_list = [[grid.clone() for _value in possibleNewTiles] for _pos in pos_list]
		for i in range(len(pos_list)):
			for j in range(len(possibleNewTiles)):
				return_list[i][j].setCellValue(pos_list[i], possibleNewTiles[j])
		return return_list


# only count left to right increase, and up to down increase
def calculateNumMonoton2(grid):
	N = grid.size
	board = grid.map

	# left -> right
	left_right = 0
	for i in range(N):
		consect = True
		for j in range(1, N):
			if board[i][j] < board[i][j-1]:
				consect = False
				break
		if consect:
			left_right += 1

	# up -> down
	up_down = 0
	for j in range(N):
		consect = True
		for i in range(1, N):
			if board[i][j] < board[i-1][j]:
				consect = False
				break
		if consect:
			up_down += 1

	return left_right + up_down


# if second largest is next to the largest tile
def isSecondAdjacent(grid):
	max_val = grid.getMaxTile()
	N = grid.size
	board = grid.map

	second_candidate = []

	for row in board:
		for x in row:
			if x < max_val:
				second_candidate.append(x)

	if len(second_candidate) == 0:  # all elements are of max_val
		return 1.0

	sec_val = max(second_candidate)

	for i in range(N):
		for j in range(N):
			if board[i][j] == sec_val:
				if (i > 0 and board[i-1][j] == max_val) or \
				   (j > 0 and board[i][j-1] == max_val) or \
				   (i < N-1 and board[i+1][j] == max_val) or \
				   (j < N-1 and board[i][j+1] == max_val):
					return 1.0
	return 0.0


# count number of merges
def calculateMergePotential(grid):
	N = grid.size
	board = grid.map

	# left -> right
	left_right = 0
	for i in range(N):
		for j in range(1, N):
			if board[i][j] == board[i][j-1]:
				left_right += 1

	# right -> left
	right_left = 0
	for i in range(N):
		for j in range(N-2, -1, -1):
			if board[i][j] == board[i][j+1]:
				right_left += 1

	# up -> down
	up_down = 0
	for j in range(N):
		for i in range(1, N):
			if board[i][j] == board[i-1][j]:
				up_down += 1

	# down -> up
	down_up = 0
	for j in range(N):
		for i in range(N-2, -1, -1):
			if board[i][j] == board[i+1][j]:
				down_up += 1

	return (left_right + right_left + up_down + down_up) / 4.0


def heuristicScore(grid, isPlayer):
	if terminalTest(grid, isPlayer):
		return eval(grid)

	actual_score = grid.getMaxTile()
	num_empty_cells = len(grid.getAvailableCells())
	merge_score = calculateMergePotential(grid)
	monoton_num = calculateNumMonoton2(grid)
	is_second_adjacent = isSecondAdjacent(grid)

	log_act_score = math.log(actual_score)

	cur_score = actual_score + log_act_score *(
				1.0 * num_empty_cells +
				0.8 * merge_score +
				1.0 * monoton_num +
				0.8 * is_second_adjacent)

	return cur_score


# Player: returns child, utility
def maximizer(grid, depth, alpha, beta, possibleNewTiles, probability):
	isPlayer = True

	if depth == 0 or terminalTest(grid, isPlayer):
		return None, heuristicScore(grid, isPlayer)

	max_child, max_utility = None, float('-inf')
	for child in getChildren(grid, isPlayer, possibleNewTiles):
		_, utility = minimizer(child, depth-1, alpha, beta, possibleNewTiles, probability)

		if utility > max_utility:
			max_child, max_utility = child, utility

		if max_utility >= beta:
			break

		if max_utility > alpha:
			alpha = max_utility

	return max_child, max_utility


# Computer: returns child, utility
def minimizer(grid, depth, alpha, beta, possibleNewTiles, probability):
	isPlayer = False

	if depth == 0 or terminalTest(grid, isPlayer):
		return None, heuristicScore(grid, isPlayer)

	min_child, min_utility = None, float('+inf')

	# each child is a position fill with all possible values
	for child in getChildren(grid, isPlayer, possibleNewTiles):
		utilities = [maximizer(sub_child, depth-1, alpha, beta, possibleNewTiles, probability)[1]
					 for sub_child in child]  # for each position, fill value

		utility = sum([u*p for u, p in zip(utilities, probability)])  # weighted average of utilities

		if utility < min_utility:
			min_child, min_utility = child, utility

		if min_utility <= alpha:
			break

		if min_utility < beta:
			beta = min_utility

	return min_child, min_utility


class PlayerAI(BaseAI):
	def __init__(self):
		self.max_depth = 3  # max depth for searching
		self.possibleNewTiles = [2, 4]
		self.probability = [0.9, 0.1]

	def getMove(self, grid):
		child, _ = maximizer(grid, self.max_depth, float('-inf'), float('+inf'),
							 self.possibleNewTiles, self.probability)

		moveset = grid.getAvailableMoves()
		for move_int, next_grid in moveset:
			if child.map == next_grid.map:
				return move_int

		return None





