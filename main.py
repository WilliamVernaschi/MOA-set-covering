import random
from dataclasses import dataclass
from functools import cmp_to_key
from math import ceil, log2, log as ln, inf
from copy import deepcopy
from typing import TypeVar

T = TypeVar('T')

# Since all costs have two decimal places, we multiply them by 100 to avoid floating point errors
WEIGHT_SCALING_FACTOR = 100

@dataclass
class Column:
    idx : int
    rows : list[int]
    cost : float

@dataclass 
class Matrix:
    data : list[Column]
    num_columns : int
    num_rows : int

@dataclass
class Solution:
    cost : float
    columns : list[int]

def flatten(xss: list[list[T]]) -> list[T]:
    """
    Flattens a list of lists into a single list.

    Parameters:
        xss (list[list[T]]): The list of lists to be flattened.

    Returns:
        list[T]: The flattened list.
    """
    return [
        x
        for xs in xss
        for x in xs
    ]


def read_input(file_name: str) -> list[list[str]]:
    """
    Read the input file and return a list of lists containing the data.

    Parameters:
        file_name (str): The name of the input file.

    Returns:
        list[list[str]]: A list of lists containing the data from the input file.
    """
    with open(file_name, "r") as f:
        return [line.split() for line in f]


def parse_input(inp : list[list[str]]) -> Matrix:
    """
    Parses the input and returns a Matrix object.

    Parameters:
        inp (list[list[str]]): The input data.

    Returns:
        Matrix: The parsed matrix object.

    Raises:
        ValueError: If the input format is invalid.
    """
    num_rows = int(inp[0][1])
    num_columns = int(inp[1][1])

    if inp[0][0].lower() != "linhas" \
            or not (inp[2][0].lower() == "densidade" or inp[2][0].lower() == "dados"):
        raise ValueError("Invalid input format")

    columns = [Column(int(column[0])-1, # idx
            [int(row)-1 for row in column[2:]], # rows
            int(WEIGHT_SCALING_FACTOR*float(column[1]))) # cost
               for column in inp[3:]]

    return Matrix(data=columns, num_columns=num_columns, num_rows=num_rows)

def get_total_coverings(solution : list[int], matrix : Matrix) -> list[int]:
    """
    Returns total_coverings, where, total_coverings[i] := how many columns in `solution` cover row `i`

    Parameters:
        solution (list[int]): The selected columns in the solution.
        matrix (Matrix): The matrix containing the data.

    Returns:
        list[int]: A list containing the total number of coverings for each row.
    """

    total_coverings = [0 for _ in range(matrix.num_rows)]

    for column in solution:
        for row in matrix.data[column].rows:
            total_coverings[row] += 1

    return total_coverings



def remove_useless_columns(solution : list[int], matrix : Matrix) -> None:
    """
    Removes useless columns from the solution based on the given matrix, this is done
    by considering the columns in decreasing order of cost, if a column, when removed, still results
    in a valid solution, remove it.

    Parameters:
        solution (list[int]): The list of column indices representing the solution.
        matrix (Matrix): The matrix containing the data.

    Returns:
        None
    """
    total_coverings = get_total_coverings(solution, matrix)

    # sort solution by cost
    solution.sort(key=cmp_to_key(lambda a, b: int(matrix.data[a].cost - matrix.data[b].cost)))

    useless = set()
    for column in solution:
        if min(total_coverings[row] for row in matrix.data[column].rows) > 1: 
            for to_remove in matrix.data[column].rows:                        
                total_coverings[to_remove] -= 1

            useless.add(column)

    solution[:] = [c for c in solution if c not in useless]


def get_total_weight(solution : list[int], matrix : Matrix) -> float:
    """
    Calculates the total weight of a given solution based on the cost values in the matrix.

    Parameters:
        solution (list[int]): The indices of the selected columns in the solution.
        matrix (Matrix): The matrix containing the cost values.

    Returns:
        float: The total weight of the solution.
    """
    return sum(matrix.data[column_idx].cost for column_idx in solution)/WEIGHT_SCALING_FACTOR

greedy_options = [
        lambda cj, kj : cj,
        lambda cj, kj : cj/kj,
        lambda cj, kj : cj/log2(kj) if log2(kj) != 0 else 1e9,
        lambda cj, kj : cj/(kj*log2(kj)) if (kj*log2(kj)) != 0 else 1e9,
        lambda cj, kj : cj/(kj*ln(kj)) if kj*ln(kj) != 0 else 1e9,
        lambda cj, kj : cj/(kj*kj),
        lambda cj, kj : cj**(1/2)/(kj*kj)]



def min_set_cover_constructive(matrix : Matrix) -> Solution:
    """
    Construct a minimum set cover solution using a constructive greedy algorithm.

    Parameters:
        matrix (Matrix): The input matrix representing the set cover problem.

    Returns:
        Solution: The minimum set cover solution, containing the total weight and the sorted set of chosen columns.
    """
    S = [] # solution set
    num_covered = 0 # number of rows already covered
    row_covered = [False for i in range(matrix.num_rows)]
    # row_covered[i] := has row `i` been covered by some column?

    while num_covered < matrix.num_rows:
        greedy = random.choice(greedy_options)

        best_pair = (inf, -1) # pair of cost of choosing a column and its index
                              # initially, no column is chosen.

        for column in matrix.data:
            num_rows_covered_by_column = sum(1 for row in column.rows if not row_covered[row])
            if num_rows_covered_by_column == 0:
                continue
            best_pair = min(best_pair, (greedy(column.cost, num_rows_covered_by_column), column.idx))


        chosenColumn = best_pair[1]

        assert chosenColumn != -1 # If this assertion fails, there is no viable solution.

        for row in matrix.data[chosenColumn].rows:
            if row_covered[row]:
                continue
            row_covered[row] = True
            num_covered += 1

        S.append(chosenColumn)

    remove_useless_columns(S, matrix)

    return Solution(get_total_weight(S, matrix), sorted(S))

def min_set_cover_search(matrix : Matrix, solution : list[int], ρ1 : float, ρ2 : float) -> Solution:
    """
    Search for a minimum set cover solution using a local search algorithm, as described by 
    Jacobs, L. W. and Brusco.

    Parameters:
        matrix (Matrix): The matrix representing the set cover problem.
        solution (list[int]): The initial solution.
        ρ1 (float): The parameter controlling the number of columns to be removed.
        ρ2 (float): The parameter controlling the weight limit on the columns that will be added.

    Returns:
        Solution: The minimum set cover solution.

    Raises:
        None.
    """

    D = ceil(ρ1 * len(solution))
    # D := number of columns to be removed
    E = ceil(ρ2 * max(matrix.data[column_idx].cost for column_idx in solution))
    # E := weight limit on the columns that will be added


    total_coverings = get_total_coverings(solution, matrix)

    # R := set of all columns that are not in S
    R = {c for c in range(matrix.num_columns)} 
    for c in solution:
        R.remove(c)

    # instead of removing D random columns, just shuffle the solution list
    # and get the last D.
    random.shuffle(solution)
    for _ in range(D):

        random_column = solution[-1]
        R.add(random_column)

        for row in matrix.data[random_column].rows:
            total_coverings[row] -= 1

        solution.pop()

    uncovered_rows = {row_idx for row_idx in range(matrix.num_rows) if total_coverings[row_idx] == 0}
    # uncovered_rows := set of all rows that aren't covered by any column

    while len(uncovered_rows) > 0:

        substitute_column_candidates = [column_idx for column_idx in R if matrix.data[column_idx].cost <= E]

        α = [sum(1 for i in matrix.data[column_idx].rows if i in uncovered_rows) \
                for column_idx in substitute_column_candidates]
        β = [matrix.data[col_idx].cost/αj if αj > 0 else inf for col_idx,αj in enumerate(α)]
        
        if len(β) == 0 or min(βj for βj in β) == inf:
            return Solution(inf, []) # There was a bad choice of ρ2
        
        βmin = min(βj for βj in β)

        K = [c for idx,c in enumerate(substitute_column_candidates) if β[idx] == βmin]
        new_chosen_column = random.choice(K)

        for row in matrix.data[new_chosen_column].rows:
            if total_coverings[row] == 0:
                uncovered_rows.remove(row)
            total_coverings[row]+=1

        solution.append(new_chosen_column)
        R.remove(new_chosen_column)

    remove_useless_columns(solution, matrix)

    return Solution(get_total_weight(solution, matrix), solution)


def min_set_cover_constructive2(matrix : Matrix) -> Solution:
    """
    Construct a minimum set cover solution by cosidering the columns in increasing weight.

    Parameters:
        matrix (Matrix): The input matrix representing the set cover problem.

    Returns:
        Solution: The minimum set cover solution.

    """
    S = []  # solution set
    num_covered = 0  # number of rows already covered
    row_covered = [False for i in range(matrix.num_rows)]
    # row_covered[i] := has row `i` been covered by some column?

    # sort columns by weight in increasing order
    sorted_columns = sorted(matrix.data, key=lambda column: column.cost)

    for column in sorted_columns:
        for row in column.rows:
            if not row_covered[row]:
                row_covered[row] = True
                num_covered += 1

        S.append(column.index)

        if num_covered == matrix.num_rows:
            break

    return Solution(get_total_weight(S, matrix), S)

def best_improvement(solution : Solution, n : int) -> Solution:
    """
    Improves the given solution by applying a local search algorithm n times and
    finding the best improvement out of every n/20 searches.

    Parameters:
        solution (Solution): The initial solution to be improved.
        n (int): The number of iterations for the local search algorithm.

    Returns:
    Solution: The improved solution.
    """
    
    improvement = deepcopy(solution)
    for i in range(n):
        ρ1 = random.uniform(0.4, 0.9)
        ρ2 = random.uniform(0.9, 1.5)
        current_search = min_set_cover_search(matrix, deepcopy(solution.columns), ρ1, ρ2)
        if current_search.cost < improvement.cost:
            improvement = deepcopy(current_search)
        if i % (n//20) == 0 and improvement.cost < solution.cost:
            solution = deepcopy(improvement)

    return solution


def next_improvement(solution : Solution, n : int) -> Solution:
    """
    Improves the given solution by applying a local search algorithm n times and
    finding the next improvement.

    Parameters:
        solution (Solution): The initial solution to be improved.
        n (int): The number of iterations for the local search algorithm.

    Returns:
    Solution: The improved solution.
    """

    for _ in range(n):
        ρ1 = random.uniform(0.4, 0.9)
        ρ2 = random.uniform(0.9, 1.5)
        current_search = min_set_cover_search(matrix, deepcopy(solution.columns), ρ1, ρ2)
        if current_search.cost < solution.cost:
            solution = deepcopy(current_search)

    return solution

def solve(matrix : Matrix) -> Solution:
    """
    Solves the set covering problem using a constructive heuristic and local search.

    Parameters:
        matrix (Matrix): The input matrix representing the set covering problem.

    Returns:
        Solution: The best solution found by the algorithm.
    """
    initial_solution = min_set_cover_constructive(matrix)
    
    return next_improvement(initial_solution, 1000)



if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python main.py <input_file>")
        sys.exit(1)

    
    input_file = sys.argv[1]
    inp = read_input(input_file)
    matrix = parse_input(inp)

    solution = solve(matrix)

    print(f"Custo: {solution.cost}")
    print(f"Colunas: {sorted([c+1 for c in solution.columns])}")


    
