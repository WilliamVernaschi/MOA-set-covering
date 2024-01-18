from dataclasses import dataclass
from functools import cmp_to_key
import random
from math import ceil, log2, log as ln, inf
from more_itertools import ilen as count
from copy import deepcopy

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


# returns a list of a list of strings
def read_input(file_name : str) -> list[list[str]]:
    with open(file_name, "r") as f:
        return [line.split() for line in f]

# given the input, returns the Matrix
def parseInput(inp : list[list[str]]) -> Matrix:
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

# returns the list total_coverings
# total_coverings[i] := how many columns in `solution` cover row `i`
def get_total_coverings(solution : list[int], matrix : Matrix) -> list[int]:

    total_coverings = [0 for i in range(matrix.num_rows)]

    for column in solution:
        for row in matrix.data[column].rows:
            total_coverings[row] += 1

    return total_coverings


# given a solution, consider the columns in decreasing 
# order of cost, if a column, when removed, still results
# in a valid solution, remove it. Modifies the solution list in-place
def remove_useless_columns(solution : list[int], matrix : Matrix) -> None:

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


# given a list of chosen columns, returns the total weight of the solution
def get_total_weight(solution : list[int], matrix : Matrix) -> float:
    return sum(matrix.data[column_idx].cost for column_idx in solution)/WEIGHT_SCALING_FACTOR

greedy_options = [
        lambda cj, kj : cj,
        lambda cj, kj : cj/kj,
        lambda cj, kj : cj/log2(kj) if log2(kj) != 0 else 1e9,
        lambda cj, kj : cj/(kj*log2(kj)) if (kj*log2(kj)) != 0 else 1e9,
        lambda cj, kj : cj/(kj*ln(kj)) if kj*ln(kj) != 0 else 1e9,
        lambda cj, kj : cj/(kj*kj),
        lambda cj, kj : cj**(1/2)/(kj*kj)]

# sorts one of the greedy functions and uses it
def random_greedy_cost(weight : float, num_rows_covered_by_column : int) -> float:
    return random.choice(greedy_options)(weight, num_rows_covered_by_column)

def greedy_cost(weight : float, num_rows_covered_by_column : int) -> float:
    return weight/num_rows_covered_by_column


# returns a pair of the total cost and the chosen columns
def min_set_cover(matrix : Matrix) -> tuple[float, list[int]]:
    S = [] # solution set
    num_covered = 0 # number of rows already covered
    row_covered = [False for i in range(matrix.num_rows)]
    # row_covered[i] := has row `i` been covered by some column?

    while num_covered < matrix.num_rows:
        best_pair = (inf, -1) # pair of cost of choosing a column and its index
                              # initially, no column is chosen.

        for column in matrix.data:
            num_rows_covered_by_column = sum(1 for row in column.rows if not row_covered[row])
            if num_rows_covered_by_column == 0:
                continue
            best_pair = min(best_pair, (random_greedy_cost(column.cost, num_rows_covered_by_column), column.idx))

        chosenColumn = best_pair[1]

        assert chosenColumn != -1 # If this assertion fails, there is no viable solution.

        for row in matrix.data[chosenColumn].rows:
            if row_covered[row]:
                continue
            row_covered[row] = True
            num_covered += 1

        S.append(chosenColumn)

    remove_useless_columns(S, matrix)

    return (get_total_weight(S, matrix), sorted(S))

def min_set_cover_search(matrix : Matrix, solution : list[int]) -> tuple[float, list[int]]:
    # Set d = 0,
    rho1 = random.uniform(0.05, 0.7)
    rho2 = random.uniform(1.1, 2)
    # D = rho1*N(S), 
    D = ceil(rho1 * len(solution))
    # E = rho2*Q(s)
    E = ceil(rho2 * max(matrix.data[column_idx].cost for column_idx in solution))

    random.shuffle(solution)
    total_coverings = get_total_coverings(solution, matrix)

    R = {c for c in range(matrix.num_columns)}
    for c in solution:
        R.remove(c)

    for _ in range(D):

        random_column = solution[-1]
        R.add(random_column)

        for row in matrix.data[random_column].rows:
            total_coverings[row] -= 1

        solution.pop()

    uncovered_rows = {row_idx for row_idx in range(matrix.num_rows) if total_coverings[row_idx] == 0}


    while len(uncovered_rows) > 0:
        substitute_column_candidates = [column_idx for column_idx in R if matrix.data[column_idx].cost <= E]


        α = [sum(1 for i in matrix.data[column_idx].rows if i in uncovered_rows) \
                for column_idx in substitute_column_candidates]

        β = [matrix.data[col_idx].cost/αj if αj > 0 else inf for col_idx,αj in enumerate(α)]

        bmin = min(βj for βj in β)

        assert bmin < inf


        K = [c for idx,c in enumerate(substitute_column_candidates) if β[idx] == bmin]

        new_chosen_column = random.choice(K)

        for row in matrix.data[new_chosen_column].rows:
            if total_coverings[row] == 0:
                uncovered_rows.remove(row)
            total_coverings[row]+=1

        solution.append(new_chosen_column)
        R.remove(new_chosen_column)


    remove_useless_columns(solution, matrix)

    return (get_total_weight(solution, matrix), solution)


# this is really bad, will remove it
def min_set_cover_constructive2(matrix : Matrix):
    num_groups = random.randint(15,25)

    group_size = ceil(matrix.num_columns / num_groups)

    matrix.data.sort(key=cmp_to_key(lambda ra, rb : sum(r for r in ra.rows) - sum(r for r in rb.rows)))

    groups = [list() for _ in range(num_groups)]

    for idx, column in enumerate(matrix.data):
        groups[idx%num_groups].append(column)


    groups_selected = [False for _ in range(num_groups)]

    def get_solution(groups_selected : list[bool]) -> tuple[float, list[int]]:
        columns = [column.idx
                   for column_group in (groups[i] for i in range(num_groups) if groups_selected[i])
                   for column in column_group]

        if not valid_solution(columns, matrix):
            return (inf, [])


        return (sum(matrix.data[column_idx].cost for column_idx in columns), columns)


    ans = inf

    print(num_groups)
    def try_all(group_idx : int):
        if group_idx == num_groups:
            return get_solution(groups_selected)[0]
        else:
            groups_selected[group_idx] = True
            a1 = try_all(group_idx+1)
            groups_selected[group_idx] = False
            a2 = try_all(group_idx+1)
            return min(a1, a2)

    return try_all(0)/100










def valid_solution(solution : list[int], matrix : Matrix):
    rows = [False for _ in range(matrix.num_rows)]

    for c in solution:
        for r in matrix.data[c].rows:
            rows[r] = True


    return all(f for f in rows)



if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python main.py <input_file>")
        sys.exit(1)

    
    input_file = sys.argv[1]
    inp = read_input(input_file)
    matrix = parseInput(inp)

    best_solution : tuple[float, list[int]] = (inf, [])
    total_cost, chosen_columns = min_set_cover(matrix)

    for _ in range(20000):

        current_solution = min_set_cover_search(matrix, chosen_columns)

        if current_solution[0] < best_solution[0]:
            best_solution = deepcopy(current_solution)
            print(best_solution)

        chosen_columns = deepcopy(best_solution[1])

    assert valid_solution(best_solution[1], matrix)
    print(f"Custo: {best_solution[0]}")
    print(f"Colunas: {sorted([c+1 for c in best_solution[1]])}")


    
