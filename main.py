from dataclasses import dataclass
from math import inf
from functools import cmp_to_key
import random
from math import log2
from math import log as ln

@dataclass
class Column:
    idx : int
    lines : list[int]
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
            [int(row)-1 for row in column[2:]], # lines
            int(100*float(column[1]))) # cost
               for column in inp[3:]]

    return Matrix(data=columns, num_columns=num_columns, num_rows=num_rows)

# given a solution, consider the columns in decreasing 
# order of cost, if a column, when removed, still results
# in a valid solution, remove it, modifies the solution list in-place
def remove_useless_columns(solution : list[int], matrix : Matrix) -> None:
    total_coverings = [0 for i in range(matrix.num_rows)]
    # total_coverings[i] := how many columns in `solution` cover row `i`

    for column in solution:
        for row in matrix.data[column].lines:
            total_coverings[row] += 1

    # sort solution by cost
    solution.sort(key=cmp_to_key(lambda a, b: 1 if matrix.data[a].cost - matrix.data[b].cost < 0 else -1))

    useless = set()
    for column in solution:
        if min(total_coverings[row] for row in matrix.data[column].lines) > 1: 
            for to_remove in matrix.data[column].lines:                        
                total_coverings[to_remove] -= 1

            useless.add(column)

    solution[:] = [c for c in solution if c not in useless]


# given a list of chosen columns, returns the total weight of the solution
def get_total_weight(solution : list[int], matrix : Matrix) -> float:
    return sum(matrix.data[column_idx].cost for column_idx in solution)

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
    numCovered = 0 # number of rows already covered
    row_covered = [False for i in range(matrix.num_rows)]
    # row_covered[i] := has row `i` been covered by some column?

    while numCovered < matrix.num_rows:
        best_pair = (inf, -1) # pair of cost of choosing a column and its index
                              # initially, no column is chosen.

        for column in matrix.data:
            num_rows_covered_by_column = sum(1 for row in column.lines if not row_covered[row])
            if num_rows_covered_by_column == 0:
                continue
            best_pair = min(best_pair, (greedy_cost(column.cost, num_rows_covered_by_column), column.idx))

        chosenColumn = best_pair[1]

        assert chosenColumn != -1 # some column must be chosen
                                  # if this assertion fails, there is no viable solution.

        for row in matrix.data[chosenColumn].lines:
            if row_covered[row]:
                continue
            row_covered[row] = True
            numCovered += 1

        S.append(chosenColumn)

    remove_useless_columns(S, matrix) # removes useless columns from S

    return (get_total_weight(S, matrix)/100, sorted(S))

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python main.py <input_file>")
        sys.exit(1)

    input_file = sys.argv[1]

    inp = read_input(input_file)
    matrix = parseInput(inp)
    total_cost, chosen_columns = min_set_cover(matrix)

    print(f"Custo total: {total_cost}")
    print(f"Colunas escolhidas: {chosen_columns}")
