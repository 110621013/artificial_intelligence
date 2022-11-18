import json
import argparse
from copy import deepcopy

ASSIGNED = 1
UNASSIGNED = 0


def is_safe(matrix: list, row: int, col: int) -> bool:
    n = len(matrix)

    # Check if it can be placed in this row
    row_data = matrix[row]
    if ASSIGNED in row_data:
        return False

    # Check if it can be placed in this col
    col_data = [temp_row_data[col] for temp_row_data in matrix]
    if ASSIGNED in col_data:
        return False

    # Check if it can be placed on this diagonal
    diff_count = min(row, col)
    row_start_index = row - diff_count
    col_start_index = col - diff_count
    diagonal_data_1 = [matrix[row_index][col_index] for row_index, col_index in
                       zip(range(row_start_index, n, 1), range(col_start_index, n, 1))]
    if ASSIGNED in diagonal_data_1:
        return False

    diff_count = min(n - row - 1, col)
    row_start_index = row + diff_count
    col_start_index = col - diff_count
    diagonal_data_2 = [matrix[row_index][col_index] for row_index, col_index in
                       zip(range(row_start_index, -1, -1), range(col_start_index, n, 1))]
    if ASSIGNED in diagonal_data_2:
        return False

    return True


def n_queen_solve(row: int) -> None:
    global puzzle_matrix
    global n_queen_result
    matrix = puzzle_matrix
    n = len(puzzle_matrix)
    # BEGIN_YOUR_CODE
    # Backtracking Algorithm
    '''
    就Backtracking，能放就放再往下做，不能時就向右找，沒有右時就退回去上一層找
    '''
    
    import copy
    row_step = 0
    while row_step >= 0:
        if ASSIGNED not in matrix[row_step]:
            puted_flag = False
            for col in range(n):
                # 可以放
                if is_safe(matrix, row_step, col):
                    matrix[row_step][col] = ASSIGNED
                    row_step += 1
                    puted_flag = True
                    break
                # 不可以放
                else:
                    continue
            # 這row全都不行
            if not puted_flag:
                row_step -= 1
        else:
            now_ASSIGNED_col = matrix[row_step].index(ASSIGNED)
            matrix[row_step][now_ASSIGNED_col] = UNASSIGNED
            puted_flag = False
            for col in range(now_ASSIGNED_col+1, n):
                # 可以放
                if is_safe(matrix, row_step, col):
                    matrix[row_step][col] = ASSIGNED
                    row_step += 1
                    puted_flag = True
                    break
                # 不可以放
                else:
                    continue
            # 這row全都不行
            if not puted_flag:
                row_step -= 1
            
        #若中ㄌ
        if row_step == n:
            n_queen_result.append(copy.deepcopy(matrix))
            row_step -= 1
        
    # END_YOUR_CODE


if __name__ == '__main__':
    puzzle_matrix = []
    n_queen_result = []

    parser = argparse.ArgumentParser()
    parser.add_argument('--puzzle_file_path', '-f', type=str, default='q3_sample_input.json')

    args = parser.parse_args()
    with open(args.puzzle_file_path, "r+") as fs:
        n_queen_info = json.load(fs)
        n_size = n_queen_info['n']

    # Create Puzzle
    for i in range(n_size):
        puzzle_matrix.append([0] * n_size)

    # Solve n Queen Problem
    n_queen_solve(0)

    for index, result in enumerate(n_queen_result):
        print('=', f'{index + 1:02d}', '='*10)
        for row_result in result:
            print(row_result)
        print()

    result_dict = {
        "length": len(n_queen_result),
        "result": n_queen_result
    }

    with open('n_queen_result.json', "w+") as fs:
        fs.write(json.dumps(result_dict, indent=4))
#python .\hw3_Constraint_Satisfaction_Problem\q3.py -f .\hw3_Constraint_Satisfaction_Problem\q3_sample_input.json