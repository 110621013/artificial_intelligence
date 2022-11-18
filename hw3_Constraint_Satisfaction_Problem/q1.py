import argparse
import json

UNASSIGNED = 0


def is_safe(matrix: list, row: int, col: int, num: int) -> bool:

    # Check if it can be placed in this row
    if num in matrix[row]:
        return False

    # Check if it can be placed in this col
    if num in [row_data[col] for row_data in matrix]:
        return False

    # Check if it can be placed in this box
    row_start_index = row - (row % SUDOKU_BOX_SIZE)
    col_start_index = col - (col % SUDOKU_BOX_SIZE)
    for tmp_row_index in range(SUDOKU_BOX_SIZE):
        tmp_row = [matrix[row_start_index + tmp_row_index][col_start_index + tmp_col_index]
                   for tmp_col_index in range(SUDOKU_BOX_SIZE)]
        if num in tmp_row:
            return False

    return True


def is_completed(matrix: list) -> bool:
    for row_data in matrix:
        if UNASSIGNED in row_data:
            return False
    return True


def sudoku_solve() -> bool:
    global SUDOKU_MATRIX
    matrix = SUDOKU_MATRIX
    row_index = 0
    col_index = 0
    
    # BEGIN_YOUR_CODE
    
    '''
    兩種做法：
    1.唯一解消除法:當一個數獨內有只能填一個數的格子時，就填那個數並且去除對應行列格格子填該數的可能性，以此類推直至沒有可填確定值的格子，不一定可完全解
    2.DFS法:找出最少選擇的格子並分支成所有選擇的DFS，並執行1的作法重複至找出對的matrix，一定可完全解
    我的作法是執行1去針對簡單數獨直接解決，若不行解決就DFS，應該為最高效率的算法
    '''

    # 唯一解消除法
    #######################################################
    # 找出所有0點的可能性, 並且記錄只有一種可能的位置
    matrix_options = [] #紀錄所有0點的可能性
    only1_position = [] #紀錄只有一種可能的0點位置
    only1_num = []      #紀錄只有一種可能的0點數字
    for row in range(len(matrix)):
        col_options = []
        for col in range(len(matrix[0])):
            point_options = []
            if matrix[row][col] == UNASSIGNED:
                for num in range(1, 10):
                    if is_safe(matrix, row, col, num):
                        point_options.append(num)
                #若無解
                if len(point_options) == 0:
                    return False
                #若只有一種可能
                elif len(point_options) == 1:
                    only1_position.append((row, col))
                    only1_num.append(point_options[0])
            col_options.append(point_options)
        matrix_options.append(col_options)

    #print('matrix         ---> ', matrix)
    #print('matrix_options ---> ', matrix_options)
    #print('only1_position ---> ', only1_position)
    #print('only1_num      ---> ', only1_num)
    
    # 針對每個確定值給入並刪去其同行同列跟同box的可能，解完跳出
    while True:
        if len(only1_position) == 0:
            break
        
        new_only1_position = []
        new_only1_num = []
        #唯一值填入
        for i in range(len(only1_position)):
            matrix[only1_position[i][0]][only1_position[i][1]] = only1_num[i]
            matrix_options[only1_position[i][0]][only1_position[i][1]] = []
        
            #列消除可能
            for col in range(len(matrix_options[0])):
                options_list = matrix_options[only1_position[i][0]][col]
                if only1_num[i] in options_list:
                    options_list.remove(only1_num[i])
                if len(options_list) == 1:
                    new_only1_position.append((only1_position[i][0], col))
                    new_only1_num.append(options_list[0])
                    matrix_options[only1_position[i][0]][col] = []
            #行消除可能
            for row in range(len(matrix_options)):
                options_list = matrix_options[row][only1_position[i][1]]
                if only1_num[i] in options_list:
                    options_list.remove(only1_num[i])
                if len(options_list) == 1:
                    new_only1_position.append((row, only1_position[i][1]))
                    new_only1_num.append(options_list[0])
                    matrix_options[row][only1_position[i][1]] = []
            #塊消除可能(注意跟列/行消除上面重疊)
            row_start_index = only1_position[i][0] - (only1_position[i][0] % SUDOKU_BOX_SIZE)
            col_start_index = only1_position[i][1] - (only1_position[i][1] % SUDOKU_BOX_SIZE)
            #print('1-', only1_position[i][0], only1_position[i][1], row_start_index, col_start_index)
            for tmp_row_index in range(SUDOKU_BOX_SIZE):
                for tmp_col_index in range(SUDOKU_BOX_SIZE):
                    options_list = matrix_options[row_start_index + tmp_row_index][col_start_index + tmp_col_index]
                    #print('2-', row_start_index + tmp_row_index, col_start_index + tmp_col_index, options_list, only1_num[i])
                    if only1_num[i] in options_list:
                        options_list.remove(only1_num[i])
                    if len(options_list) == 1 and (row_start_index+tmp_row_index, col_start_index+tmp_col_index) not in new_only1_position:
                        new_only1_position.append((row_start_index+tmp_row_index, col_start_index+tmp_col_index))
                        new_only1_num.append(options_list[0])
                        matrix_options[row_start_index+tmp_row_index][col_start_index+tmp_col_index] = []
        #print()
        #print('only1_position       ->', only1_position)
        #print('only1_num            ->', only1_num)
        #print('new_only1_position   ->', new_only1_position)
        #print('new_only1_num        ->', new_only1_num)
        #print(matrix_options)
        only1_position = new_only1_position
        only1_num = new_only1_num
        
        #import time
        #time.sleep(1)
    if is_completed(matrix):
        return True
    #######################################################
    # 若剩下還有空格都是不唯一解->從最少選擇的格子開始DFS
    else:
        #print('------------------------ going DFS ------------------------')
        DFS_matrix_stack = []
        DFS_matrix_options_stack = []
        import copy

        least_choice = 9
        dfs_row, dfs_col = -1, -1
        for row in range(len(matrix_options)):
            for col in range(len(matrix_options[0])):
                choices_num = len(matrix_options[row][col])
                if choices_num > 0 and choices_num < least_choice:
                    least_choice = len(matrix_options[row][col])
                    dfs_row, dfs_col = row, col
        # 對選擇最少的點做填值並對影響點去除可能性
        for choice in range(least_choice):
            new_matrix = copy.deepcopy(matrix)
            new_matrix_options = copy.deepcopy(matrix_options)
            new_matrix[dfs_row][dfs_col] = matrix_options[dfs_row][dfs_col][choice]
            new_matrix_options[dfs_row][dfs_col] = []
            #列消除可能
            for col in range(len(new_matrix_options[0])):
                options_list = new_matrix_options[dfs_row][col]
                if new_matrix[dfs_row][dfs_col] in options_list:
                    options_list.remove(new_matrix[dfs_row][dfs_col])
                    #print('! row remove', new_matrix[dfs_row][dfs_col], dfs_row, col, options_list)
            #行消除可能
            for row in range(len(new_matrix_options)):
                options_list = new_matrix_options[row][dfs_col]
                if new_matrix[dfs_row][dfs_col] in options_list:
                    options_list.remove(new_matrix[dfs_row][dfs_col])
                    #print('! col remove', new_matrix[dfs_row][dfs_col], row, dfs_col, options_list)
            #塊消除可能(注意跟列/行消除上面重疊)
            row_start_index = dfs_row - (dfs_row % SUDOKU_BOX_SIZE)
            col_start_index = dfs_col - (dfs_col % SUDOKU_BOX_SIZE)
            for tmp_row_index in range(SUDOKU_BOX_SIZE):
                for tmp_col_index in range(SUDOKU_BOX_SIZE):
                    options_list = new_matrix_options[row_start_index + tmp_row_index][col_start_index + tmp_col_index]
                    #print('2-', row_start_index + tmp_row_index, col_start_index + tmp_col_index, options_list, new_matrix[dfs_row][dfs_col])
                    if new_matrix[dfs_row][dfs_col] in options_list:
                        options_list.remove(new_matrix[dfs_row][dfs_col])
                        #print('! box remove', new_matrix[dfs_row][dfs_col], row_start_index+tmp_row_index, col_start_index+tmp_col_index, options_list)
            
            DFS_matrix_stack.append(new_matrix)
            DFS_matrix_options_stack.append(new_matrix_options)

########################################到上面截止為正確，找最少選擇的並列舉append進去(已包含所有行動樹)

        # DFS + 上面的唯一性解 (做到無解就重新pop)
        while DFS_matrix_options_stack != []:
            DFS_matrix = DFS_matrix_stack.pop()
            DFS_matrix_options = DFS_matrix_options_stack.pop()
            '''
            for DFS_matrix_line in DFS_matrix:
                print('init', DFS_matrix_line)
            for DFS_matrix_options_line in DFS_matrix_options:
                print('init', DFS_matrix_options_line)
            '''
            only1_position = [] #紀錄只有一種可能的0點位置
            only1_num = []      #紀錄只有一種可能的0點數字
            for row in range(len(DFS_matrix)):
                for col in range(len(DFS_matrix[0])):
                    point_options = []
                    if DFS_matrix[row][col] == UNASSIGNED:
                        for num in range(1, 10):
                            if is_safe(DFS_matrix, row, col, num):
                                point_options.append(num)
                        #若無解
                        if len(point_options) == 0:
                            return False
                        #若只有一種可能
                        elif len(point_options) == 1:
                            only1_position.append((row, col))
                            only1_num.append(point_options[0])
            #print('only1_position', only1_position)
            #print('only1_num', only1_num)
            
            # 針對每個確定值給入並刪去其同行同列跟同box的可能，解完跳出
            while True:
                if len(only1_position) == 0:
                    break
                new_only1_position = []
                new_only1_num = []
                #唯一值填入
                for i in range(len(only1_position)):
                    DFS_matrix[only1_position[i][0]][only1_position[i][1]] = only1_num[i]
                    DFS_matrix_options[only1_position[i][0]][only1_position[i][1]] = []
                    #列消除可能
                    for col in range(len(DFS_matrix_options[0])):
                        options_list = DFS_matrix_options[only1_position[i][0]][col]
                        if only1_num[i] in options_list:
                            options_list.remove(only1_num[i])
                            #print('!!! row remove', only1_num[i], only1_position[i][0], col, options_list)
                        if len(options_list) == 1:
                            new_only1_position.append((only1_position[i][0], col))
                            new_only1_num.append(options_list[0])
                            DFS_matrix_options[only1_position[i][0]][col] = []
                    #行消除可能
                    for row in range(len(DFS_matrix_options)):
                        options_list = DFS_matrix_options[row][only1_position[i][1]]
                        if only1_num[i] in options_list:
                            options_list.remove(only1_num[i])
                            #print('!!! col remove', only1_num[i], row, only1_position[i][1], options_list)
                        if len(options_list) == 1:
                            new_only1_position.append((row, only1_position[i][1]))
                            new_only1_num.append(options_list[0])
                            DFS_matrix_options[row][only1_position[i][1]] = []
                    #塊消除可能(注意跟列/行消除上面重疊)
                    row_start_index = only1_position[i][0] - (only1_position[i][0] % SUDOKU_BOX_SIZE)
                    col_start_index = only1_position[i][1] - (only1_position[i][1] % SUDOKU_BOX_SIZE)
                    #print('1-', only1_position[i][0], only1_position[i][1], row_start_index, col_start_index)
                    for tmp_row_index in range(SUDOKU_BOX_SIZE):
                        for tmp_col_index in range(SUDOKU_BOX_SIZE):
                            options_list = DFS_matrix_options[row_start_index + tmp_row_index][col_start_index + tmp_col_index]
                            #print('2-', row_start_index + tmp_row_index, col_start_index + tmp_col_index, options_list, only1_num[i])
                            if only1_num[i] in options_list:
                                options_list.remove(only1_num[i])
                                #print('!!! box remove', only1_num[i], row_start_index+tmp_row_index, col_start_index+tmp_col_index, options_list)
                            if len(options_list) == 1 and (row_start_index+tmp_row_index, col_start_index+tmp_col_index) not in new_only1_position:
                                new_only1_position.append((row_start_index+tmp_row_index, col_start_index+tmp_col_index))
                                new_only1_num.append(options_list[0])
                                DFS_matrix_options[row_start_index+tmp_row_index][col_start_index+tmp_col_index] = []
                #print()
                #print('only1_position       ->', only1_position)
                #print('only1_num            ->', only1_num)
                #print('new_only1_position   ->', new_only1_position)
                #print('new_only1_num        ->', new_only1_num)
                #print(DFS_matrix_options)
                #print(DFS_matrix)
                only1_position = new_only1_position
                only1_num = new_only1_num
                #import time
                #time.sleep(60)
                
            #print('remain len(DFS_matrix_stack)', len(DFS_matrix_stack))
            #print('remain len(DFS_matrix_options_stack)', len(DFS_matrix_options_stack))
            
            if is_completed(DFS_matrix):

                # 麻煩的全域call by ref.
                for i in range(len(SUDOKU_MATRIX)):
                    SUDOKU_MATRIX[i] = DFS_matrix[i]
                #print(id(SUDOKU_MATRIX), id(DFS_matrix))
                return True
            else:
                least_choice = 9
                dfs_row, dfs_col = -1, -1
                for row in range(len(DFS_matrix_options)):
                    for col in range(len(DFS_matrix_options[0])):
                        choices_num = len(DFS_matrix_options[row][col])
                        if choices_num > 0 and choices_num < least_choice:
                            least_choice = len(DFS_matrix_options[row][col])
                            dfs_row, dfs_col = row, col
                #print('least_choice, dfs_row, dfs_col', least_choice, dfs_row, dfs_col)
                for choice in range(least_choice):
                    new_matrix = copy.deepcopy(DFS_matrix)
                    new_matrix_options = copy.deepcopy(DFS_matrix_options)
                    new_matrix[dfs_row][dfs_col] = DFS_matrix_options[dfs_row][dfs_col][choice]
                    new_matrix_options[dfs_row][dfs_col] = []
                    #列消除可能
                    for col in range(len(new_matrix_options[0])):
                        options_list = new_matrix_options[dfs_row][col]
                        if new_matrix[dfs_row][dfs_col] in options_list:
                            options_list.remove(new_matrix[dfs_row][dfs_col])
                            #print('!! row remove', new_matrix[dfs_row][dfs_col], dfs_row, col, options_list)
                    #行消除可能
                    for row in range(len(new_matrix_options)):
                        options_list = new_matrix_options[row][dfs_col]
                        if new_matrix[dfs_row][dfs_col] in options_list:
                            options_list.remove(new_matrix[dfs_row][dfs_col])
                            #print('!! col remove', new_matrix[dfs_row][dfs_col], row, dfs_col, options_list)
                    #塊消除可能(注意跟列/行消除上面重疊)
                    row_start_index = dfs_row - (dfs_row % SUDOKU_BOX_SIZE)
                    col_start_index = dfs_col - (dfs_col % SUDOKU_BOX_SIZE)
                    for tmp_row_index in range(SUDOKU_BOX_SIZE):
                        for tmp_col_index in range(SUDOKU_BOX_SIZE):
                            options_list = new_matrix_options[row_start_index + tmp_row_index][col_start_index + tmp_col_index]
                            #print('2-', row_start_index + tmp_row_index, col_start_index + tmp_col_index, options_list, new_matrix[dfs_row][dfs_col])
                            if new_matrix[dfs_row][dfs_col] in options_list:
                                options_list.remove(new_matrix[dfs_row][dfs_col])
                                #print('!! box remove', new_matrix[dfs_row][dfs_col], row_start_index+tmp_row_index, col_start_index+tmp_col_index, options_list)
                    '''
                    for new_matrix_line in new_matrix:
                        print('in', new_matrix_line)
                    for new_matrix_options_line in new_matrix_options:
                        print('in', new_matrix_options_line)
                    '''
                    DFS_matrix_stack.append(new_matrix)
                    DFS_matrix_options_stack.append(new_matrix_options)    
            #print('len(DFS_matrix_options_stack)', len(DFS_matrix_options_stack))
            #print('-----DFS loop done!!!')
        return False
    # END_YOUR_CODE
    #return False


def sudoku_solver() -> list:
    global SUDOKU_MATRIX
    matrix = SUDOKU_MATRIX
    if sudoku_solve():
        return matrix
    return ["No Result"]


if __name__ == '__main__':
    SUDOKU_MATRIX = []
    SUDOKU_BOX_SIZE = 0
    MAX_NUMBER = 0

    parser = argparse.ArgumentParser()
    parser.add_argument('--sudoku_file_path', '-f', type=str, default='q1_sample_input.json')

    args = parser.parse_args()
    with open(args.sudoku_file_path, "r+") as fs:
        sudoku_info = json.load(fs)
        SUDOKU_MATRIX = sudoku_info['matrix']
        SUDOKU_BOX_SIZE = sudoku_info['box_size']
        MAX_NUMBER = SUDOKU_BOX_SIZE**2

    sudoku_result = sudoku_solver()

    print("Answer:")
    for result in sudoku_result:
        print(result)

    result_dict = {
        "result": sudoku_result
    }

    with open('sudoku_solver_result.json', 'w+') as fs:
        fs.write(json.dumps(result_dict, indent=4))
#python .\hw3_Constraint_Satisfaction_Problem\q1.py -f .\hw3_Constraint_Satisfaction_Problem\q1_sample_input.json
#python .\hw3_Constraint_Satisfaction_Problem\q1.py -f .\hw3_Constraint_Satisfaction_Problem\q1_sample_input2.json