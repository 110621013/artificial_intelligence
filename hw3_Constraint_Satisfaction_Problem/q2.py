import json
import argparse
from collections import defaultdict
from copy import deepcopy


def course_schedule_solver() -> list:
    global COURSE_ORDER
    order = COURSE_ORDER
    # BEGIN_YOUR_CODE
    # 如果有解, 需要回傳順序的結果
    '''
    圖論, 無解相當於有課程a-b-c-a的迴圈->需確定無迴圈
    採用迴圈DFS, 給定探訪情況的list(-1為現在正在找的迴圈, 0為還沒探訪過, 1為之前已經探訪過)
    迴圈DFS目標是找如果有人遞迴找下去找到-1表示有迴圈, 回傳[]
    沒有就針對所有沒有探訪過的做上述檢驗直至結束
    通過leetcode
    '''
    global numCourses, prerequisites
    
    # recur. DFS
    def dfs_circle(x):
        visited[x] = -1
        for y in graph[x]:
            if visited[y] < 0 or (not visited[y] and dfs_circle(y)):
                return True
        visited[x] = 1
        orders.append(x)
        return False

    # 建圖建探訪
    graph = [set() for _ in range(numCourses)]
    for d, s in prerequisites:
        graph[s].add(d)
    visited = [0]*numCourses
    orders = []
    
    for x in range(numCourses):
        if not visited[x] and dfs_circle(x):
            return []
    
    for i in range(numCourses):
        order[i] = orders[numCourses-i-1]
    return order


if __name__ == '__main__':
    COURSE_ORDER = []

    parser = argparse.ArgumentParser()
    parser.add_argument('--course_file_path', '-f', type=str, default='q2_sample_input.json')

    args = parser.parse_args()
    with open(args.course_file_path, "r+") as fs:
        course_schedule_info = json.load(fs)
        numCourses = course_schedule_info['numCourses']
        prerequisites = course_schedule_info['prerequisites']

    COURSE_ORDER = [-1 for x in range(numCourses)]
    course_order = course_schedule_solver()

    print("COURSE ORDER:")
    print(course_order)

    result_dict = {
        "result": course_order
    }

    with open('course_schedule_result.json', 'w+') as fs:
        fs.write(json.dumps(result_dict, indent=4))
#python .\hw3_Constraint_Satisfaction_Problem\q2.py -f .\hw3_Constraint_Satisfaction_Problem\q2_sample_input.json