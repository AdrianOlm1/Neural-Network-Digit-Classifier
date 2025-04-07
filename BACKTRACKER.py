import time

def isSafe(mat, row, col, num):
    for i in range(9):
        if num == mat[i][col]:
            return False
        
    for i in range(9):
        if num == mat[row][i]:
            return False
        
    startRow = row - (row % 3)
    startCol = col - (col % 3)
        
    for i in range(3):
        for j in range(3):
            if num == mat[startRow + i][startCol + j]:
                return False

    return True

def solveRec(mat, row, col):
    if row==8 and col==9:
        return True
    
    if col == 9:
        row += 1
        col = 0
    
    if mat[row][col] != 0:
        return solveRec(mat, row, col+1)
    
    for i in range(10):
        if isSafe(mat, row, col, i):
            mat[row][col] = i
            if solveRec(mat, row, col+1):
                return True
            mat[row][col] = 0

    return False

def solveSudoku(mat):
    solveRec(mat, 0, 0)

if __name__ == "__main__":
    mat = [[0,0,5,1,2,3,4,7,0],
           [0,0,4,0,0,0,0,6,0],
           [2,0,0,7,0,0,5,0,8],
           [0,0,3,0,6,5,8,9,0],
           [8,0,0,0,9,0,7,0,0],
           [0,0,6,0,0,0,3,0,0],
           [0,0,0,0,0,0,0,0,3],
           [0,4,0,0,5,1,0,0,7],
           [1,0,0,6,0,0,0,0,4]]

    start = time.time()
    solveSudoku(mat)
    end = time.time()
    print ("Time taken :",end-start)
    for row in mat:
        print(" ".join(map(str, row)))