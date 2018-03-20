# -*- coding:utf-8 -*-

class Solution4:
    # 面试题27：二叉树的镜像
    def Mirror(self, root):
        if root == None:
            return False
        if root.left != None or root.right != None:
            root.left,root.right = root.right,root.left
            if root.left != None:
                self.Mirror(root.left)
            if root.right != None:
                self.Mirror(root.right)
        return root

    # 面试题28：对称二叉树：请实现一个函数，用来判断一颗二叉树是不是对称的。
    # 注意，如果一个二叉树同此二叉树的镜像是同样的，定义其为对称的
    # 遍历，保证下一个节点的左等于右，右等于左
    def isSymmetrical(self, pRoot):
        return self.IsSelfSymmetrical(pRoot,pRoot)

    def IsSelfSymmetrical(self,pRoot1,pRoot2):
        if pRoot1 is None and pRoot2 is None:
            return True
        if pRoot1 is None or pRoot2 is None:
            return False
        if pRoot1.val != pRoot2.val:
            return False
        return self.IsSelfSymmetrical(pRoot1.left,pRoot2.right) and self.IsSelfSymmetrical(pRoot1.right,pRoot2.left)

    # 面试题29：顺时针打印矩阵：输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字
    def printMatrix(self, matrix):
        res=[]
        rows = len(matrix)
        columns = len(matrix[0])
        if matrix is None or rows<=0 or columns<=0:
            return
        start = 0
        while columns>start*2 and rows>start*2:
            self.PrintMatrixCircle(matrix,columns,rows,start)
            start+=1
    def PrintMatrixCircle(self,matrix,columns,rows,start):
        endX = columns-1-start
        endY = rows -1 - start
        for i in range(start,endX+1):   # 从左到右
            number = matrix[start][i]
            print number,
        if start<endY:     # 从上到下
            for i in range(start+1,endY+1):
                number = matrix[i][endX]
                print number,
        if start<endX and start<endY:  # 从右向左
            for i in range(endX-1,start-1,-1):
                number=matrix[endY][i]
                print number,
        if start<endX and start<endY-1:  # 从下到上
            for i in range(endY-1,start,-1):
                number = matrix[i][start]
                print number,

    # 面试题30:包含min函数的栈：定义栈的数据结构，请在该类型中实现一个能够得到栈最小元素的min函数。
    def __init__(self):
        self.stack=[]
        self.minStack=[]
    def push(self, node):
        # write code here
        self.stack.append(node)
        if self.minStack==[] or node<self.min():
            self.minStack.append(node)
        else:
            temp = self.min()
            self.minStack.append(temp)
    def pop(self):
        if self.stack == [] or self.minStack == []:
            return None
        self.minStack.pop()
        self.stack.pop()
    def top(self):
        return self.stack[-1]
    def min(self):
        return self.minStack[-1]

    # 面试题31：栈的压入、弹出序列
    # 输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否为该栈的弹出顺序。
    # 假设压入栈的所有数字均不相等。例如序列1,2,3,4,5是某栈的压入顺序，
    # 序列4，5,3,2,1是该压栈序列对应的一个弹出序列，但4,3,5,1,2就不可能是该压栈序列的弹出序列。
    # （注意：这两个序列的长度是相等的）
    def IsPopOrder(self, pushV, popV):
        if pushV is None and popV is None:
            return False
        stack = []
        for i in range(0,len(pushV)):
            stack.append(pushV[i])
            if stack[-1] != popV[0]:
                continue
            else:
                stack.pop()
                popV.pop(0)
        while len(stack)>0 and stack[-1]==popV[0]:
            stack.pop()
            popV.pop(0)
        if len(stack)==0:
            return True
        else:
            return False

    # 面试题32：从上到下打印二叉树：从上往下打印出二叉树的每个节点，同层节点从左至右打印。
    #def PrintFromTopToBottom(self, root):











if __name__ == "__main__":
    k = Solution4()
    s = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
    print k.printMatrix(s)
