# -*- coding:utf-8 -*-
import numpy as np

class Solution2:
    # 面试题3：数组中重复的数据

    # 题目一：在一个长度为n的数组里的所有数字都在0到n-1的范围内。
    # 数组中某些数字是重复的，但不知道有几个数字是重复的。也不知道每个数字重复几次。
    # 请找出数组中任意一个重复的数字。 例如，如果输入长度为7的数组{2,3,1,0,2,5,3}，那么对应的输出是第一个重复的数字2。

    # 法一：重排数组，根据下标和数值
    def duplicate1(self,numbers, duplication):
        if numbers is None:
            return False
        for num in numbers:
            if num<0 or num>len(numbers)-1:
                return False
        for i in range (len(numbers)):
            while(i!=numbers[i]):
                if numbers[i] == numbers[numbers[i]]:
                    duplication[0]=numbers[i]
                    print duplication[0]
                    return True
                temp = numbers[i]
                numbers[i] = numbers[temp]
                numbers[temp]=temp
        return False
    # 法二：排序后遍历
    def duplicate2(self,numbers,duplication):
        if numbers is None:
            return False
        for num in numbers:
            if num<0 or num>len(numbers)-1:
                return False
        num = sorted(numbers)
        for i in range(len(num)-1):
            if num[i] == num[i+1]:
                duplication[0] = num[i]
                print duplication[0]
                return True
        return False

    # 题目二 不修改数组找出重复的数字
    # 在一个长度为n+1的数组里的所有数字都在1-n的范围内，所以数组中至少有一个数字是重复的，请找出数组中任意一个重复的数字
    # 但不能修改输入的数组，例如，，如果输入长度为8的数组{2，3，5，4，3，2，6，7}，那么对应的输出是重复的数字2或者3

    # 法一：增加一个辅助数组，元素m存放到辅助数据的第m-1处，判断哪个位置有重复。空间复杂度o(n)
    def getDuplication1(self,numbers,duplication):
        if numbers is None:
            return False
        for num in numbers:
            if num<1 or num>len(numbers):
                return False
        num = [0]*8
        for i in range(len(numbers)):
            if num[numbers[i]-1]==numbers[i]:
                print numbers[i]
                return True
            else:
                num[numbers[i]-1]=numbers[i]
        return False

    # 法二：类似二分法，例如有8个元素，1-7，那么选取中间值4，将原数组分为两部分，1-4，5-7.
    # 通过计算小于等于4的元素个数，大于等于5的个数，来判断重复数字出现的位置。
    # 在没有重复的情况下，1-4的个数应该是4，5-7的个数也为3，多出来的个数则为出现重复数字的区域。
    # 缺点，只能找个其中的某一个或者两个重复的元素，不能找到所有的重复元素。例如[2,3,5,4,3,2,6,7]中的2就无法找到。

    def getDuplication2(self,numbers,duplication):
        if numbers is None:
            return False
        for num in numbers:
            if num<1 or num>len(numbers):
                return False
        start=1
        end=len(numbers)-1
        while(start<=end):
            mid = (start + end) / 2
            # print 'start='+str(start)
            # print 'end=' + str(end)
            countsum = k.countRange(numbers,start,mid)
            if(start==end):
                if countsum>1:
                    return start
                else:
                    break
            if(countsum>mid-start+1):
                end = mid
            else:
                start = mid+1

        return False

    def countRange(self,numbers,start,end):
        if numbers is None:
            return False
        countsum=0
        for i in range(len(numbers)):
            if numbers[i]>=start and numbers[i]<=end:
                countsum=countsum+1
        return countsum


    # 法三： 查找第i个元素后面，是否有元素i的存在
    def getDuplication3(self, numbers, duplication):
        if numbers is None:
            return False
        for num in numbers:
            if num<1 or num>len(numbers):
                return False
        for i in range(len(numbers)):
            count = numbers.count(numbers[i])
            if count>0:
                return numbers[i]


    # 面试题4:二维数组中的查找
    # 在一个二维数组中，每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。
    # 请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。

    # 方法：从右上角切入，如果比右上角的数字小，那么可排除整列，比右上角的数字大，那么可排除整行。当然也可以从左下角开始检测比较。
    def Find(self, target, array):
        if array is None:
            return False
        row = 0
        column = len(array[0])-1
        while(row<=len(array)-1 and column>=0):
            print array[row][column]
            if array[row][column]==target:
                return True
            elif array[row][column]>target:
                column-=1
                #print 'row='+str(row)+' '+'column='+str(column)
            else:
                row+=1
                #print 'row='+str(row)+' '+'column='+str(column)
        return False

    # 面试题5：替换空格
    # 请实现一个函数，将一个字符串中的空格替换成“%20”。例如，当字符串为We Are Happy.则经过替换之后的字符串为We%20Are%20Happy。
    # 法一：直接使用replace替换
    def replaceSpace1(self, s):
        res = s.replace(' ','%20')
        return res

    # 法二：使用由后到前的方法。先遍历字符串来计算占用的内存大小，再从后向前逐一遍历字符串，设置两个指针。python不友好，固不赘述
    # 注意java中一个字符串的大小，末尾有一个‘\0’字符，占一个字符。

    # 面试题6：从尾到头打印链表。输入一个链表，从尾到头打印链表每个节点的值。
    # 法一：遍历链表，将其结果一直插入到一个list中的首位。
    def printListFromTailToHead1(self, listNode):
        # write code here
        l = []
        head = listNode
        while head:
            l.insert(0, head.val)
            head = head.next
        return l


    # 面试题7：重建二叉树。输入某二叉树的前序遍历和中序遍历的结果，请重建出该二叉树。
    # 假设输入的前序遍历和中序遍历的结果中都不含重复的数字。
    # 例如输入前序遍历序列{1,2,4,7,3,5,6,8}和中序遍历序列{4,7,2,1,5,3,8,6}，则重建二叉树并返回。
    def reConstructBinaryTree(self, pre, tin):
        if len(pre)==0:
            return None
        if len(pre)==1:
            return TreeNode(pre[0])
        else:
            root = TreeNode(pre[0])
            index = tin.index(pre[0])
            left = tin[:index]
            right = tin[index+1:]
            root.left = self.reConstructBinaryTree(pre[1:len(left)+1],left)
            root.right = self.reConstructBinaryTree(pre[len(left)+1:],right)
            return root


    # 面试题8:二叉树的下一个节点。给定一棵二叉树和其中的一个节点，如何找出中序遍历序列的下一个节点？
    # 树中的节点除了有两个分别指向左、右子节点的指针，还有一个指向父节点的指针。
    # 判断pNode当前位置的状态，如果有右子节点，那么pNode的下一个节点就是其右子节点的左子节点。
    # 如果没有右子节点，而且它是其父节点的左子节点，那么下一个节点就是其父节点p。
    # 最复杂的情况：如果当前这个状态既没有右子节点，而且又是其父节点的右子节点，那么就继续向上寻找父节点，知道找到其不是右子节点为止。
    def GetNext(self, pNode):
        if pNode.right:
            cur = pNode.right
            while cur.left:
                cur = cur.left
            return cur
        else:
            p = pNode.next
            if not p:
                return None
            else:
                if p.left == pNode:
                    return p
                else:
                    cur = pNode
                    while p and p.right == cur:
                        cur = p
                        p = cur.next
                    return p

    # 面试题9：用两个栈实现一个队列。完成队列的Push和Pop操作。完成在队列尾部插入节点和在队列头部删除节点的功能。
    # 队列中的元素为int类型。
    # 1，整体思路是元素先依次进入栈1，再从栈1依次弹出到栈2，然后弹出栈2顶部的元素，整个过程就是一个队列的先进先出
    # 2，但是在交换元素的时候需要判断两个栈的元素情况：
    # “进队列时”，判断队列2中是否还有元素，若有，说明栈2中的元素不为空，此时就先将栈2的元素倒回到栈1 中，保持在“进队列状态”。
    # “出队列时”，将栈1的元素全部弹到栈2中，保持在“出队列状态”
    # 所以要做的判断是，进时，栈2是否为空，不为空，则栈2元素倒回到栈1。出时，将栈1元素全部弹到栈2中，直到栈1为空。
    class Solution:
        def __init__(self):
            self.stackA = []
            self.stackB = []
        def push(self, node):
            # write code here
            self.stackA.append(node)
        def pop(self):
            # return xx
            if self.stackB:
                return self.stackB.pop()
            elif not self.stackA:
                return None
            else: # B为空，A不为空
                while self.stackA:
                    self.stackB.append(self.stackA.pop())
                return self.stackB.pop()

    # 面试题10：斐波那契数列。
    # 题目一：求斐波那契数列的第N项。写一个函数，输入n,求斐波那契数列的第n项。
    #f(n)=0(n=0) or 1 (n=1) or f(n-1)+f(n-2) n>1
    # 法一：直接按照列出的公式进行递归，但是有严重的效率问题（重复计算）。例如，求f(9)时，需要先求f（8）+f（7），在求f（8）时，又要求f(7)。
    def Fibonacci1(self, n):
        if n==0:
            return 0
        elif n==1:
            return 1
        else:
            return self.Fibonacci(n-1)+self.Fibonacci(n-2)
    # 法二：为了避免计算重复的内容，选择自下而上的计算方式。先根据f(0),f(1)计算得到f(2),再由f(1)f(2)计算f(3),以此类推。
    def Fibonacci2(self, n):
        if n==0:
            return 0
        elif n==1:
            return 1
        else:
            One = 0
            Two = 1
            for i in range(n-1):
                sum = One + Two
                One = Two
                Two = sum
            return sum
    # 法三：矩阵相乘的公式。不赘述。

    # 题目二：青蛙跳台阶问题。一只青蛙一次可以跳一级台阶，也可以跳两级台阶。求该青蛙跳上一个n级的台阶总共有多少种跳法。
    def jumpFloor(self, number):
        if number==0 or number==1:
            return 1
        else:
            One = 1
            Two = 1
            for i in range(number-1):
                sum = One + Two
                One = Two
                Two = sum
            return sum
    # 题目三：变态跳台阶。一只青蛙一次可以跳上1级台阶，也可以跳上2级……它也可以跳上n级。求该青蛙跳上一个n级的台阶总共有多少种跳法。
    # 多列些一些，发现f(n)=f(n-1)+f(n-2)+...+f(2)+f(1)+f(0)=2*f(n-1)=2^(n-1)
    def jumpFloorII(self, number):
        if number==0 or number==1:
            return 1
        else:
            return 2*(self.jumpFloorII(number-1))
    # 题目四：矩形覆盖
    # 我们可以用2*1的小矩形横着或者竖着去覆盖更大的矩形。请问用n个2*1的小矩形无重叠地覆盖一个2*n的大矩形，总共有多少种方法？
    # 相同的问题
    def rectCover(self, number):
        if number==0:
            return 0
        elif number==1:
            return 1
        elif number==2:
            return 2
        else:
            One = 1
            Two = 2
            for i in range(number-2):
                sum = One + Two
                One = Two
                Two = sum
            return sum

    # 快速排序。在数组中选择一个数字，接下来把数组中的数字分为两部分，比选择的数字小的数字移到数组的左边，比选择的数字大的移到右边。
    def QuickSort(self,num,start,end):
        if start>end:
            return False
        elif start==end:
            return num
        else:
            key = num[start]
            flag = start
            for i in range(start+1,end+1):
                if key>num[i]:
                    tmp = num[i]
                    del num[i]
                    num.insert(start,tmp)
                    flag=+1
        self.QuickSort(num,start,flag-1)
        self.QuickSort(num,flag+1,end)
        return num

    # 面试题11：旋转数组的最小数字
    # 把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。输入一个递增排序的数组的一个旋转，输出旋转数组的最小元素。
    # 例如数组{3,4,5,1,2}为{1,2,3,4,5}的一个旋转，该数组的最小值为1。注意1，1，1，0，1这种特殊情况。
    def minNumberInRotateArray(self, rotateArray):
        if rotateArray is None:
            return False
        elif len(rotateArray)==1:
            return rotateArray[0]
        else:
            low=0
            high=len(rotateArray)-1
            if rotateArray[low]<rotateArray[high]:
                return rotateArray[low]
            else:
                while(low+1<high):
                    mid = (low+high)/2
                    if rotateArray[mid]==rotateArray[low] and rotateArray[mid]==rotateArray[high]:
                        for i in range(0, len(rotateArray)):
                            if rotateArray[i] < rotateArray[low]:
                                high = i
                    elif rotateArray[mid]>=rotateArray[low]:
                        low = mid
                    elif rotateArray[mid]<=rotateArray[high]:
                        high = mid
                return rotateArray[high]

    # 面试题12：矩阵中的路径
    # 请设计一个函数，用来判断在一个矩阵中是否存在一条包含某字符串所有字符的路径。
    # 路径可以从矩阵中的任意一个格子开始，每一步可以在矩阵中向左，向右，向上，向下移动一个格子。
    # 如果一条路径经过了矩阵中的某一个格子，则该路径不能再进入该格子。
    # 例如 a b c e s f c s a d e e 矩阵中包含一条字符串"bcced"的路径，但是矩阵中不包含"abcb"路径，
    # 因为字符串的第一个字符b占据了矩阵中的第一行第二个格子之后，路径不能再次进入该格子。
    # 着这种问题就只能用递归来做了，考虑清楚什么时候递归。
    def hasPath(self, matrix, rows, cols, path):
        if matrix is None or rows < 1 or cols < 1 or path is None:
            return False
        for i in range(0,rows):
            for j in range(0,cols):
                if matrix[i*cols+j] == path[0]:
                    if self.find(matrix,rows,cols,path[1:],i,j):
                        return True
        return False

    def find(self, matrix, rows, cols, path, i, j):
        if not path:
            return True
        matrix[i*cols+j] = '0'
        if j+1<cols and matrix[i*cols+j+1]==path[0]:
            self.find(matrix,rows,cols,path[1:],i,j+1)
        elif j-1>=0 and matrix[i*cols+j-1]==path[0]:
            self.find(matrix,rows,cols,path[1:],i,j-1)
        elif i+1<rows and matrix[(i+1)*cols+j]==path[0]:
            self.find(matrix,rows,cols,path[1:],i+1,j)
        elif i-1>=0 and matrix[(i-1)*cols+j]==path[0]:
            self.find(matrix,rows,cols,path[1:],i-1,j)
        else:
            return False

    # 面试题13：机器人的运动范围
    # 地上有一个m行和n列的方格。一个机器人从坐标0, 0的格子开始移动，每一次只能向左，右，上，下四个方向移动一格，
    # 但是不能进入行坐标和列坐标的数位之和大于k的格子。
    # 例如，当k为18时，机器人能够进入方格（35, 37），因为3 + 5 + 3 + 7 = 18。
    # 但是，它不能进入方格（35, 38），因为3 + 5 + 3 + 8 = 19。请问该机器人能够达到多少个格子？
    # 和矩阵的检索相似
    def movingCount(self, threshold, rows, cols):
        if rows < 1 or cols < 1 or threshold == 0:
            return False
        matrix = [[1] * cols for i in range(rows)]
        count = self.movingCountCore(threshold,rows,cols,matrix,0,0)
        del matrix
        return count

    def movingCountCore(self, threshold, rows, cols, matrix,i,j):
        count = 0
        if self.check(threshold,rows,cols, matrix, i ,j):
            matrix[i][j] = 0
            count = 1 + self.movingCountCore( threshold, rows, cols, matrix,i-1,j) + self.movingCountCore( threshold, rows, cols, matrix,i+1,j) + self.movingCountCore( threshold, rows, cols, matrix,i,j+1) + self.movingCountCore( threshold, rows, cols, matrix,i,j-1)
        return count

    def check(self,threshold,rows,cols,matrix, i,j):
        if i >=0 and i < rows and j>=0 and j< cols and matrix[i][j]==1 and self.getThreshold(i,j)<=threshold:
            return True
        return False

    def getThreshold(self,i,j):
        sum=0
        while(i>0):
            sum = sum + i%10
            i = i/10
        while(j>0):
            sum = sum +j%10
            j = j/10
        return sum

    # 面试题14： 剪绳子。给你一根长度为n的绳子，请把绳子剪成m段。（m、n都是整数，n>1并且m>1），求解长度乘积的最大值。
    # 法一：动态规划。列些方程等式。f(n)=max{f(i)*f(n-i)]，为了避免重复计算，从上而下的分析，从下而上的求解。
    def maxProductAfterCutting1(self, length):
        if length < 2:
            return False
        elif length ==2:
            return 1
        elif length ==3:
            return 2
        else:
            product = [0]*(length+1)
            product[0]=0
            product[1]=1
            product[2]=2
            product[3]=3
            for i in range(4,length):
                max = 0
                for j in range(1,i/2+1):
                    tmp = product[j]*product[i-j]
                    if max<tmp:
                        max=tmp
                    product[i]=max
            max = product[length]
            del product
            return max

    # 法二：贪婪算法。总结规律
    def maxProductAfterCutting2(self, length):
        if length < 2:
            return False
        elif length ==2:
            return 1
        elif length ==3:
            return 2
        else:
            timesOf3 = length/3
            if length-3*timesOf3==1:
                timesOf3=timesOf3-1
            timesOf2 = (length-timesOf3*3)/2
        return (3**timesOf3)*(2**timesOf2)

    # 面试题15：二进制中1的个数。输入一个整数，输出该数二进制表示中1的个数。其中负数用补码表示。将判断符左移，为什么不能右移？因为负数！
    # 法一：循环的次数等于整数二进制的维数。32
    def NumberOf1(self, n):
        flag = 1
        count = 0
        for i in range(0,32):
            if n&flag:
                count+=1
            flag = flag << 1
        return count
    # 法二：将整数减去1，再和原整数做与运算，会把该整数最右边的1变成0。那么就是由多少个1，就进行多少次的循环
    def NumberOf1(self, n):
        count = 0
        if n < 0:
            n = n & 0xffffffff    # 求出一个负数的补码，也就是一个负数应该的二进制数表示方法。
        while n:
            count += 1
            n = (n - 1) & n
        return count








class TreeLinkNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
        self.next = None
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
# Definition for singly-linked list.
class ListNode(object):
    def __init__(self):
        self.val = None
        self.next = None
class ListNode_handle:
    def __init__(self):
        self.cur_node = None

    def add(self, data):
        # add a new node pointed to previous node
        node = ListNode()
        node.val = data
        node.next = self.cur_node
        self.cur_node = node
        return node

    def print_ListNode(self, node):
        while node:
            print '\nnode: ', node, ' value: ', node.val, ' next: ', node.next
            node = node.next

    def _reverse(self, nodelist):
        list = []
        while nodelist:
            list.append(nodelist.val)
            nodelist = nodelist.next
        result = ListNode()
        result_handle = ListNode_handle()
        for i in list:
            result = result_handle.add(i)
        return result

if __name__ == "__main__":
    # nums=[2,3,5,4,3,2,6,7]
    # duplication = [2]
    k = Solution2()
    # print k.getDuplication3(nums,duplication)

    # array = [[1,2,8,9],[2,4,9,12],[4,7,10,13],[6,8,11,15]]
    # k=Solution2()
    # print k.Find(7,array)

    # str = 'We Are Happy.'
    # print k.replaceSpace1(str)

    # ListNode_1 = ListNode_handle()
    # l1 = ListNode()
    # l1_list = [1, 8, 3]
    # for i in l1_list:
    #     l1 = ListNode_1.add(i)
    # print ListNode_1.print_ListNode(l1)
    # print k.printListFromTailToHead1(l1)

    # pre = [1,2,4,7,3,5,6,8]
    # tin = [4,7,2,1,5,3,8,6]
    # print k.reConstructBinaryTree(pre,tin)

    # n=4
    #print k.Fibonacci2(n)
    # print k.jumpFloorII(n)

    # num = [1,1,1,0,1]
    # print k.minNumberInRotateArray(num)
    # matrix = [[1] * 5 for i in range(5)]
    # print type(matrix)
    # matrix[0][1]=0
    # print matrix

    # length=5
    # print k.maxProductAfterCutting2(length)

    n=-5 # 原码
    nn = n & 0xffffffff   # 补码
    print bin(nn)






