# -*- coding:utf-8 -*-
import heapq
import numpy as np
class Solution6:
    # 面试题53：在排序数组中查找数字。
    # 题目一：数字在排序数组中出现的次数。统计一个数字在排序数组中出现的次数。
    # 例如，输入排序数组[1,2,3,3,3,3,4,5]和数字3，由于3在这个数组中出现4次，因此输出4
    # 法一：简单的方法，通过二分法找到这个数，然后左右遍历，直到不为这个数时，就停止计数。
    def GetNumberOfK(self, data, k):
        if data is None or len(data)<=0:
            return 0
        start = 0
        end = len(data)-1
        mid = (start + end)/2
        while(data[mid]!=k and start<end):
            if data[mid]<k:
                start = mid + 1
            else:
                end = mid - 1
            mid = (start + end)/2
        count = 0
        left = mid
        right = mid
        while(left>=0 and data[left]==k):
            count+=1
            left-=1
        while(right <=len(data)-2 and data[right+1]==k):
            count+=1
            right+=1
        return count

    # 法二：直接通过二分查找的方式找到首位K值的位置。递归
    def GetNumberOfK2(self, data, k):
        if data is None or len(data)<=0:
            return 0
        number = 0
        first = self.GetFirst(data,k,0,len(data)-1)
        last = self.GetLast(data,k,0,len(data)-1)
        if first>-1 and last >-1:
            number = last-first+1
        return number

    def GetFirst(self,data,k,start,end):
        if start>end:
            return -1
        middleIndex = (start+end)/2
        middleData = data[middleIndex]
        if middleData == k:
            if (middleIndex>=1 and data[middleIndex-1]!=k) or middleIndex == 0:
                return middleIndex
            else:
                end = middleIndex - 1
        elif middleData > k :
            end = middleIndex - 1
        else:
            start = middleIndex +1
        return self.GetFirst(data,k,start,end)

    def GetLast(self,data,k,start,end):
        if start>end:
            return -1
        middleIndex = (start + end) / 2
        middleData = data[middleIndex]
        if middleData == k:
            if (middleIndex <=len(data)-2 and data[middleIndex + 1] != k) or middleIndex == len(data)-1:
                return middleIndex
            else:
                start = middleIndex + 1
        elif middleData > k:
            end = middleIndex - 1
        else:
            start = middleIndex + 1
        return self.GetLast(data, k, start, end)

    # 题目二：0-n-1中缺失的数字。一个长度为n-1的递增排序数组中的所有数字都是唯一的，并且每个数字都在范围0-n-1之内
    # 在范围0-n-1内的n个数字中有且只有一个数字不在该数组中，请找出这个数字。
    # 法一通过异或，法二根据求和，但复杂度都是o(n)，没有利用其有序的特性。
    def GetMissingNumber1(self,numbers):
        if numbers is None or len(numbers)==0:
            return False
        if len(numbers)==1:
            return 0
        res_1 = 0
        for i in range(1,len(numbers)):
            res_1 = res_1^numbers[i]
        for j in range(0,len(numbers)+1):
            res_1 = res_1^j
        return res_1
    def GetMissingNumber2(self,numbers):
        if numbers is None or len(numbers)==0:
            return False
        sum = 0
        for j in range(0,len(numbers)+1):
            sum = sum + j
        for i in range(0,len(numbers)):
            sum = sum - numbers[i]
        return sum
    # 法三：寻找到第一个数值与下标数值不同的元素，二分查找。
    def GetMissingNumber3(self,numbers):
        if numbers is None or len(numbers)==0:
            return False
        start = 0
        end = len(numbers)-1
        while(start<=end):
            mid = (start + end) /2
            if numbers[mid] ==mid:
                start = mid + 1
            else:
                if numbers[mid-1] == mid-1 or mid ==0:
                    return mid
                else:
                    end = mid - 1
        return -1

    # 题目三：数组中数值和下标相等的元素。假设一个单调递增的数组里的每个元素都是整数并且是唯一的，请编程实现一个函数。
    # 找出数组中任意一个数值等于其下标的元素。例如，在数组-3，-1，1，3，5中，数字3和它的小标相等。
    # 法一，遍历，复杂度高，没有考虑有序特点
    def GetNumberSameAsIndex1(self,numbers):
        if numbers is None or len(numbers)==0:
            return False
        for i in range(len(numbers)):
            if numbers[i]^i==0:
                return i
        return False
    # 法二：二分查找
    def GetNumberSameAsIndex2(self,numbers):
        if numbers is None or len(numbers)==0:
            return False
        start = 0
        end = len(numbers)-1
        while(start<=end):
            mid = (start+end)/2
            if numbers[mid]==mid:
                return mid
            elif numbers[mid]<mid:
                start = mid +1
            else:
                end = mid - 1
        return -1  # 不存在

    # 面试题54：二叉搜索树的第K大节点。给定一颗二叉搜索树，请找出其中第K大的节点。
    #  将其进行中序遍历，则得到的便是一个增序列，可遍历得到第K大的数值
    def __init__(self):
        self.treeNode = []
    def TreeDepth(self, pRoot,k):
        if k == 0 or pRoot == None:
            return
        self.inOrder(pRoot)
        if len(self.treeNode) < k:
            return None
        return self.treeNode[k - 1]

    # def inOrder(self, pRoot):
    #     if len(self.treeNode) <= 0:
    #         return None
    #     if pRoot.left:
    #         self.inOrder(pRoot.left)
    #     self.treeNode.append(pRoot)
    #     if pRoot.right:
    #         self.inOrder(pRoot.right)

    def inOrder(self,tree):  # 中序遍历
        if tree == None:
            return
        self.inOrder(tree.left)
        self.treeNode.append(tree)   # 或者写：self.treeNode.append(TreeNode(tree).val)
        self.inOrder(tree.right)

    # 面试题55：二叉树的深度。
    # 题目一：二叉树的深度。输入一颗二叉树的根节点，求该树的深度，从根节点到叶节点依次经过的节点（含根、叶节点）形成树的一条路径，最长路径的长度为数的深度。
    # 一棵树只有一个节点，那么深度为1，如果只有左子树，那么深度就是左子树的深度+1，如果只有右子树，那么深度就是右子树的深度+1，如果左右均存在，那么就取最大值+1
    # 递归
    def TreeDepth(self, pRoot):
        if pRoot is None:
            return 0
        left = self.TreeDepth(pRoot.left)
        right = self.TreeDepth(pRoot.right)
        if left>right:
            return left+1
        else:
            return right+1
    # 题目二：平衡二叉树。输入一颗二叉树的根节点，判断该树是不是平衡二叉树，如果某二叉树中任意节点的左右子树的深度相差不超过1，那么它就是一颗平衡二叉树。
    def IsBalanced_Solution(self, pRoot):
        if pRoot is None:
            return True
        left = self.TreeDepth(pRoot.left)
        right = self.TreeDepth(pRoot.right)
        if abs(left-right)>1:
            return False
        return self.IsBalanced_Solution(pRoot.left) and self.IsBalanced_Solution(pRoot.right)
    # 面试题56：数值中数字出现的次数。
    # 题目一：数组中只出现一次的两个数字。一个整型数组里除两个数字之外，其他数字都出现了两次。找出这两个只出现一次的数字，要求时间复杂度是o(n),空间o(1)
    def FindNumsAppearOnce(self, array):
        if array is None or len(array)<=0:
            return
        res_OR = 0
        for i in range(len(array)):
            res_OR = res_OR^array[i]
        index = self.FindFirstBitIs1(res_OR)
        num_1 = 0
        num_2 = 0
        for i in range(len(array)):
            if self.IsBit1(array[i],index):
                num_1 = num_1^ array[i]
            else:
                num_2 = num_2 ^array[i]
        return num_1,num_2
    def FindFirstBitIs1(self,number):
        index = 0
        while number & 1==0 and index<=32:
            index = index +1
            number = number>>1
        return index
    def IsBit1(self,number,index):
        number = number>>index
        return number&1
    # 题目二：数组中唯一只出现一次的数字。在一个数组中除一个数字只出现一次之外，其他数字都出现了三次，请找出那个只出现一次的数字。
    def FindNumsAppearOnce1(self,array):
        if array is None or len(array)<=0:
            return
        bitSum = [0]*32
        for i in range(len(array)):
            bitMask = 1
            for j in range(31,-1,-1):
                bit = array[i]&bitMask
                if bit!=0:
                    bitSum[j] = bitSum[j] +1
                bitMask = bitMask<<1
        res = 0
        print bitSum
        for i in range(32):
            res = res + bitSum[i]%3
            res = res<<1
        return res >>1
    # 面试题57：和为S的数字。
    # 题目一：和为S的两个数字。输入一个递增排序的数组和一个数字s，在数组中查找两个数，使得他们的和正好是s，如果有多对数字的和等于s，则输出任意一对即可。
    # 输出两个数的乘积最小的。小的先输出，大的后输出。两边的乘积小，中间的乘积大
    def FindNumbersWithSum(self, array, tsum):
        if array is None or len(array)<=0:
            return []
        start = 0
        end = len(array)-1
        while start<end:
            sum = array[start] + array[end]
            if sum == tsum:
                return array[start],array[end]
            elif sum>tsum:
                end = end -1
            else:
                start = start +1
        return []
    # 题目二：和为s的连续正数序列
    # 输入一个整数s，打印出所有和为s的连续正数序列（至少含有两个数）。例如，输入15，由于1+2+3+4+5=4+5+6=7+8=15，所以打印出3个连续序列1-5、4-6、7-8
    def FindContinuousSequence(self, tsum):
        if tsum<=2:
            return []
        left = 1
        right = 2
        middle = (tsum+1)/2
        sum = left + right
        res = []
        while left<middle:
            if sum == tsum:
                res.append(self.PrintSum(left,right))
            while sum>tsum and left<middle:
                sum = sum - left
                left = left + 1
                if sum == tsum:
                    res.append(self.PrintSum(left, right))
            right = right + 1
            sum = sum + right
        return res
    def Getsum(self,left,right):
        if left>right:
            return False
        sum = 0
        for i in range(left,right+1):
            sum = sum + i
        return sum
    def PrintSum(self,left,right):
        res_sub = []
        for i in range(left,right+1):
            res_sub.append(i)
        return res_sub
    # 面试题58：反转字符串。
    # 题目一：输入一个英文句子，反转句子中单词的顺序，但单词内字符的顺序不变，为简单起见，标点符号和普通字母一样处理。
    # 例如输入字符串 I am a student.,输出为student. a am I
    # 翻转再翻转的方法：首先将整个字符串翻转：.tneduts a ma I 然后再对每个空格间隔到的字符进行翻转。便得到结果
    def ReverseSentence(self, s):
        if s is None or len(s)<=0:
            return ''
        sList = list(s)
        sList = self.Reverse(sList)
        start = 0
        end = 0
        res = ''
        listtemp=[]
        while end<len(s):
            if end == len(s)-1:
                listtemp.append(self.Reverse(sList[start:]))
                break
            if sList[start]==' ':
                start = start +1
                end = end+1
                listtemp.append(' ')
            elif sList[end]==' ':
                listtemp.append(self.Reverse(sList[start:end]))
                start = end
            else:
                end = end +1
        print listtemp
        for i in range(len(listtemp)):
            res = res + ''.join(listtemp[i])
        return res
    def Reverse(self,list):  # 对一个数组的翻转
        if list is None or len(list)<=0:
            return ''
        start = 0
        end = len(list)-1
        while start<end:
            list[start],list[end]=list[end],list[start]
            start = start +1
            end = end -1
        return list

    # 法二：
    def ReverseSentence2(self, s):
        l = s.split(' ')
        return ' '.join(l[::-1])
    # 题目二：左旋转字符串。字符串的左旋转是把字符串前面的若干字符转移到字符串的尾部。请定义一个函数实现字符串左旋转操作的功能。
    # 比如，输入字符串abcdefg和数字2，该函数将返回左旋转两位得到的结果cdefgab
    # 按照上面的方法就是翻转三次，首先翻转整个字符串，然后根据n值，将字符串分为两部分，两部分再分别翻转即可。
    def LeftRotateString(self, s, n):
        if s is None or len(s)<=0:
            return ''
        if n == len(s):
            return s
        s1 = s[:n]
        s2 = s[n:]
        return s2+s1
    # 面试题59：队列的最大值
    # 题目一：滑动窗口的最大值。给定一个数组和滑动窗口的大小，请找出所有滑动窗口里的最大值。
    # 例如，如果输入数组{2,3,4,2,6,2,5,1}及滑动窗口的大小3，那么一共存在6个滑动窗口，他们的最大值分别是{4,4,6,6,6,5}
    def maxInWindows(self, num, size):
        # write code here
        if num is None or len(num) < size or size==0:
            return []
        number = len(num)-size+1
        res = []
        for i in range(number):
            subNum = list(num[i:i+size])
            maxNumber = max(subNum)
            res.append(maxNumber)
        return res
    # n个骰子的点数
    # 把n个骰子扔到地上，所有骰子朝上一面的点数之和为s，输入n,打印出s所有可能的值出现的概率。
    def PrintProbability(self,number):
        if number<1:
            return
        maxSum = 6*number
        probabilities = [0]*(maxSum-number+1)
        self.Probability(number,probabilities)
        total = 6.0**number
        for i in range(len(probabilities)):
            print '和为：'+str(i+number)+'时，概率为：'+str(probabilities[i]/total)+',次数为:'+str(probabilities[i])
        del probabilities
    def Probability(self,number,propabilities):
        for i in range(1,7):
            self.Probability_sub(number,number,i,propabilities)
    def Probability_sub(self,number,current,sum,propabilities):
        print number,current,sum,propabilities
        if current ==1:
            propabilities[sum-number]+=1
        else:
            for i in range(1,7):
                self.Probability_sub(number,current-1,i+sum,propabilities)

    def PrintProbability2(self,number):
        if number<1:
            return
        if number == 1:
            for i in range(1,7):
                print '和为：'+str(i)+'时，概率为：'+str(1/6)+',次数为:'+str(1)
                break
        maxVal = 6
        probStorage=[[],[]]
        probStorage[0]=[0]*(maxVal*number+1)
        flag=0
        for i in range(1,maxVal+1):
            probStorage[flag][i] = 1
        for time in range(2,number+1):
            probStorage[1-flag] = [0]*(maxVal*number+1)
            for current in range(time,maxVal*time+1):
                preNumber = 1
                while preNumber<current and preNumber<=maxVal:
                    probStorage[1-flag][current] += probStorage[flag][current-preNumber]
                    preNumber +=1
            flag = 1-flag
        tatal = maxVal**number
        for i in range(number,number*maxVal+1):
            ratio = probStorage[flag][i]/float(tatal)
            print '和为：'+str(i)+'时，概率为：'+str(ratio)+',次数为:'+str(probStorage[flag][i])

    # 面试题61：扑克牌中的顺子。从扑克牌中随机抽5张牌，判断是不是一个顺子，即这5张牌是不是连续的。
    # 2-10为数字本身，A为1，J为11，Q为12，K为13.而大小王为任意数字。
    def IsContinuous(self, numbers):
        count_0 = 0
        for i in range(0,len(numbers)):
            if numbers[i]=='A':
                numbers[i]=1
            elif numbers[i]=='J':
                numbers[i]=11
            elif numbers[i]=='Q':
                numbers[i]=12
            elif numbers[i]=='K':
                numbers[i]=13
            elif numbers[i]==0:
                count_0+=1
        if count_0==4:
            return True
        numbers.sort()
        chaSum = 0
        for j in range(count_0,len(numbers)-1):
            if numbers[j+1]==numbers[j]:
                return False
            cha = numbers[j+1]-numbers[j]-1
            chaSum+=cha
        if count_0>=chaSum:
            return True
        return False

    # 面试题62:圆圈中最后剩下的数字。0，1，。。。n-1这n个数字排成一个圆圈。从数字0开始，每次从这个圆圈里删除第m个数字，求出这个圆圈里剩下的最后一个数字。
    def LastRemaining_Solution(self, n, m):
        if n < 1 or m < 1:  # n可以小于m
            return -1
        people = range(n)
        i = 0
        for num in range(n,1,-1):
            i = (i+m-1)%num
            print i
            people.pop(i)
        return people[-1]

    # 面试题63：股票的最大利润。假设把某股票的价格按照时间先后顺序存储在数组中，请问卖卖该股票一次可能获得的最大利润。
    # 例如9，11，8，5，7，12，16，14.如果我们在5的时候买入，16卖出，有max
    def maxDiff(self,numbers):
        if numbers is None or len(numbers)<=1:
            return False
        buy = numbers[0]
        sell = numbers[1]
        Max = sell-buy
        for i in range(2,len(numbers)):
            if numbers[i-1]<buy:
                buy = numbers[i-1]
            CurrentMax = numbers[i]-buy
            if CurrentMax>Max:
                Max = CurrentMax
        return Max

    # 面试题64：求1+2+3+。。+n。
    # 求1+2+3+...+n，要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）。
    def __init__(self):
        self.sum = 0
    def Sum_Solution(self, n):
        def qiusum(n):
            self.sum += n
            n -= 1
            return n > 0 and self.Sum_Solution(n)
        qiusum(n)
        return self.sum

    # 面试题65：不用加减乘除做加法
    # 写一个函数，求两个整数之和，要求在函数体内不得使用+、-、*、/四则运算符号。
    # 分为三步，首先求两个二进制的相加（不进位），接着统计进位（求与再向左移动一位）
    def Add(self, num1, num2):
        Step1 = num1 ^ num2
        Step2 = (num1 & num2)<<1
        res = Step1 + Step2
        return res

    # 面试题66：构建乘积数组。
    # 给定一个数组A[0,1,...,n-1],请构建一个数组B[0,1,...,n-1],其中B中的元素B[i]=A[0]*A[1]*...*A[i-1]*A[i+1]*...*A[n-1]。不能使用除法
    def multiply(self, A):
        if A is None or len(A)<=0:
            return []
        B = [None]*len(A)
        B[0]=1
        for i in range(1,len(A)):
            B[i] = B[i-1] *A[i-1]   # 下三角的值
        tmp = 1
        for i in range(len(A)-2,-1,-1):
            tmp = tmp *A[i+1]
            B[i] = B[i] * tmp
        return B
    # 面试题67:将字符串转换为整数。
    def StrToInt(self, s):
        if s is None or len(s)<=0:
            return 0
        flag = 1
        First = s[0]
        res = 0
        if First=='-':
            flag = -1
            s = s[1:]
        elif First=='+':
            flag = 1
            s = s[1:]
        for i in range(0,len(s)):
            cur_ord = ord(s[i])
            if cur_ord>=48 and cur_ord<=57:
                res = res *10+cur_ord-48
            else:
                return 0
        return flag*res
    # 面试题68:树中两个节点的最低公共祖先
    # 对于二叉树
    def lowestCommonAncestor1(self, root, A, B):  # A,B表示两个节点
        if root is None:
            return False
        while(root.val<A and root.val<B) or (root.val>A and root.val>B):
            if root.val<A and root.val<B:
                root = root.right
            else:
                root = root.left
        return root
    #对于普通的树，树中的节点有指向父节点的指针，看成求两个链表的第一个公共节点
    def lowestCommonAncestor2(self, root, A, B):  # A,B表示两个节点
        if root is None:
            return False
        len_A = 1
        len_B = 1
        C = E = ListNode(A.val)
        D = F = ListNode(B.val)
        while(A!=root): #  求得A节点到根节点的路径存放到链表中
            A = A.parent
            C.next = ListNode(A.val)
            C = C.next
            len_A = len_A +1
        while(B!=root):   # 求得B节点到根节点得路径存放到链表中
            B = B.parent
            D.next = ListNode(B.val)
            D = D.next
            len_B = len_B + 1
        Diff = abs(len_A-len_B)
        if len_A>len_B:
            pListHeadLong = E
            pListHeadShort = F
        else:
            pListHeadLong = F
            pListHeadShort = E
        for i in range(Diff):
            pListHeadLong = pListHeadLong.next
        while pListHeadLong != 0 and pListHeadShort != 0 and pListHeadLong.val != pListHeadShort.val:
            pListHeadLong = pListHeadLong.next
            pListHeadShort = pListHeadShort.next
        pFirstCon = pListHeadShort
        return pFirstCon.val

    # 对于普通的树，没有指向父节点的指针。
    def lowestCommonAncestor3(self, root, A, B):  # A,B表示两个节点
        if root == None:
            return False
        pathA = self.storeNodes(root, A)[0]
        pathB = self.storeNodes(root, B)[0]
        if pathA and pathB:
            lenA, lenB = len(pathA), len(pathB)
            diff = abs(lenA - lenB)
            if lenA > lenB:
                markA = lenA - diff - 1
                markB = lenB - 1
            else:
                markA = lenA - 1
                markB = lenB - diff - 1
            while markA >= 0 and markB >= 0:  # 判断第一个从目标到根相同的点，即最低公共点
                if pathA[markA] == pathB[markB]:
                    return pathA[markA].val  # 返回公共节点的值
                markA -= 1
                markB -= 1

    def storeNodes(self, root, targetNode):   # 从根节点到目标节点的路径
        if root == None or targetNode == None:
            return []
        elif root.val == targetNode.val:
            return [[targetNode]]
        stack = []
        if root.left:
            stackLeft = self.storeNodes(root.left, targetNode)
            # print stackLeft
            for i in stackLeft:
                i.insert(0, root)
                # for j in range(0,len(i)):
                #     print i[j].val,
                stack.append(i)
                # print stack
        if root.right:
            stackRight = self.storeNodes(root.right, targetNode)
            for i in stackRight:
                i.insert(0, root)
                stack.append(i)
        return stack

    def preorder(self,tree): # 先序遍历
        if tree:
            print tree.val
            self.preorder(tree.left)
            self.preorder(tree.right)

class ListNode:
    def __init__(self,x):
        self.val = x
        self.next = None

class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
        self.parent = None
import random
if __name__ == "__main__":
    k=Solution6()
    s = [1, 2, 3, 3, 3, 3, 4, 5]
    #print k.GetNumberOfK2(s,3)
    #print 0^1^2^3^5^0^1^2^3^4^5
    #print k.GetMissingNumber3([0,1,2,3,5])

    a = [-3,-1,1,3,5]
    #print k.GetNumberSameAsIndex2(a)
    b = [4,4,4,5,6,6,6]
    #print k.FindNumsAppearOnce1(b)
    c = [1,2,4,7,11,15]
    #print k.FindNumbersWithSum(c,15)
    #print k.FindContinuousSequence(15)
    s ='I am a student.'
    #print k.ReverseSentence(s)
    #print k.LeftRotateString(s,3)
    #print k.PrintProbability2(2)
    #print k.IsContinuous(['A',3,4,5,0])
    #print k.LastRemaining_Solution(5,3)
    #print k.maxDiff([9,11,8,5,7,12,16,14])

    #print k.Sum_Solution(5)
    #print k.Add(99,17)
    #print k.StrToInt('1a')

    node1 = TreeNode(9)
    node2 = TreeNode(4)
    node3 = TreeNode(2)
    node4 = TreeNode(1)
    node5 = TreeNode(3)
    node6 = TreeNode(8)
    node7 = TreeNode(7)
    node8 = TreeNode(5)
    node9 = TreeNode(6)

    node1.left = node2
    node1.right = node6
    node2.left = node3
    node2.right = node7
    node3.left = node4
    node3.right = node5
    node7.left = node8
    node7.right = node9

    node5.parent = node3
    node4.parent = node3
    node8.parent = node7
    node9.parent = node7
    node3.parent = node2
    node7.parent = node2
    node2.parent = node1
    node6.parent = node1

    #print k.lowestCommonAncestor3(node1,node3,node7)
    #print k.preorder(node1)
    print random.choice([1,2,3,5,6])


