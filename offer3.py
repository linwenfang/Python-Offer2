# -*- coding:utf-8 -*-

class Solution3:
    # 面试题26：数值的整数次方。给定一个double类型的浮点数base和int类型的整数exponent。求base的exponent次方。
    # 法一，常规做
    # 注意底数为0，和次方为负数时候的影响
    def Power(self, base, exponent):
        if base == 0.0 and exponent <= 0.0:
            return 0.0
        elif exponent == 0.0:
            return 1.0
        elif exponent < 0.0:
            res = self.Multi(base, abs(exponent))
            return 1.0 / res
        else:
            return self.Multi(base, exponent)

    def Multi(self, base, expon):
        res = base
        for i in range(1, expon):
            res = res * base
        return res
    # 法二：改进版，求解Multi乘积时候的优化。例如求一个数的32次方，我们可以根据求得的2次方得到4次方，从而得到8，16，21次方，不需要循环31次进行乘机
    def Power2(self, base, exponent):
        if base == 0.0 and exponent <= 0.0:
            return 0.0
        elif exponent == 0.0:
            return 1.0
        elif exponent < 0.0:
            res = self.Multi2(base, abs(exponent))
            return 1.0 / res
        else:
            return self.Multi2(base, exponent)

    def Multi2(self, base, expon):
        if expon == 0:
            return 1;
        if expon ==1:
            return base
        res = self.Multi2(base,expon >> 1)   # 判断是否被2整除，位运算的效率比运算符的效率高
        if expon & 0x1 == 0:   # 最后一位是1的话就不能被2整除,0x表示16进制
            res = res * res
        else:
            res = res * res * base
        return res

    # 面试题17：打印从1到最大的n位数。输入数字n,按顺序打印出从1到最大的n位十进制数。
    # Python中已经对大整数可以进行自动转换了，所以不需要考虑大整数溢出问题
    def PrintToMax(self,n):
        if n <=0:
            return
        for i in range(1,10**n):
            print i,

    # 考虑溢出的思路，可转化为java或c的
    def PrintToMax2(self,n):
        if n <=0:
            return
        number = '0' * n
        number = number[:len(number)-1] + '1'

        isOverflow = False
        while not isOverflow:
            self.PrintNumber(number)
            nTakeOver = 0   # 满10 进位
            nLength = len(number)

            for i in range(nLength - 1, -1, -1):
                nSum = int(number[i]) + nTakeOver
                if i == nLength - 1:
                    nSum += 1

                if nSum >= 10:
                    if i == 0:
                        isOverflow = True
                    else:
                        nSum -= 10
                        nTakeOver = 1
                        number = number[0:i] + str(nSum) + number[i + 1:]
                        # number[i] = str(nSum)   字符串不能替换其中的某一个元素，需要转成list
                else:
                    number = number[0:i] + str(nSum) + number[i + 1:]
                    # number[i] = str(nSum)
                    break

    def PrintNumber(self,number):
        isBeginning0 = True
        nLength = len(number)

        for i in range(nLength):
            if isBeginning0 and number[i] != '0':
                isBeginning0 = False
            if not isBeginning0:
                print(number[i]),
        print ' ',

    # 法三： 0-9数据的全排列
    def PrintToMax3(self,n):
        if n<=0:
            return
        number = ['0'] * n
        for i in range(10):
            number[0] = str(i)
            self.Print1ToMaxOfDigitsRecursiverly(number,n,0)

    def Print1ToMaxOfDigitsRecursiverly(self,number,length,index):
        if index == length - 1:
            self.PrintNumber(number)
            return
        for i in range(10):
            number[index+1] = str(i)
            self.Print1ToMaxOfDigitsRecursiverly(number,length,index+1)

    #面试题18：删除链表的节点
    # 题目一：在O（1）时间内删除链表节点。给定单向链表的头指针和一个节点指针，定义一个函数在o（1）时间内删除该节点。
    # 如果要删除的点不在该链表里呢？
    def deleteNode(self, pHead, pToBeDelete):
        if not pHead or not pToBeDelete:   # 如果为空
            return
        if pToBeDelete.next != None:  # 要删除的节点不是尾节点
            pNext = pToBeDelete.next
            pToBeDelete.val = pNext.val
            pToBeDelete.next = pNext.next
            pNext.__del__()
        elif pHead == pToBeDelete:  # 链表只有一个节点，删除头节点
            pHead.__del__()
            pToBeDelete.__del__()
        else:   # 链表中有多个节点，删除尾节点
            pNode = pHead
            while pNode.next != pToBeDelete:
                pNode = pNode.next
            pNode.next = None
            pToBeDelete.__del__()
    # 题目二：删除链表中重复的节点。例如1——2——3——3——5，输出为1——2——5
    def deleteDuplication(self, pHead):
        if not pHead or not pHead.next:
            return pHead
        pPreNode = None
        pNode = pHead
        while (pNode!=None):  # 链表的遍历
            pNext = pNode.next
            needDel = False
            if pNext is not None and pNext.val == pNode.val:  # 当遇到重复的时候将标签改为true
                needDel = True
            if needDel == False:   # 如果不需要删除，那么就向后移一位。包括记录的前节点和当前节点的后移
                pPreNode = pNode
                pNode = pNode.next
            else:   # 需要删除时，定义一个删除节点
                value = pNode.val
                pToBeDel = pNode
                while pToBeDel is not None and pToBeDel.val == value:
                    pToBeDel = pToBeDel.next
                if pPreNode ==None:
                    pHead = pToBeDel
                    pNode = pToBeDel
                    continue
                else:
                    pPreNode.next = pToBeDel
                pNode = pPreNode
        return pHead
    # 面试题19：正则表达式匹配
    # 请实现一个函数用来匹配包括'.'和'*'的正则表达式。模式中的字符'.'表示任意一个字符，而'*'表示它前面的字符可以出现任意次（包含0次）。
    # 在本题中，匹配是指字符串的所有字符匹配整个模式。例如，字符串"aaa"与模式"a.a"和"ab*ac*a"匹配，但是与"aa.a"和"ab*a"均不匹配
    def match(self,s,pattern):
        if len(s)==0 and len(pattern)==0:
            return True
        elif len(s)>0 and len(pattern)==0:
            return False
        elif len(pattern)>1 and pattern[1]=='*':
            if len(s)>0 and (s[0] == pattern[0] or pattern[0]=='.'):
                return self.match(s,pattern[2:]) or self.match(s[1:],pattern[2:]) or self.match(s[1:],pattern)       # 分别为*号被利用0次，1次，n次
            else:  # 第一位不相等
                return self.match(s,pattern[2:])
        elif len(s)>0 and (pattern[0] == '.' or pattern[0] == s[0]):
            return self.match(s[1:],pattern[1:])
        return False

    # 面试题20：表示数值的字符串。
    # 请实现一个函数用来判断字符串是否表示数值（包括整数和小数）。
    # 例如，字符串"+100","5e2","-123","3.1416"和"-1E-16"都表示数值。 但是"12e","1a3.14","1.2.3","+-5"和"12e+4.3"都不是。
    # 数字的格式A.B e|E C,A表示整数部分，可以有+-符号,可以为空，如果A为空，那么必须有B，遇到点之后，则接下来为B部分，即小数部分，不能有+-，E|e表示指数，后面为C部分，即次方，可以有+-
    import re
    def isNumeric(self,s):
        return re.match(r"^[\+|\-]?[0-9]*(\.[0-9]*)?([e|E][\+\-]?[0-9]+)?$", s)
    # ^ 匹配字符串开头 $ 匹配字符串结尾   ？表示前一个字符出现0或1次，
    # *表示前面的字符出现》=0次。[ ]表示匹配字符集，|匹配左右任意表达式  + 表示前一字符出现》=1次

    def isNumeric2(self,s):
        try:
            ss = float(s)
            print ss
            return True
        except:
            return False
    # 面试题21：调整数组顺序使奇数位于偶数前面。
    # 输入一个整数数组，实现一个函数来调整该数组中数字的顺序，
    # 使得所有的奇数位于数组的前半部分，所有的偶数位于位于数组的后半部分，并保证奇数和奇数，偶数和偶数之间的相对位置不变。
    def reOrderArray(self, array):  # 不考虑位置改变时的简单做法
        if len(array) ==0:
            return array
        if len(array) ==1:
            return array
        start = 0
        end = len(array)-1
        while start<end:
            while(start<end and array[start]&0x1 !=0):  # 向后移动，直到它指向偶数
                start= start+1
            while(start<end and array[end]&0x1 ==0):    # 向前移动，知道它指向奇数
                end = end -1
            if start<end :
                temp = array[end]
                array[end] = array[start]
                array[start] = temp
        return array
    # 考虑不破坏原始的相对位置
    def reOrderArray2(self, array):
        # write code here
        if array is None:
            return False
        jishu = []
        oushu = []
        for i in range(len(array)):
            if array[i] & 0x1 != 0:
                jishu.append(array[i])
            else:
                oushu.append(array[i])
        return jishu + oushu





class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
    def __del__(self):
        self.val = None
        self.next = None

if __name__ == "__main__":
    k = Solution3()
    #print k.PrintToMax3(3)

    # node1 = ListNode(10)
    # node2 = ListNode(11)
    # node3 = ListNode(13)
    # node4 = ListNode(13)
    # node5 = ListNode(19)
    # node1.next = node2
    # node2.next = node3
    # node3.next = node4
    # node4.next = node5
    #
    # k.deleteDuplication(node1)
    # print node2.next.val

    #print k.isNumeric2('12.+67')

    s = [1,2,3,4,5]
    print s[1]
    print k.reOrderArray(s)


