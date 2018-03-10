# -*- coding:utf-8 -*-
import heapq
import numpy as np
class Solution5:
    # 面试题39： 数组中出现次数超过一般的数字
    # 数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。
    # 例如输入一个长度为9的数组{1,2,3,2,2,2,5,4,2}。由于数字2在数组中出现了5次，超过数组长度的一半，因此输出2。如果不存在则输出0。
    def MoreThanHalfNum_Solution1(self, numbers):    #  排序后，中间的数为出现次数最多的值，但是需要考虑是否存在这个值
        # write code here
        numbers.sort()
        mid = len(numbers)/2
        count =0
        for num in numbers:
            if num == numbers[mid]:
                count = count+1
        if count>mid:
            return numbers[mid]
        else:
            return 0

    def MoreThanHalfNum_Solution2(self, numbers):
        # 如果有符合条件的数字，则它出现的次数比其他所有数字出现的次数和还要多。
        # 在遍历数组时保存两个值：一是数组中一个数字，一是次数。遍历下一个数字时，若它与之前保存的数字相同，则次数加1，否则次数减1；
        # 若次数为0，则保存下一个数字，并将次数置为1。遍历结束后，所保存的数字即为所求。然后再判断它是否符合条件即可。
        if numbers is None:
            return 0
        res = numbers[0]
        count = 1
        rescount=0
        for i in range(1,len(numbers)):
            if numbers[i] == res:
                count = count +1
            else:
                count = count - 1
            if count == 0:
                res = numbers[i]
                count =1
        # 还是要遍历找到出现的次数
        for num in numbers:
            if num == res:
                rescount = rescount+1
        if rescount>(len(numbers/2)):
            return res
        else:
            return 0

    #  面试题40：最小的K个数
    # 输入n个整数，找出其中最小的K个数。例如输入4,5,1,6,2,7,3,8这8个数字，则最小的4个数字是1,2,3,4,。
    def GetLeastNumbers_Solution1(self, tinput, k):
        if len(tinput)<k or len(tinput)==0:
            return []
        return sorted(tinput)[:k]

    def GetLeastNumbers_Solution2(self,tinput,k):
        if len(tinput)<k or len(tinput)==0:
            return []
        start = 0
        end = len(tinput)-1
        flag = self.sort(tinput,start,end)
        while flag!=k-1:
            if flag>k-1:
                end = flag-1
                flag = self.sort(tinput,start,end)
            else:
                start = flag+1
                flag = self.sort(tinput,start,end)
        return tinput[:k]

    def sort(self, list, left, right):   # 快速排序第一次排序
        if left >= right:
            return
        base = list[right]
        flag = left
        for i in range(left, right):
            if list[i] <= base:
                list[flag], list[i] = list[i], list[flag]
                flag = flag + 1
        list[flag], list[right] = list[right], list[flag]
        return flag

    def GetLeastNumbers_Solution3(self,tinput,k):
        if len(tinput)<k or len(tinput)==0:
            return []
        output = []
        for number in tinput:
            if len(output) < k:
                output.append(number)
            else:
                output = heapq.nlargest(k, output)  # 构建最大堆   heapq.nsmallest(k, output) 构建最小堆
                if number >= output[0]:
                    continue
                else:
                    output[0] = number
                return output
    # 面试题41：数据流中的中位数。如何得到一个数据流中的中位数？如果从数据流中读出奇数个数值，那么中位数就是所有数值排序之后位于中间的数值
    # 如果从数据流中读出的是偶数个数值，那么就是排序之后中间两个数的平均值。
    def Insert(self, num):
        data = []
        data.append(num)
        num.sort()
        return num

    def GetMedian(self,num):
        if len(num)&0x1 == 1:
            return num[len(num)/2]
        else:
            return (num[len(num)/2]+num[len(num)/2-1])/2.0

    # 法二：根据最大最小堆原理
    def __init__(self):
        self.left = []
        self.right = []
        self.count = 0
    def Insert2(self, num):
        for i in range(0,len(num)):
            if self.count & 1 == 0:
                self.left.append(num[i])
            else:
                self.right.append(num[i])
            self.count += 1
    def GetMedian2(self):
        if self.count ==1:
            return self.left[0]
        self.left = heapq.nlargest(len(self.left), self.left)
        self.right = heapq.nsmallest(len(self.right),self.right)

        while self.left[0] > self.right[0]:
            self.left[0],self.right[0] = self.right[0],self.left[0]
            self.left = heapq.nlargest(len(self.left), self.left)
            self.right = heapq.nsmallest(len(self.right), self.right)

        if self.count&0x1 == 1:
            return self.left[0]
        else:
            return (self.left[0]+self.right[0])/2.0

    # 面试题42：连续子数组的最大和。输入一个整型数组，数组里有正数也有负数。数组中的一个或连续多个整数组成一个子数组。求所有子数组的和的最大值。要求时间复杂度o(n)
    # 找到规律，什么时候舍弃前面的数值
    def FindGreatestSumOfSubArray(self, array):
        if array is None:
            return False
        sum = 0
        max = float('-inf')
        for i in range(0,len(array)):
            sum = sum + array[i]
            if sum>max:
                max = sum
            if sum < 0 :
                sum = 0
        return max
    # 动态规划   创建一个数值来保存遍历到第i位时的和。对于第i位来说，如果前面的和比0小，那么就要舍弃前面的和，重新赋值求和，如果比0大，那么就在此基础上累加
    def FindGreatestSumOfSubArray2(self, array):
        if array is None:
            return False
        list = [0]*len(array)
        for i in range(len(array)):
            if i ==0 or list[i-1]<=0:
                list[i] = array[i]
            else:
                list[i] = list[i-1] + array[i]
        return max(list)

    # 面试题43：1-n整数中1出现的次数：求出1~13的整数中1出现的次数,并算出100~1300的整数中1出现的次数？
    # 为此他特别数了一下1~13中包含1的数字有1、10、11、12、13因此共出现6次,但是对于后面问题他就没辙了。ACMer希望你们帮帮他,并把问题更加普遍化,可以很快的求出任意非负整数区间中1出现的次数
    # 简单做法，超时，容易理解，全部遍历，求每个数里有几个1.复杂度为nlogn
    def NumberOf1Between1AndN_Solution(self, n):
        if n <=0:
            return 0
        count = 0
        for i in range(0,n):
            number = 0
            while i!=0:
                if i%10 ==1:
                    number = number+1
                i= i/10
            count = count+number
        return count

    # 法二：找规律,对于求解任意的一个数x出现的次数，那么就将后面if判断语句中的==1  ==0 改为==x,<x即可
    # 1. 如果第i位（自右至左，从1开始标号）上的数字为0，则第i位可能出现1的次数由更高位决定（若没有高位，视高位为0），等于更高位数字X当前位数的权重10i-1。
    # 2. 如果第i位上的数字为1，则第i位上可能出现1的次数不仅受更高位影响，还受低位影响（若没有低位，视低位为0），等于更高位数字X当前位数的权重10i-1+（低位数字+1）。
    # 3. 如果第i位上的数字大于1，则第i位上可能出现1的次数仅由更高位决定（若没有高位，视高位为0），等于（更高位数字+1）X当前位数的权重10i-1。
    def NumberOf1Between1AndN_Solution2(self, n):
        if n <=0:
            return 0
        high_wei = n
        count = 0
        flag = 1
        while high_wei!=0:
            high_wei = n/(10**flag)
            tmp = n%(10**flag)
            print tmp
            curr_wei = tmp/(10**(flag-1))
            low_wei = tmp%(10**(flag-1))
            print high_wei,curr_wei,low_wei

            if curr_wei == 0:
                count = count + (high_wei)*(10**(flag-1))
                print '0'+str(count)
            elif curr_wei ==1:
                count = count + (high_wei)*(10**(flag-1)) + low_wei +1
                print '1'+str(count)
            else:
                count = count + (high_wei+1)*(10 ** (flag - 1))
                print 'else'+str(count)
            flag = flag +1
        return count

    # 面试题44：数字序列中某一位的数字。数字以0123456789101112....的格式序列化到一个字符序列中，在这个序列中，第5位时5，第13位时1，第19位时4。
    # 请写一个函数，求任意第n位对应的数字。 通过枚举找到规律。0-9，10个。10-99，90个，180位。100-999，900个，270个。
    def digitAtIndex(self, index):
        if index <0:
            return False
        if index<10:
            return index
        digit = 1
        number = 0
        while number<=index:
            number = number + self.countOfInteger(digit)
            digit = digit +1
        digit = digit - 1  # 要求的数在digit位数当中
        return self.beginAtIndex(index,digit)

    def countOfInteger(self,digit):   # digit位数时，有几个数字
        if digit ==1:
            return 10
        else:
            return 9*(10**(digit-1))*(digit)

    def beginAtIndex(self,index,digit):  # 求在digit位数的第几位
        remain = index
        for i in range(1,digit):
            remain = remain - self.countOfInteger(i)
        number = remain/digit + 10**(digit-1)
        beigin = remain%digit
        return str(number)[beigin]


    # 面试题45：把数组排成最小的数。输入一个正整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。例如输入数组{3，32，321}，则打印出这三个数字能排成的最小数字为321323。
    # 类似冒泡排序，通过两两比较，找到最大的数，将其放在一个新的list中，每次都通过两两比较，冒泡方法可以得到数组的最后一位为最大值
    # 因此每次循环一次进行比较结束后，将最后一位输出到list中，最后list倒叙排列则得到最后的结果。
    def PrintMinNumber(self, numbers):
        list=[]
        length = len(numbers)
        while length > 0:
            for i in range(0, length - 1):
                if (str(numbers[i])+str(numbers[i+1]))>(str(numbers[i+1])+str(numbers[i])):   # 因为mn和nm的位数相同，换成字符串后只要比较字符串就可以了，不需要转为int
                    numbers[i], numbers[i + 1] = numbers[i + 1], numbers[i]
            print length
            list.append(numbers[length-1])
            length = length - 1

            print list
        res=''
        for i in range(len(list)-1,-1,-1):
            res = res + str(list[i])
        return res

    # 面试题46：把数字翻译成字符串。给定一个数字，我们按照如下规则把它翻译为字符串：
    # 0翻译成“a”，1翻译成“b”，。。。11 翻译成“l”，。。。25为“z”。一个数字可能有多个翻译
    # 例如12258有5中不同的翻译，分别是“bccfi”“bwfi”“bczi”“mcfi”“mzi”。请编程实现一个函数，用来计算一个数字有多少种不同的翻译方法
    # 方法：递归，对于第i位来说，它的方法个数取决于i+1之后的方法个数+(i+2)位方法的个数。f(i)= f(i+1)+g(i,i+1)f(i+1)，g(i,i+1)表示是否可以组合
    def GetTranslationCount(self,number):
        if number <0:
            return 0
        num = str(number)
        if len(num)==0 or len(num) == 1:
            return 1
        #print num[0:2]
        return self.GetTranslationCount(num[1:])+self.IsTranslate(num[0:2])*self.GetTranslationCount(num[2:])

    def IsTranslate(self,number):
        if int(number)<=25 and int(number)>=0:
            return 1
        return 0

    # 面试题47：礼物的最大价值：在一个mxn的棋盘的每一格都放有一个礼物，每个礼物都有一定的价值（价值大于0）。
    # 你可以从棋盘的左上角开始拿格子里的礼物，并每次向右或者向下移动一个，直到到达棋盘的右下角。
    # 给定一个棋盘及其上面的礼物，请计算你最多能拿到多少价值的礼物？
    def getMaxValue_solution1(self,values,rows,cols):
        if values is None or rows<=0 or cols<=0:
            return 0
        res = np.zeros((rows,cols))
        for i in range(0,rows):
            for j in range(0,cols):
                up = 0
                left = 0
                if i>0:
                    up = res[i-1][j]
                if j>0:
                    left = res[i][j-1]
                res[i][j] = max(up,left)+values[i*cols+j]
        maxValue = res[rows-1][cols-1]
        return int(maxValue)

    # 其实没有必要二维数组存放结果，对于i，j循环来说，j前面的数字只与i-1行有关。
    def getMaxValue_solution2(self,values,rows,cols):
        if values is None or rows<=0 or cols<=0:
            return 0
        res = np.zeros(cols)
        for i in range(0,rows):
            for j in range(0,cols):
                up = 0
                left = 0
                if i>0:
                    up = res[j]
                if j>0:
                    left = res[j-1]
                res[j] = max(up,left)+values[i*cols+j]
        maxValue = res[cols-1]
        return int(maxValue)

    # 面试题48：最长不含重复字符的子字符串。请从字符串中找出一个最长的不包含重复字符的子字符串，计算改最长子字符串的长度，
    # 假设字符串中只包含‘a’-'z'的字符，例如，在字符串‘arabcacfr’中，最长的不含重复子字符串是‘acfr’，长度为4
    # 列些找规律，动态规划问题，创建一个26位的数组来储存每个字母上次出现的位置，用于接下来计算两次相邻重复出现的距离差值
    # f(i) = f(i-1) + 1   或 d(d<=f(i-1))   f(i)表示长度
    def longestSubstringWithoutDuplication(self,str):
        position = [-1]*26
        maxSubtring = 0
        for i in range(0,len(str)):
            index = ord(str[i])-ord('a')
            if position[index]==-1:
                maxSubtring = maxSubtring + 1
                position[index] = i
            else:
                d = i-position[index]
                if d<=maxSubtring:
                    maxSubtring = d
                else:
                    maxSubtring = maxSubtring + 1
        return maxSubtring

    # 面试题49：丑数。把只包含因子2、3和5的数称作丑数（Ugly Number）。
    # 例如6、8都是丑数，但14不是，因为它包含因子7。 习惯上我们把1当做是第一个丑数。求按从小到大的顺序的第N个丑数
    def IsUgly(self,number):
        while(number%2==0):
            number = number/2
        while(number%3==0):
            number = number/3
        while(number%5==0):
            number = number/5
        if number==1:
            return True
        else:
            return False

    def GetUglyNumber(self,index):
        if index<=0:
            return 0
        number = 0
        count = 0
        while(count<index):
            number = number+1
            if self.IsUgly(number):
                count = count +1
        return number

    def GetUglyNumber2(self, index):
        if index<=0:
            return 0
        if index>0 and index<=6:
            return index
        res = [1,2,3,4,5,6]
        count = index - 6
        T_2 = 0
        T_3 = 0
        T_5 = 0
        while count>0:
            M = res[-1]
            while res[T_2]*2<=M:
                T_2 +=1
            while res[T_3]*3<=M:
                T_3 +=1
            while res[T_5]*5<=M:
                T_5 +=1
            print T_2,T_3,T_5
            res.append(min(res[T_2]*2,res[T_3]*3,res[T_5]*5))
            count = count - 1
        return res[-1]
    # 面试题50:第一个只出现一次的字符。
    # 题目一：字符串中第一个只出现一次的字符。例如‘abaccdeff’，输出为‘b’
    # 第一次遍历时，建立一个哈希表用于储存键值和出现的次数，再遍历一遍找到次数为1对应的键值
    def FirstNotRepeatingChar(self, s):
        if s is None or len(s)<=0:
            return -1
        alphabet = {}   # hash表的建立
        alist = list(s)
        for i in range(len(alist)):
            if not alphabet.has_key(alist[i]):  # 判断哈希表中是否有键值alist[i]
            #if alist[i] not in alphabet.keys():
                alphabet[alist[i]] = 0
            alphabet[alist[i]]+=1
        print alphabet
        for j in range(len(alist)):
            if alphabet[alist[j]] ==1:
                return alist[j] # 若返回其位置，那么就是return j
        return -1
    # 法二：在一个字符串(1<=字符串长度<=10000，全部由字母组成)中找到第一个只出现一次的字符,如果返回位置的话，就要用哈希表，建立对应关系
    # 字母26位，所以建立一个26位的list
    def FirstNotRepeatingChar2(self, s):
        if s is None or len(s)<=0:
            return -1
        count = [0]*26
        for i in range(len(s)):
            count[ord(s[i])-ord('a')]+=1
        for i in range(len(count)):
            if count[i] == 1:
                return chr(i+ord('a'))
        return -1

    # 题目二：字符流中第一个只出现一次的字符。找出字符流中第一个只出现一次的字符。
    # 例如：当从字符流中读出前两个字符‘go’时，第一个只出现一次的字符时‘g’，当从改字符流中读出前6个字符‘google’时，第一个只出现一次的字符时‘l’
    def __init__(self):
        self.adict = {}
        self.alist = []

    def FirstAppearingOnce(self):
        while len(self.alist) > 0 and self.adict[self.alist[0]] == 2:
            self.alist.pop(0)
        if len(self.alist) == 0:
            return '#'
        else:
            return self.alist[0]

    def Insert(self, char):
        if char not in self.adict.keys():
            self.adict[char] = 1
            self.alist.append(char)
        elif self.adict[char]:
            self.adict[char] = 2
    # 面试题51：数组中的逆序对。在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。
    # 输入一个数组,求出这个数组中的逆序对的总数P。并将P对1000000007取模的结果输出。 即输出P%1000000007
    def InversePairs(self, data):
        length = len(data)
        if data == None or length <= 0:
            return 0
        copy = [0] * length
        count = self.InversePairsCore(data, copy, 0, length - 1)
        return count

    def InversePairsCore(self,data,copy,start,end):
        if start == end:
            copy[start] = data[start]
            return 0
        length = (end - start) >>1
        left = self.InversePairsCore(copy, data, start, start + length)
        right = self.InversePairsCore(copy, data, start + length + 1, end)

        # i初始化为前半段最后一个数字的下标
        i = start + length
        # j初始化为后半段最后一个数字的下标
        j = end
        indexCopy = end   # 要存入数组copy中的位置
        count = 0
        while i >= start and j >= start + length + 1:
            if data[i] > data[j]:
                copy[indexCopy] = data[i]
                indexCopy -= 1
                i -= 1
                count += j - start - length
            else:
                copy[indexCopy] = data[j]
                indexCopy -= 1
                j -= 1
        while i >= start:
            copy[indexCopy] = data[i]
            indexCopy -= 1
            i -= 1
        while j >= start + length + 1:
            copy[indexCopy] = data[j]
            indexCopy -= 1
            j -= 1
        for i in range(start,end+1):
            data[i] = copy[i]
        return left + right + count

    # 面试题52：两个链表的第一个公共节点。输入两个链表，找出它们的第一个公共结点。
    def FindFirstCommonNode(self, pHead1, pHead2):
        nlength1 = self.GetListLength(pHead1)
        nlength2 = self.GetListLength(pHead2)
        nlengthDiff = abs(nlength1 - nlength2)
        if nlength1>nlength2:
            pListHeadLong = pHead1
            pListHeadShort = pHead2
        else:
            pListHeadLong = pHead2
            pListHeadShort = pHead1
        for i in range(nlengthDiff):
            pListHeadLong = pListHeadLong.next
        while pListHeadLong!=0 and pListHeadShort!=0 and pListHeadLong!=pListHeadShort:
            pListHeadLong = pListHeadLong.next
            pListHeadShort = pListHeadShort.next
        pFirstCon = pListHeadShort
        return pFirstCon

    def GetListLength(self, pHead):
        nLength = 0
        while pHead != None:
            pHead = pHead.next
            nLength += 1
        return nLength

class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
if __name__ == "__main__":
    k = Solution5()
    s = [1,4,8,3,14,6,7,9,1,7,10,6]
    a = [1,-2,3,10,-4,7,2,-5]
    b = [3,32,321]
    c = [1,10,3,8,12,2,9,6,5,7,4,11,3,7,16,5]
    #print k.GetLeastNumbers_Solution3(s,5)
    #print k.sort(s,0,5)
    #print k.Insert2(s)
    #print k.GetMedian2()
    #print k.FindGreatestSumOfSubArray2(a)

    #print k.NumberOf1Between1AndN_Solution2(21345)

    #print k.digitAtIndex(1001)
    #print k.PrintMinNumber(b)

    #print k.GetTranslationCount(12258)
    #print k.getMaxValue_solution2(c,4,4)

    #print ord('c')-ord('a')
    #print k.longestSubstringWithoutDuplication('arabcacfr')

    #print k.GetUglyNumber2(9)
    #print k.FirstNotRepeatingChar2('abaccdeff')

    data = [7,5,6,4]
    print k.InversePairs(data)

