# -*- coding:utf-8 -*-
import itertools
class Solution:
    # 插入排序——直接插入排序
    def sort1(self,list):
        for i in range(1,len(list)):
            get = list[i]    # 手里拿到的牌
            base = i-1       # 与其前面的牌进行比较
            while(base>=0 and list[base]>get):
                list[base+1] = list[base]
                base = base - 1
            list[base+1] = get
        return list

    # 插入排序——二分插入排序
    def sort2(self,list):
        for i in range(1,len(list)):
            get = list[i]    # 手里拿到的牌
            start = 0
            end = i-1
            while(start<=end):    # 二分查找插入位置
                mid = (start+end)/2
                if list[mid] <= get:
                    start = mid+1
                elif list[mid]>get:
                    end = mid-1
            for j in range(i,start,-1):    # 将插入位置以后的均后移一位
                list[j] = list[j-1]
            list[start] = get
        return list

    # 插入排序——希尔排序
    def sort3(self,list):    # 将数组按照等间距分成几个组，对小组进行组内的一个插入排序，不断缩小抽取的间距，最后其实为直接插入排序，间距为1
        step = 2   # 间距初始值设置为2，接下来的间距会不断缩短一半
        group = len(list)/step   # 组数
        while group>0:
            for i in range(0,group):
                for j in range(i+group,len(list),group):   # 组内进行插入排序，
                    get = list[j]   # 手里拿到的牌
                    base = j-group   # 与其前面的牌进行比较，从后向前的方式进行比较
                    while base>=0 and list[base]>get:
                            list[base+group]=list[base]
                            base = base-group
                    list[base+group]=get
            group=group/step     # 组数由间距设定
        return list

    # 选择排序——简单选择排序
    def sort4(self,list):
        for i in range(0,len(list)-1):
            min = list[i]
            for j in range(i+1,len(list)):    # 从未排序的序列中找到最小值和最小值对应的标号
                if list[j]<min:
                    min = list[j]
                    label = j
            tmp = list[i]
            list[i]=min
            list[label]=tmp
        return list

    # 交换排序——冒泡排序
    def sort5(self,list):
        length = len(list)
        while length>=0:
            for i in range(0,length-1):
                if list[i] > list[i+1]:
                    list[i], list[i+1] = list[i+1], list[i]
            length = length - 1
        return list
    # 交换排序——冒泡排序的改进——鸡尾酒排序,
    def sort6(self,list):
        start = 0
        end = len(list)-1
        while(start < end):
            for i in range(start,end):
                if list[i] > list[i+1]:
                    list[i], list[i+1] = list[i+1], list[i]
            end = end - 1
            for j in range(end-1,start-1,-1):
                if list[j] > list[j+1]:
                    list[j], list[j+1] = list[j+1], list[j]
            start = start + 1
        return list

    # 交换排序——快速排序
    def sort7(self,list,left,right):
        if left >= right:
            return
        base = list[right]
        flag = left
        for i in range(left,right):
            if list[i] <= base:
                list[flag],list[i]=list[i],list[flag]
                flag = flag + 1
        list[flag],list[right] = list[right],list[flag]
        self.sort7(list,left,flag-1)
        self.sort7(list,flag+1,right)
        return list

    # 归并排序
    def sort8(self, list):
        if len(list) <= 1:  # 子序列
            return list
        mid = (len(list) / 2)
        left = self.sort8(list[:mid])  # 递归的切片操作
        right = self.sort8(list[mid:len(list)])
        result = []
        i,j=0,0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                result.append(left[i])
                i = i + 1
            else:
                result.append(right[j])
                j = j + 1
            print result
        result += left[i:]
        result += right[j:]
        return result


    # 堆排序,小顶堆
    def sort9(self,lists):
        size = len(lists)
        self.build_heap(lists, size)
        for i in range(0, size)[::-1]:
            lists[0], lists[i] = lists[i], lists[0]
            self.adjust_heap(lists, 0, i)
        return lists

    # 调整堆
    def adjust_heap(self,lists, i, size):
        lchild = 2 * i + 1
        rchild = 2 * i + 2
        max = i
        if i < size / 2:
            if lchild < size and lists[lchild] > lists[max]:
                max = lchild
            if rchild < size and lists[rchild] > lists[max]:
                max = rchild
            if max != i:
                lists[max], lists[i] = lists[i], lists[max]
                self.adjust_heap(lists, max, size)

    # 创建堆
    def build_heap(self,lists, size):
        for i in range(0, (size / 2))[::-1]:    # 将0到size/2-1 倒序赋值
            self.adjust_heap(lists, i, size)






    def Permutation(self, ss):    # 牛客网
        if not ss:
            return []
        res=[]
        for i in itertools.permutations(ss, len(ss)):
            tmp = ''.join(i)
            if tmp not in res:
                res.append(tmp)
        return res

if __name__ == "__main__":
    k=Solution()
    list=[49,38,65,97,76,13,27,49]
    print k.sort9(list)


    # ss = '123'        # 迭代器，itertools。permutations(p[,r]);返回p中任意取r个元素做排列的元组的迭代器
    # res = []
    # for i in itertools.permutations(ss,len(ss)):
    #     tmp = ''.join(i)
    #     res.append(tmp)
    # print res


