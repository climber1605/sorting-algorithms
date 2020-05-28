# -*- coding:utf-8 -*-
import time
import random
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline

# 冒泡排序第1版，超级辣鸡，即使原数组有序仍耗时O(n^2)
def bubbleSort1(nums):
    for i in xrange(len(nums) - 1):
        for j in xrange(len(nums) - i - 1):
            if nums[j] > nums[j+1]:
                nums[j], nums[j+1] = nums[j+1], nums[j]

# 冒泡排序第2版，使用布尔变量判断一轮排序后数组是否已经有序, 最好情况耗时O(1), 然而实测发现，平均意义下的耗时反而略有增加...
def bubbleSort2(nums):
    for i in xrange(len(nums) - 1):
        isSorted = True
        for j in xrange(len(nums) - i - 1):
            if nums[j] > nums[j+1]:
                nums[j], nums[j+1] = nums[j+1], nums[j] 
                isSorted = False
        if isSorted:
            break

# 冒泡排序第3版，在第2版基础上又增加了数组无序区的边界，实测发现，平均意义下的耗时比第2版还高...
def bubbleSort3(nums):
    sortBorder = len(nums) - 1
    for i in xrange(len(nums) - 1):
        isSorted = True
        newSortBoard = sortBorder
        for j in xrange(sortBorder):
            if nums[j] > nums[j+1]:
                nums[j], nums[j+1] = nums[j+1], nums[j] 
                isSorted = False
                newSortBorder = j
        sortBoard = newSortBorder
        if isSorted:
            break

# 鸡尾酒排序，类似冒泡排序，区别在于比较和交换过程是双向的，平均性能略优于冒泡排序
def cocktailSort(nums):
    for i in xrange(len(nums)//2):
        isSorted = True
        for j in xrange(i, len(nums) - i - 1):
            if nums[j] > nums[j+1]:
                nums[j], nums[j+1] = nums[j+1], nums[j]
                isSorted = False
        if isSorted:
            break
        isSorted = True
        for j in xrange(len(nums) - i - 1, i, -1):
            if nums[j] < nums[j-1]:
                nums[j], nums[j-1] = nums[j-1], nums[j]
                isSorted = False
        if isSorted:
            break

# 选择排序，在O(n^2)的算法里性能最好
def selectionSort(nums):
    for i in xrange(len(nums) - 1, 0, -1):
        max_idx = 0
        for j in xrange(i + 1):
            if nums[j] > nums[max_idx]:
                max_idx = j
        nums[max_idx], nums[i] = nums[i], nums[max_idx]

# 希尔排序，插入排序的变种，但性能比插入排序高得多
def shellSort(nums):
    gap = len(nums) // 2
    while gap:
        for i in xrange(gap, len(nums)):
            j = i
            temp = nums[j]
            while j - gap >= 0 and temp < nums[j-gap]:
                nums[j] = nums[j-gap]
                j -= gap
            nums[j] = temp 
        gap //= 2

# 快速排序递归版，在O(nlogn)的算法里性能最好
def quickSort1(nums):
    # 双边循环法
    def partition1(nums, left, right):
        pivot = nums[left]
        l, r = left, right
        while l < r:
            while l < r and nums[r] >= pivot:
                r -= 1
            while l < r and nums[l] <= pivot:
                l += 1
            if l < r:
                nums[l], nums[r] = nums[r], nums[l]
        nums[left] = nums[l]
        nums[l] = pivot
        return l

    #单边循环法
    def partition2(nums, left, right):
        pivot = nums[left]
        mark = i = left
        while i <= right:
            if nums[i] < pivot:
                mark += 1
                nums[mark], nums[i] = nums[i], nums[mark]
            i += 1
        nums[left] = nums[mark]
        nums[mark] = pivot
        return mark

    def quickSort(nums, left, right):
        if left >= right:
            return
        #pivotIndex = partition1(nums, left, right)
        pivotIndex = partition2(nums, left, right)
        quickSort(nums, left, pivotIndex - 1)
        quickSort(nums, pivotIndex + 1, right)

    quickSort(nums, 0, len(nums) - 1)

# 快速排序迭代版，跟递归版性能差不多
def quickSort2(nums):
    def partition2(nums, left, right):
        pivot = nums[left]
        mark = i = left
        while i <= right:
            if nums[i] < pivot:
                mark += 1
                nums[mark], nums[i] = nums[i], nums[mark]
            i += 1
        nums[left] = nums[mark]
        nums[mark] = pivot
        return mark

    def quickSort(nums, left, right):
        stack = []
        stack.append((left, right))
        while stack:
            l, r = stack.pop()
            pivotIndex = partition2(nums, l, r)
            if l < pivotIndex - 1:
                stack.append((l, pivotIndex - 1))
            if r > pivotIndex + 1:
                stack.append((pivotIndex + 1, r))
    
    quickSort(nums, 0, len(nums) - 1)

# 堆排序
def heapSort(nums):
    def downAdjust(nums, i, n):
        temp = nums[i]
        j = 2 * i + 1
        while j < n:
            if j + 1 < n and nums[j+1] > nums[j]:
                j += 1
            if temp >= nums[j]:
                break
            nums[i] = nums[j] 
            i = j
            j = 2 * j + 1
        nums[i] = temp  
    
    for i in xrange(len(nums) // 2 - 1, -1, -1):
        downAdjust(nums, i, len(nums))
    for i in xrange(len(nums) - 1, 0, -1):
        temp = nums[i]
        nums[i] = nums[0]
        nums[0] = temp
        downAdjust(nums, 0, i)

# 归并排序递归版
def mergeSort1(nums):
    def merge(nums, left, mid, right):
        L = nums[left:mid+1]
        R = nums[mid+1:right+1]
        i, j, k = 0, 0, left
        while i < len(L) and j < len(R):
            if L[i] < R[j]:
                nums[k] = L[i]
                k, i = k + 1, i + 1
            else:
                nums[k] = R[j]
                k, j = k + 1, j + 1
        while i < len(L):
            nums[k] = L[i]
            k, i = k + 1, i + 1
        while j < len(R):
            nums[k] = R[j]
            k, j = k + 1, j + 1

    def mergeSort(nums, left, right):
        if left >= right:
            return
        mid = (left + right) // 2
        mergeSort(nums, left, mid)
        mergeSort(nums, mid + 1, right)
        merge(nums, left, mid, right)
    
    mergeSort(nums, 0, len(nums) - 1)

# 归并排序递归版，用切片代替边界参数
def mergeSort2(nums):
    if len(nums) <= 1:
        return
    mid = len(nums) // 2
    L = nums[:mid]
    R = nums[mid:]
    mergeSort2(L)
    mergeSort2(R)
    i = j = k = 0
    while i < len(L) and j < len(R):
        if L[i] < R[j]:
            nums[k] = L[i]
            k, i = k + 1, i + 1
        else:
            nums[k] = R[j]
            k, j = k + 1, j + 1
    while i < len(L):
        nums[k] = L[i]
        k, i = k + 1, i + 1
    while j < len(R):
        nums[k] = R[j]
        k, j = k + 1, j + 1

# 归并排序迭代版，跟递归版性能差不多
def mergeSort3(nums):
    size = 1
    while size < len(nums):
        left = 0
        while left < len(nums):
            mid = left + size - 1
            if mid >= len(nums) - 1:
                break
            right = len(nums) - 1 if mid + size >= len(nums) else mid + size
            L = nums[left:mid+1]
            R = nums[mid+1:right+1]
            i, j, k = 0, 0, left
            while i < len(L) and j < len(R):
                if L[i] < R[j]:
                    nums[k] = L[i]
                    k, i = k + 1, i + 1
                else:
                    nums[k] = R[j]
                    k, j = k + 1, j + 1
            while i < len(L):
                nums[k] = L[i]
                k, i = k + 1, i + 1
            while j < len(R):
                nums[k] = R[j]
                k, j = k + 1, j + 1
            left += 2 * size
        size *= 2

# 计数排序，时间复杂度为O(n + m)，其中 m 为数组中最大值与最小值之差，由于 m 可能远大于 n ，导致性能可能很低
def countSort(nums):
    _min, _max = min(nums), max(nums)
    count = [0] * (_max - _min + 1)
    for num in nums:
        count[num-_min] += 1
    for i in xrange(1, len(count)):
        count[i] += count[i-1]
    temp = nums[::-1]
    for num in temp:
        nums[count[num-_min]-1] = num
        count[num -_min] -= 1

# 插入排序
def insertionSort(nums):
    for i in xrange(1, len(nums)):
        temp = nums[i]
        j = i - 1
        while j >= 0 and temp < nums[j]:
            nums[j+1] = nums[j]
            j -= 1
        nums[j+1] = temp

# 桶排序，在O(n)的算法里性能最好
def bucketSort(nums): 
    size = max(nums) // len(nums)
    buckets = [[] for _ in xrange(len(nums))]
    for i in xrange(len(nums)):
        j = nums[i] // size
        if j < len(nums):
            buckets[j].append(nums[i])
        else:
            buckets[len(nums)-1].append(nums[i])
    for i in xrange(len(nums)):
        insertionSort(buckets[i])
    index = 0
    for i in xrange(len(buckets)):
        for j in xrange(len(buckets[i])):
            nums[index] = buckets[i][j]
            index += 1

# 基数排序，正负数同时出现就不能用了
def radixSort(nums):
    def radixSort(nums, digits):
        for i in xrange(digits):
            count = [0] * 10
            for num in nums:
                d = (num // (10 ** i)) % 10
                count[d] += 1
            for j in xrange(1, 10):
                count[j] += count[j-1]
            temp = nums[::-1]
            for num in temp:
                d = (num // (10 ** i)) % 10
                nums[count[d] - 1] = num
                count[d] -= 1 

    digits = 0
    for num in nums:
        digits = max(digits, len(str(num)))
    radixSort(nums, digits)

# 测试排序算法的正确性
def test():
    T, n = 10000, 20
    accepted = True
    for _ in xrange(T):
        nums = [random.randint(1, n * n) for i in xrange(n)]
        ans = sorted(nums)
        arr = nums[:]
        quickSort2(arr)
        for i in xrange(n):
            if arr[i] != ans[i]:
                print 'Wrong answer'
                print nums
                print arr
                print ans
                accepted = False
                break
        if not accepted:
            break
    if accepted:
        print 'Pass'

if __name__ == '__main__':
    test()

    #sizes = [100, 200, 500, 1000, 2000, 5000, 10000]
    #labels = ['bubbleSort1', 'bubbleSort2', 'bubbleSort3', 'cocktailSort', 'insertionSort', 'selectionSort', 'shellSort']
    #funcs = [bubbleSort1, bubbleSort2, bubbleSort3, cocktailSort, insertionSort, selectionSort, shellSort]

    #sizes = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000,500000, 1000000]
    #labels = ['quickSort','heapSort', 'mergeSort1', 'mergeSort2']
    #funcs = [quickSort, heapSort, mergeSort1, mergeSort2]

    #sizes = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000,500000, 1000000]
    #labels = ['bucketSort', 'radixSort']
    #funcs = [bucketSort, radixSort]

    #sizes = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
    #labels = ['quickSort1', 'quickSort2', 'mergeSort1', 'mergeSort2', 'mergeSort3']
    #funcs = [quickSort1, quickSort2, mergeSort1, mergeSort2, mergeSort3]

    sizes = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000,500000, 1000000]
    labels = ['defaultSort', 'quickSort','heapSort', 'mergeSort2', 'bucketSort']
    funcs = [sorted, quickSort, heapSort, mergeSort2, bucketSort]

    T = [[] for _ in xrange(len(labels))]
    N = []
    for n in sizes:
        N.append(n)
        nums = [random.randint(1, n * n) for _ in xrange(n)]
        #nums = [n - i for i in xrange(n)]
        #nums = [i for i in xrange(n)]
        #nums = [10 - i for i in xrange(10)] + [10 + i for i in xrange(1, n - 9)]
        sorted_nums = sorted(nums)
        for i in xrange(len(labels)):
            temp = nums[:]
            start = time.clock()
            funcs[i](temp)
            end = time.clock()
            print '{0}, n = {1}, time = {2}'.format(labels[i], n, end - start)
            T[i].append(end - start)
        print '\n'

    N_smooth = np.linspace(min(N), max(N), 1000)
    for i in xrange(len(labels)):
        T_smooth = make_interp_spline(N, T[i])(N_smooth)
        plt.plot(N_smooth, T_smooth, label=labels[i])
    plt.xlabel('Size')
    plt.ylabel('Time')
    plt.legend()
    #plt.savefig('./sort_comparison3.jpg')
    plt.show()
    
    
    

                      
    