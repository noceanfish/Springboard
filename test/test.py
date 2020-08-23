def merge(l, r):
    i = 0
    j = 0
    index = 0
    res = []


    return res


def merge_sort(vol_array):
    length = len(vol_array)

    if length < 2:
        return vol_array

    index = length // 2

    left = vol_array[:index]
    right = vol_array[index:]

    merge_sort(left)
    merge_sort(right)

    i = 0
    j = 0
    index = 0

    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            vol_array[index] = left[i]
            i = i + 1
            index = index + 1
        else:
            vol_array[index] = right[j]
            j += 1
            index += 1
    while i < len(left):
        vol_array[index] = left[i]
        index += 1
        i += 1
    while j < len(right):
        vol_array[index] = right[j]
        index += 1
        j += 1
    return vol_array


if __name__ == "__main__":
    year_vol = [34, 3, 64, 54, 23, 21, 98, 203, 1, -10]
    merge_sort(year_vol)
    print(merge_sort)

    n = 5
    res = [None] * n
    print(res)

