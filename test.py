def test(arr):
    occur = []
    visited = []
    for i in arr:
        n = 0
        for j in arr:
            if i == j and i not in visited:
                n += 1
        visited.append(i)
        if n != 0:
            occur.append(n)

    seen = set()
    for num in occur:
        if num in seen:
            return False  # Return False if a duplicate is found
        seen.add(num)
    return True  # Return True if no duplicates were found


test([1,2,2,1,1,3])