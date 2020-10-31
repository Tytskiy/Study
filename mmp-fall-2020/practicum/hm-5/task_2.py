def linearize(arr):
    if(not hasattr(arr, "__iter__")):
        raise TypeError
    linear = []
    for elem in arr:
        if(hasattr(elem, "__iter__")):
            tmp = linearize(elem)
            linear.extend(tmp)
        else:
            linear.append(elem)
    return linear
