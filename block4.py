import time
# Create a list and a dictionary with 10^6 elements
lst = list(range(10**7))
dct = {i: None for i in range(10**7)}
# Search for an element that doesn't exist
search_element = 10**7 + 1
# Measure the time taken to search in the list
start_time = time.time()
print(search_element in lst)
end_time = time.time()
print("Time taken to search in list: %d seconds" % (end_time - start_time))
# Measure the time taken to search in the dictionary
start_time = time.time()
print(search_element in dct)
end_time = time.time()
print("Time taken to search in dict: %d seconds" % (end_time - start_time))


class Stack:
    def __init__(self):
        self.stack = []

    def push(self, item):
        self.stack.append(item)

    def pop(self):
        if len(self.stack) < 1:
            return None
        return self.stack.pop()

    def is_empty(self):
        return len(self.stack) == 0


# def test_stack():
#     print("Running tests")
#     stack = Stack()
#     stack.push(1)
#     stack.push(2)
#     stack.push(3)
#     assert (stack.is_empty() == False)
#     assert (stack.pop() == 3)
#     assert (stack.pop() == 2)
#     assert (stack.pop() == 1)
#     assert (stack.is_empty() == True)
#     print("Tests complete")
#
#
# test_stack()
#
# my_string = "BANANA"
# stack = Stack()
# for letter in my_string:
#     stack.push(letter)
#
# reversed_string = ''
# while not stack.is_empty():
#     reversed_string += stack.pop()
# print(my_string)
# print(reversed_string)


def is_balanced(string):
    L = Stack()
    s = {']': '[', '}': '{', ')': '('}

    for i in string:
        if i in ['[', '{', '(']:
            L.push(i)
        elif i in [']', '}', ')']:
            if L.is_empty() or L.pop() == s[i]:
                return True

    return L.is_empty()




def test_is_balanced():
    assert is_balanced("(){}[]") == True, "Test case 1 failed"
    assert is_balanced("({[]})") == True, "Test case 2 failed"
    assert is_balanced("({[})") == False, "Test case 3 failed"
    assert is_balanced("({[}") == False, "Test case 4 failed"
    assert is_balanced("") == True, "Test case 5 failed"
    assert is_balanced("({[hello]})") == True, "Test case 6 failed"
    assert is_balanced("[x**2 for x in range(10)]") == True, "Test case 7 failed"
    assert is_balanced("for i in range(10):\n\tprint(i)") == True, "Test case 8 failed"
    print("All test cases passed")


test_is_balanced()
