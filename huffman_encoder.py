##Author: Mohammed Kashwah - 500 724 449


import numpy as np
import sys
import matplotlib.pyplot as plt
import collections as clct
import math as mth
import cv2

###########################
####### output of numpy does not get truncated
# np.set_printoptions(threshold = sys.maxsize)


############Classes for Huffman Encoding###########
class Node:
    def __init__(self, symbol = None, weight = None):
        self.symbol = symbol
        self.weight = weight
        self.parent = None
        self.lchild = None
        self.rchild = None
        self.smallest = None
        self.leaf_node = 1
        self.code_word = ""

########Functions#######
def occurnaces(x):
    a = clct.Counter(x)
    return a

def myEntropy(x):
    a = clct.Counter(x)
    ttl = 0
    for x in a: #will count total number of elemnts in x
        ttl = ttl + a[x]

    entrp_props = {}
    for x in a:
        entrp_props[x] = a[x] / ttl
    return entrp_props

def calcEntropy(probabilites_dictionary):
    H_s = 0
    for x in probabilites_dictionary:
        H_s = H_s + (probabilites_dictionary[x] * mth.log2(1 / probabilites_dictionary[x]))
    return H_s




def smokeTrees(nodes_list):
    N1 , N2 = None, None
    i = 0
    weight_list = []


    #making list of weights
    for k in range (len(nodes_list)-1):
        for node in nodes_list:
            if node.smallest == None:
                weight_list.append(node.weight)
                i = i+1
            elif node.smallest == 1:
                weight_list.append(float('inf'))

        #finding the smallest weight's index
        indx1 = weight_list.index(min(weight_list))
        #set smallest to 1
        nodes_list[indx1].smallest = 1
        N1 = nodes_list[indx1]
        # print('size of the first weight list', len(weight_list))
        # print('index of the smallest', indx1 )
        #search for the next smallest
        weight_list.clear()
        i = 0
        for node in nodes_list:
            if node.smallest == None:
                weight_list.append(node.weight)
                i = i+1
            elif node.smallest == 1:
                weight_list.append(float('inf'))

        indx2 = weight_list.index(min(weight_list))
        nodes_list[indx2].smallest = 1
        N2 = nodes_list[indx2]
        # print('size of the second weight list', len(weight_list))
        # print('index of the smallest', indx2 )

        #set parent node for those two nodes
        parent_node = Node(None, N1.weight + N2.weight)
        parent_node.lchild = N1
        parent_node.rchild = N2
        parent_node.leaf_node = 0
        parent_node.parent = Node(None, None)
        # parent_node.parent.lchild = N1
        # parent_node.parent.rchild = N2

        N1.parent = parent_node
        N2.parent = parent_node
        nodes_list.append(parent_node)
        weight_list.clear()
        # print(nodes_list.size)
        # for node in nodes_list:
        #     if nodes_list[node].smallest is not None:
    print("Huffman Tree was built")
    return nodes_list

#Traversing and assigning code word
def makingCodeWord(rootNode):
    if rootNode is None:
        return

    # create an empty stack and push root to it
    nodeStack = []
    nodeStack.append(rootNode)
    #  Pop all items one by one. Do following for every popped item
    #   a) print it
    #   b) push its right child
    #   c) push its left child
    # Note that right child is pushed first so that left
    # is processed first */
    while(len(nodeStack) > 0):

        # Pop the top item from stack and print it
        node = nodeStack.pop()

        # Push right and left children of the popped node
        # to stack
        if node.rchild is not None:
            nodeStack.append(node.rchild)
            node.rchild.code_word = node.code_word + "1"
        if node.lchild is not None:
            nodeStack.append(node.lchild)
            node.lchild.code_word = node.code_word + "0"
    print("code words were made")

#This function is encoder function
#Returns an encoded string in binrary: code_word = "010101010101..."
def encode(array, tree):
    code_word = ""
    for a in array:
        for node in tree:
            if a == node.symbol:
                code_word = code_word + node.code_word
    return code_word

##this function is used for decoding
def decode(rootNode, encoded_string):
    decoded_string = ""
    node = rootNode

    for i in range(len(encoded_string)):
        if node.lchild.leaf_node != 1 and encoded_string[i] == "0":
            node = node.lchild
        elif node.rchild.leaf_node != 1 and encoded_string[i] == "1":
            node = node.rchild
        elif node.lchild.leaf_node == 1 and encoded_string[i] == "0":
            decoded_string = decoded_string + node.lchild.symbol
            node = rootNode
        elif node.rchild.leaf_node == 1 and encoded_string[i] == "1":
            decoded_string = decoded_string + node.rchild.symbol
            node = rootNode




    # for n in encoded_string:
    #     if node.leaf_node != 1 and n == "0":
    #         node = node.lchild
    #     elif node.leaf_node != 1 and n == "1":
    #         node = node.rchild
    #
    #     decoded_string = decoded_string + node.symbol
    #     node = rootNode


    return decoded_string

#this function build the leaf nodes and retruns them in a list
#Returns list[] of nodes where list[i] = Node(symbol, weight)
def nodes_init(occurances_dictionary):
    nodes_list = []
    i = 0
    for symbol, occurnaces in occurances_dictionary.items():
        nodes_list.append(Node(str(symbol), occurnaces))
        i = i+1
    return nodes_list

#This function prints nodes list
def print_node_list(nodes_list):
    for node in nodes_list:
        print(node.symbol, node.weight)

def print_tree(smokedTree):
    print("symbol, weight, parent, lchild, rchild, code word")
    for i in range(len(smokedTree)):
        if smokedTree[i].leaf_node == 1:
            print(smokedTree[i].symbol, smokedTree[i].weight, smokedTree[i].parent.weight, "No Left Child", "No Right Child", smokedTree[i].code_word)
        elif smokedTree[i].leaf_node == 0:
            print(smokedTree[i].symbol, smokedTree[i].weight, smokedTree[i].parent.weight, smokedTree[i].lchild.weight, smokedTree[i].rchild.weight, smokedTree[i].code_word)

#################
###### Testing#########
test_array = 'HUFFMAN IS THE BEST COMPRESSION ALGORITHM'

# print(occurnaces(test_array))
# print(myEntropy(test_array))

a = myEntropy(test_array)
H_s = calcEntropy(a)
print('Entropy of the test statment =', H_s)

##########importing the image
img_imprt = cv2.imread('TheCorona.jpg')

############different image colors
img_clrd_BGR = np.copy(img_imprt)
img_clrd_RGB = cv2.cvtColor(img_clrd_BGR, cv2.COLOR_BGR2RGB)
img_clrd_YCrCb = cv2.cvtColor(img_clrd_RGB, cv2.COLOR_RGB2YCrCb)
img_gray = cv2.cvtColor(img_clrd_BGR, cv2.COLOR_BGR2GRAY)


#Separating the components of the RGB picture
img_clrd_YCrCb_Y = img_clrd_YCrCb[:,:,0]
img_clrd_YCrCb_Cr = img_clrd_YCrCb[:,:,1]
img_clrd_YCrCb_Cb = img_clrd_YCrCb[:,:,2]

#Reshaping the lenght x length matrix into 2 x length vector
img_clrd_YCrCb_Y_reshaped = img_clrd_YCrCb_Y.reshape(img_clrd_YCrCb_Y.size)
img_clrd_YCrCb_Cr_reshaped = img_clrd_YCrCb_Cr.reshape(img_clrd_YCrCb_Cr.size)
img_clrd_YCrCb_Cb_reshaped = img_clrd_YCrCb_Cb.reshape(img_clrd_YCrCb_Cb.size)
img_gray_reshaped = img_gray.reshape(img_gray.size)

nodes_list = nodes_init(occurnaces(test_array))
nodes_list2 = nodes_init(occurnaces(img_gray_reshaped))

# print_node_list(nodes_list)
print_node_list(nodes_list)

####testing encoder and decoder
tree = smokeTrees(nodes_list)
makingCodeWord(tree[len(tree)-1])
print_tree(tree)

encoded_string = encode(test_array, tree)
# encoded_string = encode(test_array, tree)
print("Encoded string is :\n")
print(encoded_string)

# encoded_string = "101010101010"

##decode(rootNode, encoded_string )
decoded_string = decode(tree[len(tree)-1], encoded_string)
print("decoded string is :\n")
print(decoded_string)





# print_tree(tree)







# ##uncomment this part for demo
#open txt file
f = open('output.txt', 'w')

#Calculating Entropy for each channel


y_entropy = calcEntropy(myEntropy(img_clrd_YCrCb_Y_reshaped))
print('Entropy of Y channel =', y_entropy, file = f)
cr_entropy = calcEntropy(myEntropy(img_clrd_YCrCb_Cb_reshaped))
print('Entropy of Cr channel =', cr_entropy, file = f)
cb_entropy = calcEntropy(myEntropy(img_clrd_YCrCb_Cr_reshaped))
print('Entropy of Cb channel =', cb_entropy, file = f)
gray_entropy = calcEntropy(myEntropy(img_gray_reshaped))
print('Entropy of the gray image =', gray_entropy, file = f)

print('gray image array', img_gray_reshaped, file = f)
print(list(clct.Counter(img_gray_reshaped))[3])
print('gray image array', clct.Counter(img_gray_reshaped), file = f)


#closing the file
f.close()
# #### Demo ends here part 1
###############################################################################3
###############################################################################

##Huffman implementation






#testing tings
# c = np.array([[0, 111], [2,3], [3,4]])
# d = c.reshape(6)
# d = myEntropy(d)
# print(d)
# print(img_clrd_YCrCb_Y_reshaped)


##############Figures

plt.figure(1)
plt.imshow(img_clrd_BGR)
cv2.imshow('BGR image', img_clrd_BGR)

plt.title('BGR image')

plt.figure(2)
plt.imshow(img_clrd_RGB)
cv2.imshow('RGB image', img_clrd_RGB)
plt.title('RGB image')

plt.figure(3)
plt.imshow(img_clrd_YCrCb)
plt.title('YCrCb image')
cv2.imshow('YCrCb image', img_clrd_YCrCb)

plt.figure(4)
plt.imshow(img_gray)
plt.title('Gray image')
plt.show()
cv2.imshow('Grayscale image', img_gray)
cv2.waitKey(0)
print(a.get('S'))
