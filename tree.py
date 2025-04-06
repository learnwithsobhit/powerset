class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def serialize(root):
    def preorder(node):
        if not node:
            var.append("#")
        else:
            var.append(str(node.val))
            preorder(node.left)
            preorder(node.right)
    var = []
    preorder(root)
    return " ".join(var)

def deserialize(data):
    def preorder(var):
        node = next(var)
        if node == "#":
            return None
        else:
            temp_node = TreeNode(node)
            temp_node.left = preorder(var)
            temp_node.right = preorder(var)
            return temp_node
    var = iter(data.split())
    return preorder(var)

    
#write tests
def test():
    # Test case 1
    root = TreeNode(1)
    root.left = TreeNode(2)
    root.right = TreeNode(3)
    root.left.left = TreeNode(4)
    root.left.right = TreeNode(5)
    serialized = serialize(root)
    print("Serialized:", serialized)  # Expected output: "1 2 4 # # 5 # # 3 # #"
    deserialized = deserialize(serialized)
    assert serialized == serialize(deserialized)

    # Test case 2
    root = None
    serialized = serialize(root)
    print("Serialized:", serialized)  # Expected output: "#"
    deserialized = deserialize(serialized)
    assert serialized == serialize(deserialized)
    
    # Test case 3
    root = TreeNode(1)
    root.left = TreeNode(2)
    root.right = TreeNode(3)
    root.left.left = TreeNode(4)
    root.left.right = TreeNode(5)
    serialized = serialize(root)
    print("Serialized:", serialized)  # Expected output: "1 2 4 # # 5 # # 3 # #"
    deserialized = deserialize(serialized)
    assert serialized == serialize(deserialized)
    
    print("All tests passed!")
    
    



    



    
if __name__ == "__main__":
    test()
    
    



    



    