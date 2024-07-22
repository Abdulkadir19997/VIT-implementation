import os

def print_tree(startpath, prefix=""):
    for item in os.listdir(startpath):
        path = os.path.join(startpath, item)
        if os.path.isdir(path):
            print(prefix + "|-- " + item)
            print_tree(path, prefix + "|   ")
        else:
            print(prefix + "|-- " + item)

# Replace 'your_directory_path' with the path to your project directory
your_directory_path = 'C:\projects\Github_projects\Vision_transformers_implementation'
print(your_directory_path)
print_tree(your_directory_path)