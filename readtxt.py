with open('sentences.txt', 'r') as file:
    for line in file:
        print(line.strip())  # .strip() removes trailing newlines
