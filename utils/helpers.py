def get_pathes(path: str):
    f = open(file=path, mode='r')
    lines = [line.strip() for line in f.readlines()]
    return lines
