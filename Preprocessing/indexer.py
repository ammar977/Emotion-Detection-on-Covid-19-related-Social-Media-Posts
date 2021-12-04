# Jena Jordahl Final Project Preprocessing
from sys import argv
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

def value_typed(s,debug):
    if s == '?':
        v = float("NaN")
    else:
        try:
            v = int(s)
        except ValueError:
            try:
                v = float(s)
            except ValueError:
                v = s
    return v


def columns_listed(strings, debug):
    column_count = 0
    X = []
    y = []
    for s in strings:
        column_count += 1
        vt = value_typed(s, debug)
        if column_count <= len(strings) - 2: X.append(vt)
        else: y.append(vt)
    inverted_index(X[0], X[1], y[0])
    inverted_index_letter(X[0], X[1], y[0])
    return column_count, y, X


def import_data(filename,debug):
    X = []
    y = []
    row_count = 0
    with open(filename, 'r') as f:
        lines = f.readlines()
        # Separate the values from the CVS to list
        for line in lines:
            row_count += 1
            strings = line.split("\t")
            col_cnt, val, col_list = columns_listed(strings,debug)
            # end of line so append the row data
            y.append(val)
            X.append(col_list)
    print("input data cols: " + str(col_cnt) + ", rows: " +  str(row_count))
    return X, y


def write_word(id, ind, word,y):
    with open('./worddb/'+ word + ".txt", 'a') as file1:
        ref = str(id) + ',' + str(ind) + ',' + str(y) + '\n'
        file1.write(ref)
    file1.close
    ind += 1
    return ind

def inverted_index(id,text,y):
    # Appending to file
    ind = 0
    word = ''
    for letter in text:
        if not letter in ['/',' ','.','!']:
            word += letter
        if letter in [' ','.','!']:
            if letter in ['.','!'] and len(word)>1:
                ind = write_word(id,ind,word,y)
                word = letter
            if word != '':
                if word == '':
                    word = letter
                ind = write_word(id,ind,word,y)
            word = ''


def inverted_index_letter(id, text, y):
    # Appending to file
    ind = 0
    for letter in text:
        if not letter in ['/']:
            with open('./letterdb/' + letter + ".txt", 'a') as file1:
                ref = str(id) + ',' + str(ind) + ',' + str(y) + '\n'
                file1.write(ref)
            ind += 1

def print_hi(name,argv):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.
    debug = False
    infile=argv[1]
    # infile='../Data Twitter/Train/joy-ratings-0to1.train.txt'
    X, y = import_data(infile, debug)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm',argv)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/