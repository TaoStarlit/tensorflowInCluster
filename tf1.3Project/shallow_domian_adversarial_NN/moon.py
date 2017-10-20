from DANN import DANN #from file name(module) import class (or method and value)

def main():

    # training data
    x_train = [[1, 2], [3, 4], [-1, -2], [-3, -4]]
    y_train = [[0, 1], [0, 1], [1, 0], [1, 0]]

    dann=DANN()
    dann.fit(x_train,y_train)
    dann.evaluate(x_train,y_train)

main()
