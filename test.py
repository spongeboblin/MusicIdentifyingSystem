from recognizer import SimpleRecognizer

if __name__ == '__main__':
    testQuery = [['123', 0.1], ['234', 1.1], ['345', 2.1]];
    testData = {};
    testData['123'] = [(1, 4.3), (1, 1.2), (2, 1.3)];
    testData['234'] = [(1, 1.3), (2, 2.3)];
    testData['345'] = [(1, 7.3), (2, 3.3)];

    reg = SimpleRecognizer();
    print reg.recognize(testQuery, testData);
