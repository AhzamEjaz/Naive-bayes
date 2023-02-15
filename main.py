from NaiveBayes import NaiveBayes
from fileSelection import fileSelect
# Getting training and testing data

def changeTestData(naiveBayesObj, test_df):
    naiveBayesObj.setTestingData(test_df)
    return naiveBayesObj

def changeTrainData(naiveBayesObj, train_df):
    naiveBayesObj.setTrainingData(train_df)
    return naiveBayesObj

def printAccuracy(acc):
    print('|--------------------------------------| \n|                                      | \n|                                      |')
    print('|Accuracy of model: ', acc)
    print('|                                      | \n|                                      | \n|--------------------------------------|\n')


fs = fileSelect()
train_df, test_df = fs.importFiles()

# Start model createion
print("Program Running")
nv = NaiveBayes(train_df = train_df, test_df = test_df)
nv.train()
printAccuracy(nv.test())

# State managment
state_variable = '0'
while(state_variable != 'q'):
    print('Select from the following options: \n Press 1 to select new test dataset \n press 2 to select new training dataset \n Press 3 to test on existing dataset \n Press q to to quit Program \n')
    state_variable = input()
    if(state_variable == '1'):
        test_df = fs.importTestFile()
        changeTestData(nv, test_df)
        printAccuracy(nv.test())

    elif(state_variable == '2'):
        train_df = fs.importTrainFile()
        changeTrainData(nv, train_df)
        nv.train()
    elif(state_variable == '3'):
        printAccuracy(nv.test())

print("Program Ended")


