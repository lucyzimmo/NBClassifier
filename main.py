import csv

#establish variables
train = 'file_to_train.csv'
testing = 'file_to_test.csv'
numCol = 3
numToExclude = 1
y_0_p = 1
y_1_p = 1


#use csv library to import csv file into a list of lists of values
def import_data(filename):
    dataset = []
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
            row_list =(',').join(row)
            row_list = list(row_list.replace(',', ''))
            dataset.append(row_list)
        del dataset[0]
    return (dataset)


#function that reads in the data and sorts into readable columns with
#each respective count
def buildCount(dataset):
    #create lists to feed in data
    y_0_with_x1 = [0] * (numCol - numToExclude)
    y_1_with_x1 = [0] * (numCol - numToExclude)
    y_0_with_x0 = [0] * (numCol - numToExclude)
    y_1_with_x0 = [0] * (numCol - numToExclude)

    #count up the data
    for i in range(len(dataset)):
        currRow = dataset[i]
        x = 0
        #if y value is 0
        if int(currRow[-1]) == 0:
            for col in range(numCol - numToExclude):
                if (int(currRow[col]) == 0):
                    y_0_with_x0[col] += 1
                else:
                    y_0_with_x1[col] += 1
                    
        #if y value is 1
        else:
            for col in range(numCol - numToExclude):
                if (int(currRow[col]) == 0):
                    y_1_with_x0[col] += 1
                else:
                    y_1_with_x1[col] += 1
    #create lists for y = 0 and y = 1 so we can access it through nested lists            
    y_0_list = [y_0_with_x1, y_0_with_x0]
    y_1_list = [y_1_with_x1, y_1_with_x0]
    final = [y_0_list, y_1_list]
    return(final)

#count the y = 0 values and y = 1 values
def y_count(dataset):
    y_0_count = 0
    y_1_count = 0
    for i in range(len(dataset)):
        currRow = dataset[i]
        if int(currRow[-1]) == 0:
            y_0_count += 1
        else:
            y_1_count += 1
    return[y_0_count, y_1_count]

#runs through testing file and predicts y = ? using calculated probabilities
def test(dataset, trainedData):
    accurate_count = 0
    trials = 0
    print(trainedData)
    for i in range(len(dataset)):
        trials += 1
        currRow = dataset[i]
        p_for_y_0 = 1
        p_for_y_1 = 1
        for col in range(numCol - numToExclude):
            #calc for x = 0
            if (int(currRow[col]) == 0):
                #calc for y = 0
                p_for_y_0 *= trainedData[0][0][col]
                #calc for y = 1
                p_for_y_1 *= trainedData[1][0][col]
                
            #calc for x = 1
            else:
                #calc for y = 0
                p_for_y_0 *= trainedData[0][1][col]
                #calc for y = 1
                p_for_y_1 *= trainedData[1][1][col]

        #multiply through the final factor
        p_for_y_0 *= y_0_p
        p_for_y_1 *= y_1_p

        #argmax the prediction
        if (p_for_y_0 >= p_for_y_1):
            prediction = 0
        else:
            prediction = 1
        if (prediction == int(currRow[-1])):
            accurate_count += 1
            
    return(accurate_count/trials)

#calculate each probability from the list of data using laplace smoothing
def calc_each_prob_x(listData, y_0_count, y_1_count):
    print(listData)
    list_of_prob = [[[],[]],[[],[]]]
    for x in range(len(listData)):
        #p for y = 0, x = 0
        list_of_prob[0][0].append(float(listData[0][0][x] + 1) /(y_0_count + 2))
        #p for y = 1, x = 0
        list_of_prob[1][0].append(float(listData[1][0][x] + 1) /(y_1_count + 2))
        #p for y = 0, x = 1
        list_of_prob[0][1].append(float(listData[0][1][x] + 1) /(y_0_count + 2))
        #p for y = 1, x = 1
        list_of_prob[1][1].append(float(listData[1][1][x] + 1) /(y_1_count + 2))
    return(list_of_prob)


def main():
    #training
    dataset = import_data(train)
    listData = buildCount(dataset)
    y_count_0 = y_count(dataset)[0]
    y_count_1 = y_count(dataset)[1]
    y_0_p = (y_count_0+1)/(y_count_0+y_count_1+2)
    print("y_0: " + str(y_0_p))
    y_1_p = (y_count_1+1)/(y_count_0+y_count_1+2)
    print("y_1: " + str(y_1_p))
    trainedDataSet = calc_each_prob_x(listData, y_count_0, y_count_1)
    print("x = 1 | Y = 0 and x = 1 | Y = 1 " + str(trainedDataSet))

    #testing
    test_dataset = import_data(testing)
    print(test(test_dataset, trainedDataSet))
    


if __name__ == "__main__":
    main()

