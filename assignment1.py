'''
CMPUT 175 Final Assignment 
Author: Arron Roasa
Created: 03/31/2025
'''
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import carousel

#   PART 5.1
def read_data(data):
    '''
    Reads csv file
    Returns a list
    data: file to be read, type .csv
    '''
    file_list = []
    with open(data, 'r') as file:
        for lines in file:
            rows = lines.strip().split(',')
            file_list.append(rows)
    return file_list

def filter_data(data):
    '''
    Filters and removes rows that are missing values
    Returns a list
    data: data to be filtered, type list
    '''
    header = data[0]
    print(f'Initial number of rows: {len(data[1:])}')
    
    num_cols = len(header)
    missing_elements = [[] for _ in range(num_cols)]
    
    missing_vals = {'', 'nan', 'null', 'none'}
    for row in data[1:]:
        for index, value in enumerate(row):
            if str(value).strip().lower() in missing_vals:
                missing_elements[index].append(value)
    
    filtered_data = [
        row for row in data[1:] if all(value.strip() not in missing_vals for value in row)
    ]

    letters = ['a','b','c','d','e','f','g','h','i','j','k','l']
    for i in range(len(header)):
        if (len(missing_elements[i])) > 0:
            print(f'Column {letters[i]}: {len(missing_elements[i])} values missing')
    print(f'Remaining number of rows: {len(filtered_data)}')

    return filtered_data

def filter_90(data):
    '''
    Filters rows with ages greater than 90
    Returns a list
    data: data to be filtered, type list
    '''
    over_90 = []
    for row in data:
        if int(row[0]) > 90:
            over_90.append(row[0])

    filtered_data = [
        row for row in data if all (value.strip() not in over_90 for value in row)
    ]

    print(f'Number of records with age > 90: {len(over_90)}')
    print(f'Remaining number of rows: {len(filtered_data)}')
    return filtered_data

def analyze_by_age(data):
    '''
    Analyzes the loan distribution by age and plots the data on a histogram
    data: data to be analyzed, type list
    '''
    in_default = []
    not_in_default = []
    
    #   Check default status
    for row in data:
        if int(row[8]) == 1:
            in_default.append(int(row[0]))
        elif int(row[8]) == 0:
            not_in_default.append(int(row[0]))
    
    #   Plot histograms
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].hist(in_default, bins=5, color='steelblue', edgecolor='black')      #   Set age buckets + other visual parameters
    axs[0].set_title('Loans in Default')      #   Set title
    axs[0].set_xlim(-5, 105)    #   Set x limit

    axs[1].hist(not_in_default, bins=5, color='coral', edgecolor='black')     #   Set age buckets + other visual parameters
    axs[1].set_title('Loans Not in Defaul')      #   Set title
    axs[1].set_xlim(-5, 105)    #   Set x limit

    for ax in axs:      #   Set labels
        ax.set_xlabel('Age')
        ax.set_ylabel('No. of Borrowers')

    plt.tight_layout()
    plt.show(block = False)

def analyze_default_rates(data):
    '''
    Analyzes the default rates among homeowners and plots the data on a pie chart
    data: data to be analyzed, type list
    '''
    own_house = []
    in_default = []
    not_in_default = []

    for row in data:
        if row[2].lower() == 'own':
            own_house.append(row)

    #   Checking default status
    for row in own_house:
        if int(row[8]) == 1:
            in_default.append(int(row[8]))
        elif int(row[8]) == 0:
            not_in_default.append(int(row[8]))
    
    labels = 'Not Defaulted', 'Defaulted'   #   Set labels
    sizes = [len(in_default), len(not_in_default)]      #   Set sizes
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, colors = ['coral', 'steelblue'], autopct='%1.1f%%')   #   Set parameters
    plt.title('Homeowners: Default vs Not Default')     #   Set title
    plt.show(block = False)

def balance_data(data):
    '''
    Checks for class imbalances and undersamples the data if imbalances are present
    Returns a list
    data: data to be checked, type list
    '''
    in_default = []
    not_in_default = []
    for row in data:
        if int(row[8]) == 1:    #   If default
            in_default.append(row)
        elif int(row[8]) == 0:  #   If not default
            not_in_default.append(row)

    #   Print out number of borrowers who did and did not default
    print(f'Number of borrowers who defaulted: {len(in_default)}')
    print(f'Number of borrowers who did not default: {len(not_in_default)}')

    #   Undersample data by popping the top n rows (in_default) from not_in_default, uncomment if you would like to test
    #while len(not_in_default) > len(in_default):
        #not_in_default.pop()

    #print(f'Number of borrowers who defaulted: {len(in_default)}')
    #print(f'Number of borrowers who did not default: {len(not_in_default)}')

    return in_default + not_in_default

#   PART 5.2
def scale_data(data, training, scaler = None):
    '''
    Scales data for better prediction accuracy
    Returns a nested list
    data: data to be scaled, type list
    training: checks if we are scaling the training data or a different dataset, type boolean
    scaler: StandardScaler that gets passed if we are using it again
    '''
    #   Data: [person_income, loan_amnt]
    unscaled_data = []
    
    for row in data:
        if (row[6]).isdigit():  #   credit_risk_train.csv
            unscaled_data.append([int(row[1]), int(row[6])])    #   Append appropriate indexes
        else:   #   loan_requests.csv
            unscaled_data.append([int(row[2]), int(row[7])])    #   Append appropriate indexes

    if training:    #   Scaling training data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(unscaled_data)
    else:           #   Scaling other data
        if scaler is None:
            raise ValueError("Scaler must be provided when training=False")
        scaled_data = scaler.transform(unscaled_data)
        
    return scaled_data, scaler  #   Additionally return scaler so it can be re-used when we pass it

def build_decision_tree(balanced_data, scaled_data, training, scaler):
    '''
    Builds a decision tree from the balanced and scaled data, trains it, and makes predictions
    Returns a list
    balanced_data: balanced data, aka unscaled data, type list
    scaled_data: scaled data, type list
    training: checks if we are using the testing data or the prediction data, type boolean
    '''
    #   Build training data and model
    x_train = []
    y_train = []

    #   Append credit history length to scaled_data list
    for i in range(len(scaled_data)):
        features = [float(val) for val in scaled_data[i]]
        label = int(balanced_data[i][11])
        x_train.append(features + [label])

    for i in range(len(balanced_data)):
        y_train.append(int(balanced_data[i][8]))    #       Append loan status

    clf = DecisionTreeClassifier(random_state=42)
    clf = DecisionTreeClassifier(random_state=42)  # Create the model
    clf.fit(x_train, y_train)   # Train the model with the data

    #   Build testing data
    x_test = []
    y_test = []
    if training:    #   Testing training data, PART 5.2
        #   Run previous methods to filter, balance, and scale the testing data
        test_data = read_data('credit_risk_test.csv')
        print(f'TESTING DATA')
        filter1_test = filter_data(test_data)
        print()
        filter2_test = filter_90(filter1_test)
        print()
        balanced_test = balance_data(filter2_test)
        print()
        scaled_test, _ = scale_data(balanced_test, False, scaler)

        #   Same append methods as before
        for i in range(len(scaled_test)):
            features = [float(val) for val in scaled_test[i]]
            label = int(balanced_test[i][11])
            x_test.append(features + [label])

        for i in range(len(balanced_test)):
            y_test.append(int(balanced_test[i][8]))

        # Get predictions from the trained model
        y_pred = clf.predict(x_test)

        # Print evaluation metrics
        print(f'\nTest Accuracy: {round(accuracy_score(y_test, y_pred), 2)}')  # Accuracy
        print(f'Accuracy Report: {classification_report(y_test, y_pred)}')  # Precision, Recall, F1-score
        print(f'\nConfusion Matrix:')
        print(confusion_matrix(y_test, y_pred))  # Confusion Matrix
        print()
    else:   #   Predicting loan status, PART 5.3
        #   Read loan data, filter, and scale
        prediction_data = read_data('loan_requests.csv')
        print(f'PREDICTION DATA')
        filter1_pred = filter_data(prediction_data)
        print()
        scaled_pred, _ = scale_data(filter1_pred, False, scaler)

        #   Same append method as before, minus y test as it's not needed
        for i in range(len(scaled_pred)):
            features = [float(val) for val in scaled_pred[i]]
            label = int(prediction_data[i+1][11])
            x_test.append(features + [label])

        #   Make predictions
        y_pred = clf.predict(x_test)
        y_pred = y_pred.tolist()
        print(f'Loan status predictions (0: not default, 1: default): {y_pred}')

        #   Append prediction to csv list representation
        header = prediction_data[0]
        header.append('prediction')

        for i, pred in enumerate(y_pred):
            prediction_data[i+1].append(str(pred))

        return prediction_data

def create_carousel(data):
    '''
    Creates a carousel from a dataset + includes navigation logic
    data: the data to be used
    '''
    v_carousel = carousel.Carousel()
    header = data[0]
    data_dict = [dict(zip(header, row)) for row in data[1:]]    #   Create dictionaries from dataset
    
    for i in data_dict:
        v_carousel.add(i)   #   Append each dictionary to the carousel
    
    #   Initialization
    running = True
    v_carousel.current = v_carousel.head
    data = v_carousel.getCurrentData()

    while running:
        print('--------------------------------------------------')
        print(f'Borrower: {data['borrower']}')
        print(f'Age: {data['person_age']}')
        print(f'Income: {data['person_income']}')
        print(f'Home Ownership: {data['person_home_ownership']}')
        print(f'Employment: {data['person_emp_length']}')
        print(f'Loan Intent: {data['loan_intent']}')
        print(f'Loan Grade: {data['loan_grade']}')
        print(f'Loan Amount: {data['loan_amnt']}')
        print(f'Interest Rate: {data['loan_int_rate']}')
        print(f'Loan Percent Income: {data['loan_percent_income']}')
        print(f'Historical Defaults: {data['cb_person_default_on_file']}')
        print(f'Credit History: {data['cb_person_cred_hist_length']}')
        print('--------------------------------------------------')
        if int(data['prediction']) == 0:    #   Not default predicted
            print(f'Predicted loan_status: Will not default')
            print('Recommend: Accept')
        elif int(data['prediction']) == 1:  #   Default predicted
            print(f'Predicted loan_status: Will default')
            print('Recommend: Reject')
        print('--------------------------------------------------')

        try:
            ask = input('Enter [1] to go to next, [2] to go back, or [0] to quit: ')

            if int(ask) == 1:   #   Next
                v_carousel.moveNext()
                data = v_carousel.getCurrentData()
            elif int(ask) == 2: #   Previous
                v_carousel.movePrevious()
                data = v_carousel.getCurrentData()
            elif int(ask) == 0: #   Exit
                print("Have a nice day!")
                break
        except:     #   Invalid input
            print('ERROR: Invalid Input')

def main():
    #   PART 5.1
    data = read_data('credit_risk_train.csv')
    print(f'TRAINING DATA')
    filter1 = filter_data(data)
    print()
    filter2 = filter_90(filter1)
    analyze_by_age(filter2)
    analyze_default_rates(filter2)
    print()
    balanced_data = balance_data(filter2)

    #   PART 5.2
    print()
    scaled_data, scaler = scale_data(balanced_data, True)
    build_decision_tree(balanced_data, scaled_data, True, scaler)

    #   PART 5.3
    prediction = build_decision_tree(balanced_data, scaled_data, False, scaler)
    print()

    input(f'Press Enter to continue...')

    #   INTERFACE
    create_carousel(prediction)

if __name__ == '__main__':
    main()