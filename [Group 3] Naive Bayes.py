# Created By: Group 3
# Members:
# Berse, Nikko
# Narvasa, Chyril 
# Waquez, Emerson Tamsi
# BSCS 4-2

##### Import need libraries ######
import pandas as pd
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

###### Import the needed file and drop NA values ######
adaptability = pd.read_csv ("students_adaptability_level_online_education-Cleaned.csv")
adaptability_df = pd.DataFrame(adaptability)
adaptability_df = adaptability_df.dropna()
print(adaptability_df)
print('\n=============================================================================\n')

###### Transform Qualitative data from the dataset to Quantitative with the use of LabelEncoder ######
number = LabelEncoder()
adaptability_transformed = adaptability_df
adaptability_transformed['Institution Type'] = number.fit_transform(adaptability_transformed['Institution Type'])         #government->0  non-government->1
adaptability_transformed['IT Student'] = number.fit_transform(adaptability_transformed['IT Student'])                     #no->0          yes->1
adaptability_transformed['Location'] = number.fit_transform(adaptability_transformed['Location'])                         #no->0          yes->1
adaptability_transformed['Load-shedding'] = number.fit_transform(adaptability_transformed['Load-shedding'])               #high->0        low->1
adaptability_transformed['Financial Condition'] = number.fit_transform(adaptability_transformed['Financial Condition'])   #mid->0         poor->1         rich->2
adaptability_transformed['Internet Type'] = number.fit_transform(adaptability_transformed['Internet Type'])               #mobile data->0 wifi->1
adaptability_transformed['Network Type'] = number.fit_transform(adaptability_transformed['Network Type'])                 #2G->0          3G->1           4G->2
adaptability_transformed['Device'] = number.fit_transform(adaptability_transformed['Device'])                             #computer->0    mobile->1       tab->2

adaptability_transformed['Adaptivity Level'] = number.fit_transform(adaptability_transformed['Adaptivity Level'])         #high->0        low->1          moderate->2
adaptability_transformed = adaptability_df
print(adaptability_transformed)

###### Select needed features and target in creating the model which is in this case 8 features and 1 target ######
features = ['Institution Type', 'IT Student', 'Location', 'Load-shedding', 'Financial Condition', 'Internet Type', 'Network Type', 'Device']
target = ['Adaptivity Level']
###### Split data for traning and testing. 80% Training and 20% Testing with a random state of 10 is chosen for this instance ######
features_train, features_test, target_train, target_test = train_test_split (adaptability_transformed[features], adaptability_transformed[target],test_size=0.20,random_state=10)
print('\n=============================================================================')
print('\nTraining Feature\n' + str(features_train))
print('\nTesting Feature\n'+ str(features_test))
print('\nTraining Target\n'+ str(target_train))
print('\nTesting Target\n'+ str(target_test))

###### Create the Naive Bayes model using the training set ######
nb_model = GaussianNB()
nb_model.fit (features_train.values,target_train.values.ravel()) 

print('\n=============================================================================')

###### Test the created model using the test set and find its accuracy by using the target test data ######
prediction = nb_model.predict(features_test.values)
accuracy = accuracy_score(target_test,prediction)
print('\nModel Accuracy: ', round(accuracy*100, 2),'%')

###### Simulate the model using the test input data and produce its prediction ######

#Input Sample Attributes for Prediction

#Input                :     0               1                   2
#Institution Type     :government     non-government
#IT Student           :no             yes
#Location in Town     :no             yes
#Load-shedding        :high           low
#Financial Condition  :mid            poor                    rich
#Internet Type        :mobile data    wifi        
#Network Type         :2G             3G                      4G
#Device               :computer       mobile                  tab
#Output
#Adaptivity Level     :high           low                     moderate

###### Each attributes with their corresponding options ######
data_columns = ['Institution Type:', 'IT Student:', 'Location in Town:', 'Load-shedding:', 'Financial Condition:', 'Internet Type:', 'Network Type:', 'Device:']
labels = [['Government','Non-Government'],['No','Yes'],['No','Yes'],['High','Low'],['Middle','Poor','Rich'],['Mobile Data','WiFi'],['2G','3G','4G'],['Computer','Mobile','Tablet']]

prediction_input = []
print('\n=============================================================================')
print('\nInput Sample attributes for Simulation (Input Only: 0, 1, 2)')

###### The user can simulate any scenario within the range of the attributes ######
inst_type = int(input("\nInstitution Type \t[0]Government [1]Non-Government\t\t:"))
prediction_input.append(inst_type)

it_student = int(input("\nIT Student \t\t[0]No [1]Yes\t\t\t\t:"))
prediction_input.append(it_student)

location = int(input("\nLocation in Town \t[0]No [1]Yes\t\t\t\t:"))
prediction_input.append(location)

load_shed = int(input("\nLoad-Shedding \t\t[0]High [1]Low\t\t\t\t:"))
prediction_input.append(load_shed)

financial_con = int(input("\nFinancial Condition \t[0]Mid [1]Poor [2]Rich\t\t\t:"))
prediction_input.append(financial_con)

internet_type = int(input("\nInternet Type \t\t[0]Mobile [1]WiFi\t\t\t:"))
prediction_input.append(internet_type)

network_type = int(input("\nNetwork Type \t\t[0]2G [1]3G [2]4G\t\t\t:"))
prediction_input.append(network_type)

device = int(input("\nDevice \t\t\t[0]Computer [1]Mobile [2]Tablet\t\t:"))
prediction_input.append(device)

#print(prediction_input)
print('\n=============================================================================')

###### Summarize the chosen scenario of the user ######
values = []
for i in range(8):
    if prediction_input[i]==0:
        values.append(labels[i][0])
    elif prediction_input[i]==1:
        values.append(labels[i][1])  
    else:
        values.append(labels[i][2])

data = {'Attributes':data_columns, 'Values':values, 'Label Encoded':prediction_input}
attr_list = pd.DataFrame(data)
print('\n',attr_list,'\n')

###### Print the predicted Adaptability Level computed by the created Naive Bayes model ######
adaptability = nb_model.predict([prediction_input])
if adaptability == 0:
    print ("Student's Adaptability in Online Education: High")
elif adaptability == 1:
    print ("Student's Adaptability in Online Education: Low")
elif adaptability==2:
    print ("Student's Adaptability in Online Education: Moderate")

###### Check if the scenario inputted by the user is present in the dataset ######
temp_df = adaptability_transformed
for i in range (8):
    filtered_df = temp_df[temp_df.iloc[:,i]==prediction_input[i]]
    temp_df = filtered_df

print('\n=============================================================================')

if (filtered_df.empty==False):  ###### If there are similar rows, count how many are in each level of adaptability ######
    print('\nSimilar Rows in the dataset: ', len(filtered_df.index))
    
    print('\nAdaptability Level Summary of Similar Rows:\n')
    adapatability_count = pd.DataFrame(filtered_df['Adaptivity Level'].value_counts())
    adapatability_count = adapatability_count.rename(columns= {'Adaptivity Level':'Count'}, index={2:'Moderate',1:'Low',0:'High'}, inplace = False)

    adapatability_percentage = pd.DataFrame(filtered_df['Adaptivity Level'].value_counts(normalize = True)*100)
    adapatability_percentage = adapatability_percentage.rename(columns= {'Adaptivity Level':'Percentage'}, index={2:'Moderate',1:'Low',0:'High'}, inplace = False)
    
    col = adapatability_percentage['Percentage']
    adapatability_count = adapatability_count.join(col)
    adapatability_count = adapatability_count.to_string(formatters={'Percentage':'{:.2f}%'.format})
    print(adapatability_count)

else:   ###### Else print this text ######
    print('\nSample Input is not in the dataset')

#Sample Input
# Government            : 0
# IT Student            : 1
# Location in Town      : 0
# Load Shedding         : 1
# Financial Condition   : 0
# Internet Type         : 1
# Network Type          : 2
# Device                : 0

#Sample Input
#11010120-Moderate  6 in dataset
#00010120-Moderate  Not in Dataset
#01110120-Moderate  6 in dataset
#01000120-Low       Not in Dataset  
#01011120-Moderate  Not in Dataset
#01012120-High      Not in Dataset
#01010020-Moderate  Not in Dataset
#01010100-Low       Not in Dataset
#01010110-Moderate  Not in Dataset
#01010121-Low       Not in Dataset
#01010122-Moderate  Not in Dataset