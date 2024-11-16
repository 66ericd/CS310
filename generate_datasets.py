import csv
import random

def biased_hiring():
    data = []
    for i in range(500):  
        score = random.randint(0, 1000) 
        gender = random.choice(["Male", "Female"]) 
        if gender == "Male":
            prob = score / 1000
        if gender == "Female":
            prob = score / 2000
    
        accepted = 1 if prob > random.random() else 0
        data.append([score, gender, accepted])

    with open("biased_hiring.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Score", "Gender", "Accepted"])
        writer.writerows(data)  

def biased_hiring_predictions():
    data = []
    for i in range(500):  
        score = random.randint(0, 1000) 
        gender = random.choice(["Male", "Female"]) 
        prob = score / 1000 
        if gender == "Male":
            prediction_prob = score / 1000 * 2  
        elif gender == "Female":
            prediction_prob = score / 1000 * 0.5 
        rn = random.random()
        accepted = 1 if prob > rn else 0  
        predicted_accepted = 1 if prediction_prob > rn else 0  
        data.append([score, gender, accepted, predicted_accepted])

    with open("biased_hiring_predictions.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Score", "Gender", "Accepted", "Predicted"])
        writer.writerows(data)

biased_hiring()