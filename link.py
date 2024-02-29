import joblib
model = joblib.load('voting_classifier.joblib')

while True:
  user_link = input("Enter a URL (type 'q' to quit): ")

  # Quit if user enters 'q'
  if user_link == 'q':
    break

  new_data = [user_link]
  prediction = model.predict(new_data)[0]

  print("Prediction:", prediction)
