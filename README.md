## Inspiration
The idea for SwoleMate stemmed from a desire to help people track their exercise form and improve their workouts through technology. Many fitness enthusiasts struggle with maintaining proper form during exercises, which can lead to inefficient workouts or even injuries. For example, when working out, we notice that our form might not be optimal. SwoleMate aims to provide real-time feedback to users by analyzing their form and offering insights for improvement.

## What it does
SwoleMate is a fitness tracking app designed to monitor your exercise form using a live camera feed. It analyzes the movements in real-time and compares them to a pre-trained model to detect errors or improvements in technique. The app then provides feedback on the form, allowing users to adjust and improve their performance over time.

## How we built it
We used OpenCV's VideoCapture to process both prerecorded training videos and live camera feeds from the Flask app. These videos were broken down into frames, which were then analyzed using Mediapipe's pose landmarks. The x, y, z, and visibility coordinates of the key landmarks were extracted for each specific exercise. Next, we compiled the data into sequences of frames representing a single rep. These sequences were labeled based on the quality of the rep—whether it was a good rep (labeled as 0) or had some form issues (labeled with a non-zero code indicating the type of error). This labeled data was then fed into a custom LSTM model built with TensorFlow. The model, a categorical classifier, estimated the validity of each rep with an accuracy of 86%. The user interface consists of a single-page web app, where users can start, stop, and save their workout sessions. Additionally, the app features a chatbot powered

## Challenges we ran into
**Data Quality**: Gathering enough labeled data for training the LSTM model proved challenging, especially for various exercises. We had to ensure that the data was diverse enough to cover different body types, movement variations, and exercise styles to improve the model's robustness.

**Pose Estimation Accuracy**: Mediapipe’s pose landmarks are quite accurate, but it sometimes struggled with difficult poses or occlusions. For instance, certain exercises, like squats or shoulder presses, could lead to parts of the body being out of frame, which affected landmark accuracy and required extra handling.

**Real-time Processing**: The real-time aspect of the application was difficult to manage, especially when processing video frames and running them through the LSTM model. Ensuring smooth performance while maintaining accuracy was a constant challenge.

**User Interface Design**: Creating a simple and intuitive UI that effectively communicated the workout data and feedback without overwhelming the user was also tricky. We had to make sure the app was easy to use, even with all the complex backend processes happening behind the scenes.

## Accomplishments that we're proud of
**Model Accuracy**: Achieving an 86% accuracy in rep validation with our custom LSTM model was a major accomplishment. It demonstrates the effectiveness of our approach and how well the system can evaluate form.

**Chatbot Integration**: The integration of the chatbot powered by Gemini AI was a great addition. It provides users with actionable feedback on their form in a conversational way, enhancing the overall user experience.

**Seamless Workflow**: We were able to create a smooth workflow from video capture to model inference to providing feedback, all within a single-page web app. This allowed users to quickly start tracking their workouts and get real-time analysis.

## What we learned
**The Importance of Data**: We learned just how critical high-quality and varied data is for training machine learning models. The more diverse the dataset, the better the model could generalize to real-world scenarios.

**Pose Estimation Limitations**: While Mediapipe is a powerful tool, we gained a deeper understanding of its limitations in complex poses or scenarios with occlusions. This has pushed us to explore ways to improve the model’s robustness.

**Real-time Model Deployment**: Deploying machine learning models in real-time applications is much more complex than training them offline. We learned a lot about optimizing performance and ensuring real-time inference without sacrificing accuracy.
## What's next for SwoleMate

_There are a lot of ways in which we can continue updating and further exploring this app. Some of them are listed below_

**Expanding Exercise Library**: We plan to expand the number of exercises supported by SwoleMate, including more complex movements and different variations for each exercise.

**Improving Pose Estimation**: We're working on improving the accuracy of the pose estimation system to handle more challenging poses and body types more effectively.

**User Feedback Loop**: We're considering implementing a feedback loop where users can improve their own workout data by confirming rep validity, allowing the model to learn from their input and improve over time.

**Mobile App Version**: We're exploring the possibility of developing a mobile app version of SwoleMate to make the platform even more accessible for users during their workouts.
