### This is the resolution of Machine Learning Assessment for Sequencr. For this task I develop a Classification System using either Logistic Regression and Bert models. The dataset I choose was BBC News Summary(https://www.kaggle.com/datasets/pariza/bbc-news-summary/data).

### To use the API as its running in localhost one must run APIClassification.py in one terminal (to start the service, typing 'Python APIClassification.py') and then they can run TestAPI.py script. 

### In TestAPI.py one must edit the file introducing what text they want to predict and choose which model, then simply run by terminal "Python TestAPI.py" and the output will the prediction. In order for this to work, regarding file size issues, one must download a file called "model.safetensors" from this link:

### https://drive.google.com/file/d/1-4ba5eqYFBvnbRHvwYMfmDQEp5kUzb1D/view?usp=drive_link

### Afterwards it must be save inside the folder "BBC News Summary". If there is any Python library left to install, please use the file requirements.txt by typing in terminal "Python -m  pip install -r requirements. txt".

### Alternatively, its possible to run a Docker file that creates an image with all necessary dependencies. First, in the same directory as currect project one must type the command "docker build -t flask-api ." to create the image and setup the docker. Afterwards, type in terminal "docker run -d -p 5000:5000 flask-api" so docker runs in background. Afterwards, just simply run again "TestAPI.py" to make predictions.
