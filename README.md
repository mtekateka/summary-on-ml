#Introduction to Machine Learning

Machine Learning is the science (and art) of programming computers so they can learn from data. For example, Spam filter is a Machine Learning program that can learn to flag spam given examples (training set) of spam emails (e.g., flagged by users) and examples of regular (nonspam, also called “ham”) emails.

#Use of Machine Learning:

• Problems for which existing solutions require a lot of hand-tuning or long lists of rules: one
Machine Learning algorithm can often simplify code and perform better.
• Complex problems for which there is no good solution at all using a traditional approach:
the best Machine Learning techniques can find a solution.
• Fluctuating environments: a Machine Learning system can adapt to new data. Getting
insights about complex problems and large amounts of data.

#Types of Machine Learning Systems

Machine Learning types are categorized according to.
• Whether or not they are trained with human supervision (supervised, unsupervised, semisupervised, and Reinforcement Learning).
• Whether or not they can learn incrementally on the fly (online versus batch learning).
• Whether they work by simply comparing new data points to known data points, or instead
detect patterns in the training data and build a predictive model, much like scientists do (instance-based versus model-based learning).

Note: These criteria can be combined together.

#Supervised Learning

In supervised learning, the data set(training data) that used to feed your algorithm contains labels. A typical supervised learning task are:
• Classification where by an object is predicted to be member of given class or not (according to your problem). Example if a given email is spam or not.
• To predict a target numeric value, such as the price of a car.

#Some of most important supervised learning algorithm are:

• k-Nearest Neighbors
• Linear Regression
• Logistic Regression
• Support Vector Machines (SVMs)
• Decision Trees and Random Forests
• Neural networks

#Unsupervised learning

This algorithm use training data which has no labeles.
Some of most important unsupervised learning algorithms are: • Clustering
k-Means
Hierarchical Cluster Analysis (HCA) Expectation Maximization
• Visualization and dimensionality reduction Principal Component Analysis (PCA)
Kernel PCA
Locally-Linear Embedding (LLE)
t-distributed Stochastic Neighbor Embedding (t-SNE)
• Association rule learning Apriori
Eclat

Visualization algorithms are good examples of unsupervised learning algorithms: where by they fed with a lot of complex and unlabeled data, and they output a 2D or 3D representation of your data that can easily be plotted. These algorithms try to preserve as much structure as they can, so you can understand how the data is organized and perhaps identify unsuspected patterns.

Related tasks of unsupervised learning

• Dimensionality reduction, in which the goal is to simplify the data without losing too much
information. One way to do this is to merge several correlated features into one.
• Anomaly detection. For example, detecting unusual credit card transactions to prevent fraud,
catching manufacturing defects, or automatically removing outliers from a dataset before feeding it to another learning algorithm.

The system is trained with normal instances, and when it sees a new instance it can tell whether it looks like a normal one or whether it is likely an anomaly.

#Semisupervised learning

This is whereby some algorithms can deal with partially labeled training data, usually a lot of unlabeled data and a little bit of labeled data.
Example: Photo hosting services like Google Photos, which is capable of identifying a person from all uploaded photos but not capable of identifying names(this is unsupervised), so you need to label only one time the name of that person letter on the system will be able to identify that person with his/her name from any photo that you have uploaded.

#Reinforcement Learning

Reinforcement Learning is a very different from other learning methods. The learning system, called an agent which is capable of observing it’s environment, select and perform actions based ont it’s environment, and get rewards in return (or penalties in the form of negative rewards). The agent must then learn by itself what is the best strategy, called a policy, to get the most reward over time. A policy defines what action the agent should choose when it is in a given situation.
For example, many robots implement Reinforcement Learning algorithms to learn how to walk. DeepMind’s AlphaGo program is also a good example of Reinforcement Learning.

#Batch and Online Learning

Another criterion used to classify Machine Learning systems is whether or not the system can learn incrementally from a stream of incoming data.

#Batch learning

In batch learning, the system must be trained using all the available data. Which is time consuming, need a lot of computing resources, so this process is done offline. So after training the system does not need to learn anymore, it applies what it has learned. This is called offline learning.
If you want a batch learning system to know about new data , you need to train a new version of the system from scratch on the full dataset (not just the new data, but also the old data), then stop the old system and replace it with the new one.

#Online learning

In online learning, you train the system incrementally by feeding it data instances sequentially, either individually or by small groups called mini-batches. Each learning step is fast and cheap, so the system can learn about new data on the fly, as it arrives.

Online learning is great for systems that receive data as a continuous flow (e.g., stock prices) and need to adapt to change rapidly or autonomously. It is also a good option if you have limited computing resources: once an online learning system has learned about new data instances, it does not need them any more, so you can discard them (unless you want to be able to roll back to a previous state and “replay” the data). This can save a huge amount of space.

Online learning algorithms can also be used to train systems on huge datasets that cannot fit in one machine’s main memory (this is called out-of-core learning). The algorithm loads part of the data, runs a training step on that data, and repeats the process until it has run on all of the data.

Note: This whole process is usually done offline (i.e., not on the live system), so online learning can
be a confusing name. Think of it as incremental learning.
The learning rate in an important parameter of online learning systems which decide how fast the system should adapt to changing data. It should be well regulated, should not be set either high or low. Since high learning rate the system will adapt to new data, but will likely forget old data, and if learning rate is set to low the system will learn more slowly, but it will also be less sensitive to noise in the new data or to sequences of nonrepresentative data points.

#Approaches to Generalization

There are two main approaches to generalization.
• instance-based learning and
• model-based learning.

#Instance-based learning

The system learns the examples by heart, then generalizes to new cases using a similarity measure. For example, Spam filter will flag all spam emails that are identical to emails that have alredy been flagged by users. It not the worst solution, but certainly not the best.

#Model-based learning

Another way to generalize from a set of examples is to build a model of these examples, then use that model to make predictions.

#Main Challenges of Machine Learning

Since the main task is to select a learning algorithm and train it on some data, hence the challenges that we have are:
• bad data and
• bad algorithm

#EXAMPLES OF BAD DATA Insufficient Quantity of Training Data

Machine Learning algorithms require a lot of data in order to solve the problem well, or so as to work properly. Even for very simple problems it require thousands of examples, and for complex problems such as image or speech recognition you may need millions of examples (unless you can reuse parts of an existing model).

#Nonrepresentative Training Data

In order to generalize well, it is crucial that your training data be representative of the new cases you want to generalize to. This is true whether you use instance-based learning or model-based learning. By using nonrepresentative training set to train your model you will result in getting inaccurate predictions. It is very important to use a training set that is representative of the cases you want to generalize to.

If the sample is too small, you will have sampling noise (i.e., nonrepresentative data as a result of chance), but even very large samples can be nonrepresentative if the sampling method is flawed. This is called sampling bias.
Poor-Quality Data

If your training data is full of errors, outliers, and noise such that due to poor-quality measurements, it will make it harder for that system to detect the underlying patterns, which will result your system not to work/perform well. So it is very important to spend your time on cleaning up your data.

#For example:

• Remove or fixing instances which are outliers.
• Ignoring some instances which have missing features, or just ignoring the attribute
altogether which has missing values. Or you may fix these missing values by fill in the missing values (e.g., with the median age).

#Irrelevant Features

As the saying goes: garbage in, garbage out.
The system will only be capable of learning if the training data contains enough relevant features and not too many irrelevant ones, because if your modal fed with garbage data, then expect to get garbage result. A critical part of the success of a Machine Learning project is coming up with a good set of features to train on. The process is called feature engineering.

#Feature engineering, involve the following:

• Feature selection: selecting the most useful features to train on among existing features.
• Feature extraction: combining existing features to produce a more useful one (such as
dimensionality reduction algorithms).
• Creating new features by gathering new data.

#EXAMPLES OF BAD ALGORITHM Overfitting the training datasets

This is the situation whereby the model performs well on the training data, but it does not generalize well. Complex models such as deep neural networks can detect subtle patterns in the data, but if the training set is noisy, or if it is too small (which introduces sampling noise), then the model is likely to detect patterns in the noise itself. Obviously these patterns will not generalize to new instances.
Underfitting the Training Data
Underfitting is the opposite of overfitting, it occurs when your model is too simple to learn the underlying structure of the data.

#The main options to fix this problem are:

• Selecting a more powerful model, with more parameters
• Feeding better features to the learning algorithm (feature engineering)
• Reducing the constraints on the model (e.g., reducing the regularization hyperparameter)
