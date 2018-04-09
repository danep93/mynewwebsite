---
title: "What is a name? Intro to Long Short Term Networks"
date: 2018-03-28T13:55:10-04:00
draft: false
---
![no-name](img/no-name.jpg)

<br><br>
Today I want to talk about names. Mine has been around since the Old Testament, but recently celebrities have been getting pretty creative with what they call their kids. A few weeks ago my team was assigned a project parsing resumes for new hires, and when I realized that some students actually don't put their name at the top of the page, I thought this would be an interesting problem to solve using a neural net. That is today’s mission: to distinguish names from regular words, and eventually, distinguish male names from female names.

Long Short Term Memory Networks (LSTM) are a type of recurrent neural network that are great at this application; they use context from previous input features to predict what the next output should be. For example, if you wanted to predict the end of the sentence “in France they speak ___”, and use each (tokenized) word of the sentence as a feature, the LSTM would use “France” and “speak” to deduce the sentence ends in “French”. This assumes you have a network that’s already been trained on an extensive corpus of the English language, such as the Glove dataset. That’s exactly what Spacy is; it’s a Natural Language Processing library that’s pre-trained on this general English language corpus and uses sentence context to do things like classify words as adjectives, proper nouns, etc… Unfortunately, without sentence context, Spacy comes up short. When parsing the resume header

john smith
<br>
JohnSmith68@gmail.com
<br>
My mission is to better the world and get paid

Spacy comes up short and fails to catch john or smith. If you want to try it for yourself then install spacy (easiest through pip) and test out some sentences. Disclaimer, be extra careful with word capitalization.


{{< highlight go "linenos=table,linenostart=1" >}}
Import spacy
nlp = spacy.import(‘en’)
mystr = ‘my friend John wants dave to know what is popular these days’
document = nlp(mystr)
print([token for token in document if token.ent_type_ == 'PERSON'])
{{< / highlight >}}

I started with a few datasets found off Kaggle which is a really great platform that’s gotten a bunch of people in the competitive data science space. The datasets I used were a list of male names, female names, and the most common internet words. First thing I did was load them into python as clean data frame objects.


{{< highlight go "linenos=table,linenostart=1" >}}
with open(females_names_path) as f:
    female_lines = f.read().splitlines()
with open(male_names_path) as f:
    male_lines = f.read().splitlines()
names_list = list(set(male_lines + female_lines))
names_df = pd.DataFrame(np.array(names_list), columns = ['word'])
names_df['target'] = 1
# create DF out of internet words, drop count column add target column
#want 50/50 distribution in data, no bias
internet_df = pd.read_csv(internet_words_path, nrows=names_df.size)
internet_df['target'] = 0
internet_df = internet_df.drop(['count'], axis=1)
# use outer so names in internet_df that appear in names_df get target 1
merged_df = pd.merge(names_df, internet_df, how='inner', on=['word', 'target'])
merged_df = merged_df.dropna()
merged_df['word'] = merged_df['word'].str.lower()
merged_df['word'] = merged_df['word'].str.strip()
merged_df['word_length'] = merged_df['word'].apply(lambda x: len(x))
X = merged_df['word']
y = merged_df['target']
valid_chars = {x:idx+1 for idx, x in enumerate(set(''.join(X)))}
max_word_len = np.max([len(x) for x in X])
max_features = len(valid_chars) + 1
x_data_sequences = [[valid_chars[char] for char in word] for word in X]
x_data_sequences = sequence.pad_sequences(x_data_sequences, maxlen=max_word_len)
{{< / highlight >}}

Where a target value of 1 means “name” and 0 means “internet word”

Neural networks are bad at working with sentences but great at working with vectors, which is why we’ll continue with our preprocessing by tokenizing each character, followed by converting each word into a vector of integers. For example, if we create the mapping of ‘d’=1, ‘o’ = 2, ‘g’ = 3, the word dog will (almost) show up to our network as <1,2,3>. I say almost for two reasons: first because it’s important to give our LSTM network only vectors of a constant size. This is why we pad each vector with a bunch of 0’s until they’re all the same length. Second because before we reach the LSTM layer of our network we want to pass our input through an Embedding Layer. This turns our wasteful vectors (so many wasteful 0’s) into dense vectors of higher dimensionality. Without wasting more time I give you Deep Learning

{{< highlight go "linenos=table,linenostart=1" >}}
embedding_layer = Embedding(max_features, output_dim=64, input_length=max_word_len)
lstm_layer = LSTM(max_features)
dropout_layer = Dropout(0.5)
dense_layer = Dense(1)
sigmoid_layer = Activation('sigmoid')

model = Sequential([embedding_layer, lstm_layer, dropout_layer, dense_layer, sigmoid_layer])
model.compile(loss='binary_crossentropy', optimizer='SGD')

epochs = 18
batch_size = 32
X_train, X_test, y_train, y_test = train_test_split(x_data_sequences, y, test_size=0.2, random_state=0)
history = model.fit(X_train, y_train, epochs = epochs, validation_split=0.33, batch_size = batch_size)
predicted_y_test = model.predict(X_test)
{{< / highlight >}}


So that was a lot, let’s walk it back a little bit.
**batch_size** refers to how many iterations (not epochs, but training rows) you process before you update the parameter weights. An **epoch** is a complete pass through of all the training data.
Our **embedding layer** takes in vectors of size max_word_len, with max_features distinct values, and outputs dense vectors of 64 dimensions each. Our **Dropout layer** dictates that after each batch_size iterations we will randomly drop 50% of all neurons in our LSTM layer (where we have max_features neurons) before backpropagation (the part where a neural net goes back and updates it's connection weights). The values outputted by these dropped neurons do not affect the weight changes sent by backpropagation, nor will they be changed by those weight changes. LSTMs are notorious over-fitters so this helps keep classifications unbiased.
 Our **Dense layer** takes our multiple neurons outputted by our LSTM layer and outputs into a single scalar value
Our Sigmoid Activation layer takes that single value and fits it to a score between the range of 0 and 1, great for binary classification.

Now we split our data into train and test sets and fit the model, specifying a total of 10 epochs and designating 33% of our training data to be used as a holdout/validation set. Holdout cross validation isn’t ideal but we’ll talk about that later. Below we’ve plotted the loss vs epoch

![first-loss-graph](img/first-loss-graph.png)

As you can see, after each epoch our training and validation loss both go down, which is what we love to see. Here you can see the sweet spot for training and validation data where validation error is at a global minimum

![ideal-loss-graph](img/ideal-loss.png)

Our graph hasn’t evened out at the end which means our model is relatively under-fitted. We’ll talk more about that later and what we can do to correct it.

Let’s use a couple common metrics to determine how well our model performed. We create a DataFrame with all our columns and make a few new ones. Rounded_y_pred takes the sigmoid score and rounds to either 1 or 0 since our y_true values are either 1 or 0, and never in between.

{{< highlight go "linenos=table,linenostart=1" >}}
f1_score = metrics.f1_score(eval_df['y_true'].values, eval_df['rounded_y_pred'].values)
accuracy_score = metrics.accuracy_score(eval_df['y_true'].values, eval_df['rounded_y_pred'].values)
print("accuracy score: ", str(accuracy_score))
print("f1 score: ", str(f1_score))
print("log loss score: ", str(history.history['loss'][-1]))
{{< / highlight >}}

```
('accuracy score: ', '0.706134564644')
('f1 score: ', '0.693076128143')
('log loss score: ', '0.652990719739')
```
**F1** gives us a score based off the number of true/false negatives and true/false positives. This is very informative for binary classification
**Accuracy** gives us a fraction of how many correct classifications vs incorrect classifications.
**Log Loss** goes a bit deeper. If our sigmoid score is 0.51 for the word “Annabelle”, that means our model is 51% sure that word is a name, which is a hesitant yes but a yes at the end of the day. This shows up as a perfect classification as far as F1 and Accuracy are concerned. Log Loss incorporates the uncertainty since ideally we want our model to give us a score as close as possible to 1.

You might be confused about the difference between accuracy and f1 score, especially considering we got similar scores for both. Accuracy works well when you assign the same "penalty" to False Positives (FP) and False Negatives (FN), but this is
not always the case. Imagine you work for NASA; you'd much rather get a false alarm that oxygen levels are dangerously low compared to no alarm. For a more in depth explanation, look up Precision and Recall and how they relate to F1.


While our scores aren't bad, I think we can do better.

One of the hardest parts of getting a well performing Neural Net is tuning the hyper parameters just right. While parameters are the weights used to favor certain connections over others, hyperparameters dictate how you’re going to fit your model. Tuning the hyperparameters results in more accurate parameters which gives us a better accuracy. Here we use GridSearchCV to wrap our neural network in a KerasClassifier object that tries every possible combination of hyper parameters that we pass in. There are many more hyper parameters but these are some of the most common for tuning.
As an added benefit, GridSearchCV employs KFold cross validation instead of holdout validation, which means each one of our training points will be eventually used for training our model. More training points means more opportunities for our model to learn. If k = 3, we will perform training 3 times, where each time we use a different 66% of our data for training and 33% for cross validation.

{{< highlight go "linenos=table,linenostart=1" >}}
def create_model(shuffle=True, optimizer='Adam', dropout=0.5, embed_dimensions=64):
    embedding_layer = Embedding(max_features, output_dim=embed_dimensions, input_length=max_word_len)
    lstm_layer = LSTM(max_features)
    dropout_layer = Dropout(dropout)
    dense_layer = Dense(1)
    sigmoid_layer = Activation('sigmoid')
    model = Sequential([embedding_layer, lstm_layer, dropout_layer, dense_layer, sigmoid_layer])
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

X_train, X_test, y_train, y_test = train_test_split(x_data_sequences, y, test_size=0.2, random_state=0)
batch_size = 32
epochs = 5
hyperparams_dict = {
    'optimizer': ['SGD', 'Adam', 'RMSprop', 'Adagrad', 'Adadelta', 'Adamax'],
    'dropout': [0.1, 0.2, 0.4, 0.6],
    'embed_dimensions': [32,64,128,256],
    'batch_size': [20, 32, 40],
    'epochs': [18,25]
}

model = KerasClassifier(build_fn=create_model, verbose=1)
grid = GridSearchCV(estimator=model, param_grid=hyperparams_dict, n_jobs=-1, verbose=1)
grid_result = grid.fit(X_train, y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
{{< / highlight >}}


Sure enough, grid_result tells us we get our best accuracy with . . .<br>
*Optimizer*: Adam <br>
*Dropout*: 0.2 <br>
*embed_dimensions*: 256 <br>
*batch_size*: 32 <br>
*Epochs*: 18 <br>

Disclaimer, have your computer plugged in if you’re going to try this. I ran this overnight and it was still running when I woke up.
Now let’s take a look at our updated loss graph

![second-loss-graph](img/second-loss-graph.png)

As you can see, we’re much closer to that sweet spot that’s circled above. Thanks to gridsearchcv our loss at epoch 1 is less than our loss at epoch 15 when using our initial hyper parameters. Now let’s check out our accuracy

```
accuracy score:  0.822559366755
f1 score:  0.821381142098
log loss score:  0.315463500042
```

As you can see we improved our metrics across the board by at least 10 points.

Exploratory Data Analysis (EDA) is an important part of any data science problem. Rarely do you get perfect classification on your first try. Once you understand what type of problems your model is bad at you can attack the problem at the source

Below I’ve visualized the F1 scores, where green dots are points that are correctly classified while red points are incorrectly classified. Points appearing in the left side of the screen received a sigmoid score of under 0.5, meaning our model predicted they are words, while points on the right side are predicted as names. The worst thing we can see is a bunch of red points at either x-extreme. Here I’m graphing the sigmoid score against word length and vowel:consonant ratio to see if either of those features affect classification.

![vowel ratio](img/vowel-graph.png)
![word length](img/word-length-graph.png)


Unfortunately there doesn’t appear to be any correlation between these dependent and independent variables. However, I think this is a great opportunity to point something out. Looking at our graph above, we have a bunch of green points located at both X-axis extremes of the graph; this is great because it means not only did we correctly classify both words and names, but often times we were VERY sure it was a name (sigmoid score ~1) or a word (sigmoid score ~0). Additionally, most of our red dots (incorrectly classified) are located near the 0.5 barrier, which means although we got them wrong, our model made classifications with a grain of salt. Let's compare this to the same graph we got before we tuned our hyperparameters

![first word length](img/first-wordlength.png)

As you can see, our original model had plenty of correctly classified words and names but all our green points gravitated towards the middle area. Out of the thousands of words we passed in, there wasn't a
single one that our model was 80% sure of what it was. I know we already mentioned that our accuracy went up a few percentage points after hyperparameter tuning, but I think this does a better job showing us exactly how much better of a model we ended up with.


We ended up with a model that can distinguish between names and non-names with an accuracy of 83% and a log loss bordering 0.3. Ideally it would have performed a little better, however, I have to cut our model some slack. Names are a tough domain, and most of the times I’m not even sure how I feel about certain ones. Liz Lemon has plenty of opinions though

https://www.youtube.com/watch?v=-2hvM-FNOEA


What if we wanted to go one step further and not just determine if a word is a name, but a boys name or girls name. A few things change. First off we modify our pandas data wrangling. Here we use scikit-learn get_dummies to one hot encode our vectors. If there are X possible values, one-hot encoding creates X new features, where each row will be populated by all 0’s except for one 1 designating

{{< highlight go "linenos=table,linenostart=1" >}}
male_df = pd.DataFrame({'word': male_list, 'target': 'male'})
female_df = pd.DataFrame({'word': female_list, 'target': 'female'})

internet_df = pd.read_csv(internet_words_path) #want 50/50 distribution in data, no bias
internet_df = internet_df.drop(['count'], axis=1)
internet_df['target'] = 'internet'

min_size = min(male_df.shape[0], female_df.shape[0], internet_df.shape[0])
male_df = male_df.head(min_size)
female_df = female_df.head(min_size)
internet_df = internet_df.head(min_size)
# merge them, clean up, apply get_dummies
name_frames = [male_df, female_df]
merged_names_df = pd.concat(name_frames)
#drop all names that appear in both boys and girls list. Later we'll test some gender neutral names
merged_names_df = merged_names_df.drop_duplicates(subset='word', keep=False)

frames = [merged_names_df, internet_df]
merged_df = pd.concat(frames)
#drop all names that appear in internet words list by getting rid of duplicates and keeping first (merged_names)
merged_df = merged_df.drop_duplicates(subset='word', keep='first')
merged_df = pd.get_dummies(merged_df, columns=['target'])
merged_df = merged_df.dropna()
merged_df['word'] = merged_df['word'].str.lower()
merged_df['word'] = merged_df['word'].str.strip()
X = merged_df['word']
y = merged_df[['target_male', 'target_female', 'target_internet']].values
{{< / highlight >}}

We can no longer assign a target of 1 or 0 because it’s no longer a yes no question. Similarly, a single sigmoid score in the range of 0-1 doesn’t tell us much. Why can’t we just separate sigmoid into [0-.33) (.33-.66] (.66 -1] ? Final scores are highly dependent on the gradient that exists at your exact point on an activation function. Sigmoid has a steep curve between X values [-2,2], which means at that region, small changes in X will result in huge changes in Y. This means the function naturally gravitates towards either end of the curve.

Instead we will use a softmax function, which is the multi class version of sigmoid which assigns probability scores to all classes where all probabilities add to 1. Here is the code for our new three way classification problem

{{< highlight go "linenos=table,linenostart=1" >}}
batch_size = 32
epochs = 18
embedding_layer = Embedding(max_features, 256, input_length=max_word_len)
lstm_layer = LSTM(max_features)
dropout_layer = Dropout(0.2)
dense_layer = Dense(3)
softmax_layer = Activation('softmax')

model = Sequential([embedding_layer, lstm_layer, dropout_layer, dense_layer, softmax_layer])
model.compile(loss='binary_crossentropy', optimizer='adam')

X_train, X_test, y_train, y_test = train_test_split(x_data_sequences, y, test_size=0.2, random_state=0)
history = model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size, validation_split=0.33)
{{< / highlight >}}



Notice that we changed our Dense(1) layer to Dense(3). That’s because before we wanted to pass a single scalar to our Sigmoid function, but now we want to pass 3 scores to our softmax function to be converted to probabilities of certainty. Anything different would cause an error complaining about mismatched dimensions. We fit our model and use it to predict values on test data, indicated by y_true columns. Below are a few examples of how our predictions match up against our true test values.

![softmax-predictions](img/softmax-predictions.png)

Woah. That's both impressive and hard to interpret so let me explain. When we changed our y-values to an array
built by pd.get_dummies, we passed in targets where each target was an array of size 3, where each array has
exactly 1 non-zero value. If the non-zero is in the first, second, or third column it represents a row that is
a boys name, girls name, or internet word, respectively.
Also remember, our softmax function turns our three scores (outputted by our Dense(3) layer) into probabilities
of certainty for classification, where all certainties add up to 100%. Taking the first row as an example, our
neural network is 18% sure that the word "usually" is a boys name, 20% sure it's a girls name, and 62% sure it's
an internet word. In softmax world, we go with whatever classification has the highest confidence, which here is resoundingly
class 3, or belonging to an internet word. The "flat y_true" column indicates the index of the nonzero value which gives us the class, while flat y_pred gives you the index of the maximum element in y_pred, which is the predicted classification.



Here are some more examples in case you want to continue to pit yourself against a machine

![more-softmax-predictions](img/more-softmax-predictions.png)

Pshh.... I knew Grissel was a girls name. Teutonic baby name
meaning gray haired heroine. That name is *so hot right now*.


Now let’s check out our new overall accuracy

{{< highlight go "linenos=table,linenostart=1" >}}
accuracy score:  0.747023809524
f1 score:  0.745798319467
log loss score:  0.34714355651
{{< / highlight >}}


While our overall accuracy went down after switching from two-way to three-way classification, it makes sense. While initially our model had a 50% chance of getting any prediction correct, now it has a 33% chance, and there are many names where it’s a toss up (Taylor, Charlie, Alex ect...). In fact, let’s try some of those out.

First, it's important to ensure that none of these "testing" words appear in our training dataset. Otherwise
it's cheating because our model has already seen them and seen the correct label

{{< highlight go "linenos=table,linenostart=1" >}}
training_names = [sequence_to_string(x) for x in X_train]
difficult_names = ['taylor','tylor','charli','charlie','charlie','alex','alexandra','alexander']
in_training = set(training_names).intersection(set(difficult_names))
print(list(in_training))
{{< / highlight >}}
{{< highlight go "linenos=table,linenostart=1" >}}
['charlie', 'taylor', 'alexander']
{{< / highlight >}}

Now we know that we can't test our model on Charlie, Taylor, or Alexander. Moving forwards, let's see what our model (which correctly called Grissel by the way) tells us about these other names

![final-testing-names](img/final-testing-names.png)

You love to see that. Our model correctly identified Charli as a girls name and Charly as (just barely) a boys name. Think our model could have done better? 2 of these are celebrity baby names, 2 are regular words. Let’s see how many you get.

Bronx
Journey
Racer
Java

And there you have it! Today we learned what it takes to reformat and preprocess your data, pass through an LSTM network, perform grid search cross validation to perfect hyper parameters, and evaluate results. Thank you for reading, if you’ve gotten this far you should know I lied, all four of those names belong to real celebrity babies. Here you can find all notebooks used in this blogpost --> https://github.com/danep93/lstm-names/notebooks. Want to try your own words? Check out plug-n-chug-softmax. A special thanks to Domenic Puzio https://www.linkedin.com/in/domenicpuzio/ for helping me get started and machinelearningmastery for covering every topic under the sun.
