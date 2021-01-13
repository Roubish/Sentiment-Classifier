# Build a Sentiment Classifier using Python and IMDB Reviews

### About the Project

In this project, we will be primarily working on the IMDB reviews given by the viewers about various movies.

We will be getting the dataset from the tensorflow_datasets. Then we will develop an understanding of how to preprocess the data, convert the English words to numerical representations, and prepare it to be fed as input for our deep learning model with GRUs.
### Importing the Modules

Let us begin by importing the necessary modules, and setting the random seed so as to get reproducible results.

### Loading the Dataset

First, we will load the original IMDb reviews, as text (byte strings), using TensorFlow Datasets.


### Exploring the Data

Let us see how the data looks like.
Note:

- datasets["train"] contains the train data. Similarly, datasets["test"] contains the test data.

- datasets["train"].batch(2) batches 2 data samples at a time.

- datasets["train"].batch(2).take(1) allows to take 1 batch at a time.

- Each batch is of type Eager Temsor. We could convert it to numpy array using X_batch.numpy().

### Defining the preprocess function

Now we will create this preprocessing function where we will:

Truncate the reviews, keeping only the first 300 characters of each since you can generally tell whether a review is positive or not in the first sentence or two.

Then we use regular expressions to replace
tags with spaces and to replace any characters other than letters and quotes with spaces.

Finally, the preprocess() function splits the reviews by the spaces, which returns a ragged tensor, and it converts this ragged tensor to a dense tensor, padding all reviews with the padding token
#### Note:

- tf.strings - Operations for working with string Tensors.

- tf.strings.substr(X_batch, 0, 300) - For each string in the input Tensor X_batch, it creates a substring starting at index pos(here 0) with a total length of len(here 300). So basically, it returns substrings from Tensor of strings.

- tf.strings.regex_replace(X_batch, rb"<br\s/?>", b" ") - Replaces elements of X_batch matching regex pattern <br\s/?> with rewrite .

- tf.strings.split(X_batch) - Split elements of input X_batch into a RaggedTensor.

- X_batch.to_tensor(default_value=b"


### Constructing the Vocabulary

Next, we will construct the vocabulary. This requires going through the whole training set once, applying our preprocess() function, and using a Counter() to count the number of occurrences of each word.
#### Note:

    - Counter().update() : We can add values to the Counter by using update() method.

    - map(myfunc) of the tensorflow datasets maps the function(or applies the function) myfunc across all the samples of the given dataset


### Truncating the Vocabulary

There are more than 50,000 words in the vocabulary. So let us truncate it to have only 10,000 most common words.


Creating a lookup table

Computer can only process numbers but not words. Thus we need to convert the words in truncated_vocabulary into numbers.

So we now need to add a preprocessing step to replace each word with its ID (i.e., its index in the truncated_vocabulary). We will create a lookup table for this, using 1,000 out-of-vocabulary (oov) buckets.

We shall create the lookup table such that the most frequently occurring words have lower indices than less frequently occurring words.
Note:

    - tf.lookup.KeyValueTensorInitializer : Table initializer given keys and values tensors. More here

    - tf.lookup.StaticVocabularyTable : String to Id table wrapper that assigns out-of-vocabulary keys to buckets. More here

    - If table.lookup : Looks up keys in the table, outputs the corresponding values.


### Creating the Final Train and Test sets

Now we will create the final training and test sets.

For creating the final training set train_set,

   -  we batch the reviews

   -  then we convert them to short sequences of words using the preprocess() function

   -  then encode these words using a simple encode_words() function that uses the table we just built and finally prefetch the next batch.

Let us test the model(after training) on 1000 samples of the test data as it takes a lot of time to test on the whole test set. So we shall create the final test set on 1000 samples as follows.

For creating the final test set test_set,

   -  we create a batch of 1000 test samples

   -   then we convert them to short sequences of words using the preprocess() function

   -  then encode these words using a simple encode_words() function that uses the table we just built.

#### Note:

   -  dataset.repeat().batch(32) repeatedly creates the batches of 32 samples in the dataset.

   -  dataset.repeat().batch(32).map(preprocess) applies the function preprocess on every batch.

  -   dataset.map(encode_words).prefetch(1) applies the function encode_words to the data samples and paralelly fetches the next batch.


### Building the Model

   - Now that we have preprocessed and created the dataset, we can create the model:
        The first layer is an Embedding layer, which will convert word IDs into embeddings. The embedding matrix needs to have one row per - word ID (vocab_size + num_oov_buckets) and one column per embedding dimension (this example uses 128 dimensions, but this is a hyperparameter you could tune).
        Whereas the inputs of the model will be 2D tensors of shape [batch size, time steps], the output of the Embedding layer will be a 3D tensor of shape [batch size, time steps, embedding size]. #### Note:

   - keras.layers.Embedding : Turns positive integers (indexes) into dense vectors of fixed size. More here.
   - keras.layers.GRU : The GRU(Gated Recurrent Unit) Layer.


### Training and Testing the Model

   -  It's time for training the model on the train data.

   -  Let us also measure the time of training using time module.

   -  Finally, let us test the model performance on the test data.

