# Migrate_Tensorflow_Code_To_AWS
TensorFlow's tf.keras API is used with Script Mode for a text classification task. An important aspect of the example is showing how to load preexisting word embeddings such as Fasttext in Script Mode

With Script Mode, you can use training scripts similar to those you would use outside SageMaker with SageMaker's prebuilt containers for various deep learning frameworks such TensorFlow, Keras, PyTorch, and Apache MXNet.



###SOME MEMO FOR NLP###

The pre-trained word2vector I used is from fasttext.as follows. (ja-300d)
https://fasttext.cc/docs/en/crawl-vectors.html

For tokenizer, I used Sudachi rather than most popular Japansese Tokenizer Mecab. (https://github.com/WorksApplications/SudachiPy)

The most advantage I can take is Sudachi has a feature called normalize can can help me normalize differen words which has same meaning.
