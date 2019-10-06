# Migrate_Tensorflow_Code_To_AWS
Using Keras API, as well asload pre-trained word embeddings such as Fasttext in Script Mode

With Script Mode, I can use training scripts similar to those you would use outside SageMaker with SageMaker's prebuilt containers for various deep learning frameworks such TensorFlow, Keras, PyTorch, and Apache MXNet.



###SOME MEMO FOR NLP tasks###

The pre-trained word2vector I used is from fasttext as follows. (ja-300d)
https://fasttext.cc/docs/en/crawl-vectors.html

For tokenizer, I used Sudachi rather than most popular Japansese Tokenizer Mecab. (https://github.com/WorksApplications/SudachiPy)

The most advantage I can take of is Sudachi has a feature called normalize can can help me normalize differen words which has same meaning.

Such as 分かる、わかる、分った、判る、解る, they have different appearances but has same meaning.

For this use case, since comment is usually short, I prefer to use CNN model with mutile filters and conn2D, it's having a good performance for most scenarios.
