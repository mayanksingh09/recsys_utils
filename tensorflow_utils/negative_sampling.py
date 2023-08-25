import tensorflow as tf
import random


class InBatchNegativeSampling(tf.keras.layers.Layer):
    """
    Custom Keras layer that performs implicit in-batch negative sampling on query and item embeddings

    :param dense_layer_size: int
        Size of the dense layer after the pooling layer.
    :param n_negative_sample: int
        Number of negative samples to generate for each positive sample.
    :param batch_size: int
        Size of the input batch.

    Methods
    -------
    call(inputs)
        Method that performs the negative sampling and computes dot product.

    get_config()
        Method to get the configuration of the layer.

    Returns
    -------
    query_embeddings_neg: tensor
        A tensor of shape `((n_negative_sample + 1) * batch_size, embedding_dim)` representing the negative-sampled query embeddings.
    item_embeddings: tensor
        A tensor of shape `((n_negative_sample + 1) * batch_size, embedding_dim)` representing the negative-sampled item embeddings.

    """

    def __init__(self, dense_layer_size, n_negative_sample, batch_size, **kwargs):
        self.dense_layer_size = dense_layer_size
        self.n_negative_sample = n_negative_sample
        self.batch_size = batch_size
        super(InBatchNegativeSampling, self).__init__(**kwargs)

    def call(self, inputs):
        """
        Method that performs the negative sampling

        :param inputs: tuple (query_embedding, item_embedding)
            query_embedding: shape `(batch_size, embedding_dim)`
            item_embedding: shape `(batch_size, embedding_dim)`

        """
        query_embeddings, item_embeddings = inputs

        # in-batch negative sampling
        negative_samples = random.sample(range(1, self.batch_size), self.n_negative_sample)

        item_temp_embeddings = item_embeddings

        # slice and concatenate negative samples
        for a_sample in negative_samples:
            item_embeddings = tf.concat([item_embeddings,
                                         tf.slice(item_temp_embeddings, begin=[a_sample, 0], size=[self.batch_size - a_sample, -1]),
                                         tf.slice(item_temp_embeddings, begin=[0, 0], size=[a_sample, -1])], 0)

        query_embeddings_neg = tf.tile(query_embeddings, [self.n_negative_sample + 1, 1])

        return query_embeddings_neg, item_embeddings

    def get_config(self):
        config = super(InBatchNegativeSampling, self).get_config()
        config.update({'dense_layer_size': self.dense_layer_size,
                       'n_negative_sample': self.n_negative_sample,
                       'batch_size': self.batch_size})
        return config


class InBatchNegativeSamplingDotProduct(tf.keras.layers.Layer):
    """
    Custom Keras layer that performs in-batch negative sampling on query and item embeddings and computes dot product.
    Uses negative labels that impact the positive example's probability during softmax computation
    Suited for Retrieval tasks

    :param dense_layer_size: int
        Size of the dense layer after the pooling layer.
    :param n_negative_sample: int
        Number of negative samples to generate for each positive sample.
    :param batch_size: int
        Size of the input batch.
    :param scaling_constant: float, optional
        Constant to smooth the dot product. Default is 1.0.
        1, 5, 10, 20 are good values to try
    :param temperature: float, optional
        Temperature to scale the dot product. Default is 1.0.

    Methods
    -------
    call(inputs)
        Method that performs the negative sampling and computes dot product.

    get_config()
        Method to get the configuration of the layer.

    Returns
    -------
    pred: tensor
        A tensor of shape `(dense_layer_size, 1)` representing the positive-example probabilities

    """

    def __init__(self, dense_layer_size, n_negative_sample, batch_size,
                 scaling_constant=1.0, temperature=1.0,
                 **kwargs):
        self.dense_layer_size = dense_layer_size
        self.n_negative_sample = n_negative_sample
        self.batch_size = batch_size
        self.scaling_constant = scaling_constant
        self.temperature = temperature
        super(InBatchNegativeSamplingDotProduct, self).__init__(**kwargs)

        assert self.scaling_constant > 0, "Scaling constant should be greater than 0"
        assert self.temperature > 0, "Temperature should be greater than 0"

    def call(self, inputs):
        """
        Method that performs the negative sampling and computes dot product.

        :param inputs: tuple (query_embedding, item_embedding)
            query_embedding: shape `((n_negative_sample + 1) * batch_size, embedding_dim)`
            item_embedding: shape `((n_negative_sample + 1) * batch_size, embedding_dim)`
        :return: pred: tensor
            Tensor of shape `(dense_layer_size, 1)` representing the predicted scores for each query-item pair.

        """
        query_embeddings, item_embeddings = inputs

        query_embeddings_neg, item_embeddings = InBatchNegativeSampling(self.dense_layer_size,
                                                                        self.n_negative_sample,
                                                                        self.batch_size)([query_embeddings,
                                                                                          item_embeddings])


        # Assert statement to check the shape of the embeddings output from InBatchNegativeSampling
        if query_embeddings_neg.shape[0] is not None and item_embeddings.shape[0] is not None:
            assert query_embeddings_neg.shape == ((self.n_negative_sample + 1) * self.batch_size, self.dense_layer_size), \
                f"Shape of query embeddings {query_embeddings_neg.shape} is not {(self.n_negative_sample + 1) * self.batch_size}"
            assert item_embeddings.shape == ((self.n_negative_sample + 1) * self.batch_size, self.dense_layer_size), \
                f"Shape of item embeddings {item_embeddings.shape} is not {(self.n_negative_sample + 1) * self.batch_size}"

        # Calculate the dot product
        scores = tf.keras.layers.Dot(axes=1)([query_embeddings_neg, item_embeddings])

        # Scale it with the dot product multiplier
        scores = tf.multiply(scores, self.scaling_constant)

        scores = tf.transpose(
            tf.reshape(tf.transpose(scores), [self.n_negative_sample + 1, self.batch_size]))

        # calculate probability value
        scores = tf.nn.softmax(scores/self.temperature)

        # only extract the first column which represents the positive example prob
        # label shape [dense_layer_size, 1]
        scores = tf.slice(scores, [0, 0], [-1, 1])

        return scores

    def get_config(self):
        config = super(InBatchNegativeSamplingDotProduct, self).get_config()
        config.update({'dense_layer_size': self.dense_layer_size,
                       'n_negative_sample': self.n_negative_sample,
                       'batch_size': self.batch_size,
                       'scaling_constant': self.scaling_constant,
                       'temperature': self.temperature})
        return config


class InBatchNegativeSamplingCosineSimilarity(tf.keras.layers.Layer):
    """
    Custom Keras layer that performs in-batch negative sampling on query and item embeddings and computes cosine similarity.
    Uses negative labels that impact the positive example's probability during softmax computation
    Suited for Retrieval tasks

    :param dense_layer_size: int
        Size of the dense layer after the pooling layer.
    :param n_negative_sample: int
        Number of negative samples to generate for each positive sample.
    :param batch_size: int
        Size of the input batch.
    :param scaling_constant: float, optional
        Constant to smooth the dot product. Default is 1.0.
        1, 5, 10, 20 are good values to try
    :param temperature: float, optional
        Temperature to scale the logits. Default is 1.0.

    Methods
    -------
    call(inputs)
        Method that performs the negative sampling and computes dot product.

    get_config()
        Method to get the configuration of the layer.

    Returns
    -------
    pred: tensor
        A tensor of shape `(dense_layer_size, 1)` representing the positive-example probabilities

    """

    def __init__(self, dense_layer_size, n_negative_sample, batch_size,
                 scaling_constant=1.0, temperature=1.0,
                 **kwargs):
        self.dense_layer_size = dense_layer_size
        self.n_negative_sample = n_negative_sample
        self.batch_size = batch_size
        self.scaling_constant = scaling_constant
        self.temperature = temperature
        super(InBatchNegativeSamplingCosineSimilarity, self).__init__(**kwargs)

        assert self.scaling_constant > 0, "Scaling constant should be greater than 0"
        assert self.temperature > 0, "Temperature should be greater than 0"

    def call(self, inputs):
        """
        Method that performs the negative sampling and computes cosine similarity.

        :param inputs: tuple (query_embedding, item_embedding)
            query_embedding: shape `((n_negative_sample + 1) * batch_size, embedding_dim)`
            item_embedding: shape `((n_negative_sample + 1) * batch_size, embedding_dim)`
        :return: pred: tensor
            Tensor of shape `(dense_layer_size, 1)` representing the predicted scores for each query-item pair.

        """
        query_embeddings, item_embeddings = inputs

        query_embeddings_neg, item_embeddings = InBatchNegativeSampling(self.dense_layer_size,
                                                                        self.n_negative_sample,
                                                                        self.batch_size)([query_embeddings,
                                                                                          item_embeddings])

        # Assert statement to check the shape of the embeddings output from InBatchNegativeSampling
        if query_embeddings_neg.shape[0] is not None and item_embeddings.shape[0] is not None:
            assert query_embeddings_neg.shape == ((self.n_negative_sample + 1) * self.batch_size, self.dense_layer_size), \
                f"Shape of query embeddings {query_embeddings_neg.shape} is not {(self.n_negative_sample + 1) * self.batch_size}"
            assert item_embeddings.shape == ((self.n_negative_sample + 1) * self.batch_size, self.dense_layer_size), \
                f"Shape of item embeddings {item_embeddings.shape} is not {(self.n_negative_sample + 1) * self.batch_size}"

        # Compute the cosine similarity (dot product of normalized embeddings)
        scores = tf.keras.layers.Dot(axes=1, normalize=True)([query_embeddings_neg, item_embeddings])

        # Scale the cosine similarity
        scores = tf.multiply(scores, self.scaling_constant)

        scores = tf.transpose(
            tf.reshape(tf.transpose(scores), [self.n_negative_sample + 1, self.batch_size]))

        # Calculate probability value
        scores = tf.nn.softmax(scores/self.temperature)

        # Only extract the first column which represents the positive example prob
        # Label shape [dense_layer_size, 1]
        scores = tf.slice(scores, [0, 0], [-1, 1])

        return scores

    def get_config(self):
        config = super(InBatchNegativeSamplingCosineSimilarity, self).get_config()
        config.update({'dense_layer_size': self.dense_layer_size,
                       'n_negative_sample': self.n_negative_sample,
                       'batch_size': self.batch_size,
                       'scaling_constant': self.scaling_constant,
                       'temperature': self.temperature})
        return config

