import abc
import tensorflow as tf
from src.utils import Logger, __fn__, get_envar, read_config, get_timestamp, mkdir
import numpy as np


logger = Logger(__fn__())


class BaseModel(object, metaclass=abc.ABCMeta):

    NAME = 'BaseModel'

    RETRIEVABLES = dict(
        loss='Loss/LOSS',
        regularizer='Loss/REGL',
        train_op='TrainOp/TRAIN_OP',
        acc3='Evaluation/ACC3',
        X='X',
        asp='asp',
        lx='lx',
        y='y',
        dropout_keep='dropout_keep'
    )

    OPTIMIZERS = dict(
        adagrad=tf.train.AdagradOptimizer,
        adam=tf.train.AdamOptimizer,
        sgd=tf.train.GradientDescentOptimizer,
        momentum=tf.train.MomentumOptimizer,
        rmsprop=tf.train.RMSPropOptimizer
    )

    def __init__(self, datamanager=None, parameters=None):
        self.graph = None
        self.sess = None
        self.dm = datamanager
        self.p = parameters


    def train(self, train_df, val=None):

        if self.graph is None:
            self.graph = self.build_graph()

        T = self.retrieve_tensors()

        run_args = [
            T['loss'],
            T['regularizer'],
            T['train_op'],
            T['acc3']
        ]

        placeholders = [
            T['X'],
            T['asp'],
            T['lx'],
            T['y']
        ]

        with tf.Session(graph=self.graph) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            for epoch in range(self.p['epochs']):
                batch_generator = self.dm.batch_generator(train_df, batch_size=self.p['batch_size'], shuffle=self.p['shuffle'])
                epoch_memory = np.zeros([self.dm.n_batches, 2])

                for i,(_, batch) in enumerate(batch_generator):
                    #_X, _asp, _lx, _y = batch
                    loss_, regl_, _, acc3_ = sess.run(run_args, feed_dict=dict(zip(placeholders, batch)))
                    epoch_memory[i,:] = [loss_, acc3_]
                    logger.debug('epoch {epoch:03d}/{epochs:03d}\t'
                                 'batch {i:03d}/{n_batches:03d}\t'
                                 'loss={loss:4.4f}\t'
                                 'l2={l2:4.4f}\t'
                                 'train_acc3={acc:.4%}'
                                 .format(epoch=epoch+1, epochs=self.p['epochs'], i=i, n_batches=self.dm.n_batches,
                                         loss=loss_, acc=acc3_, l2=regl_))

                epoch_loss, epoch_acc = epoch_memory.mean(axis=0)
                logger.info('epoch {epoch:03d}/{epochs:03d}\t'
                            'train_loss={loss:4.4f}\t'
                            'train_acc3={acc:.4%}'
                            .format(epoch=epoch+1, epochs=self.p['epochs'], loss=epoch_loss, acc=epoch_acc))

                if val is not None:
                    _, val_batch = next(self.dm.batch_generator(val, batch_size=-1))
                    # X_val, asp_val, lx_val, y_val = val_batch
                    val_acc3_, val_loss_ = sess.run([T['acc3'], T['loss']], feed_dict=dict(zip(placeholders, val_batch)))
                    logger.info('epoch {epoch:03d}/{epochs:03d}\t'
                                'val_loss={loss:4.4f}\t'
                                'val_acc3={acc:.4%}'
                                .format(epoch=epoch+1, epochs=self.p['epochs'], loss=val_loss_, acc=val_acc3_))
            self.sess = sess

    def retrieve_tensors(self):
        return {k:self.graph.get_tensor_by_name(v+':0') for k,v in self.RETRIEVABLES}


    # def feed_dict(self, placeholders, inputs):
    #     return dict(zip(placeholders, inputs))


    def get_optimizer(self):
        kwargs = dict(
            learning_rate=self.p['learning_rate'],
            momentum=self.p['momentum']
        )
        return self.OPTIMIZERS[self.p['optimizer']](**kwargs)



    def predict(self, X):
        pass



    @abc.abstractmethod
    def build_graph(self):
        # graph = tf.Graph()
        # with graph.as_default():
        #     pass
        # return graph
        raise NotImplementedError


    def load_graph(self, path):
        # TODO: IMPLEMENT
        pass


    def save(self, wdir):
        saver = tf.train.Saver()
        mkdir(wdir)
        saver.save(self.sess, wdir + self.NAME)
        logger.info("Model saved to '{}'".format(wdir))




    def score(self):
        pass

