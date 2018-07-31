import abc
import tensorflow as tf
from src.utils import Logger, __fn__, get_envar, read_config, get_timestamp, mkdir


logger = Logger(__fn__())


class BaseModel(object, metaclass=abc.ABCMeta):

    NAME = 'BaseModel'

    def __init__(self, datamanager=None, parameters=None):
        self.graph = None
        self.dm = datamanager
        self.p = parameters

    def train(self, train_df, val=None):
        if self.graph is None:
            self.graph = self.build_graph()

        # TODO: Refactor to method
        loss = self.graph.get_tensor_by_name('Loss/LOSS:0')
        train_op = self.graph.get_tensor_by_name('TrainOp/TRAIN_OP:0')
        acc3 = self.graph.get_tensor_by_name('Evaluation/ACC3:0')
        X = self.graph.get_tensor_by_name('X:0')
        asp = self.graph.get_tensor_by_name('asp:0')
        y = self.graph.get_tensor_by_name('y:0')
        dropout_keep = self.graph.get_tensor_by_name('dropout_keep:0')

        with tf.Session(graph=self.graph) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            for epoch in range(self.p['epochs']):
                batch_generator = self.dm.batch_generator(train_df, batch_size=self.p['batch_size'], shuffle=self.p['shuffle'])
                for i,(_, batch) in enumerate(batch_generator):
                    _X, _asp, _lx, _y = batch
                    # TODO: Refactor feed_dict
                    _, loss_, acc3_ = sess.run([train_op, loss, acc3],
                                               feed_dict={X: _X,
                                                          asp: _asp,
                                                          y: _y,
                                                          dropout_keep: self.p['dropout_keep_prob']})
                    logger.info('epoch {epoch:03d}/{epochs:03d}\t'
                                'loss={loss:4.4f}\t'
                                'train_acc3={acc:.2%}'
                                .format(epoch=epoch+1, epochs=self.p['epochs'], loss=loss_, acc=acc3_))
                if val is not None:
                    _, val_batch = next(self.dm.batch_generator(val, batch_size=-1))
                    X_val, asp_val, lx_val, y_val = val_batch
                    val_acc3_ = sess.run(acc3, feed_dict={X: X_val, asp: asp_val, y: y_val})
                    logger.info('epoch {epoch:03d}/{epochs:03d} \t'
                                'val_acc3: {va:.2%} \n'
                                .format(epoch=epoch, epochs=self.p['epochs'], va=val_acc3_))
            self.sess = sess




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


