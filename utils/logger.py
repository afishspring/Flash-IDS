from gensim.models.callbacks import CallbackAny2Vec

class EpochSaver(CallbackAny2Vec):
    '''Callback to save model after each epoch.'''

    def __init__(self, model_path):
        self.epoch = 0
        self.model_path = model_path

    def on_epoch_end(self, model):
        model.save(self.model_path)
        self.epoch += 1

class EpochLogger(CallbackAny2Vec):
    '''Callback to log information about training'''

    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        print("Epoch #{} start".format(self.epoch))

    def on_epoch_end(self, model):
        print("Epoch #{} end".format(self.epoch))
        self.epoch += 1