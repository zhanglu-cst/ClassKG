import os
import pickle

from compent.comm import get_rank


class Saver():
    def __init__(self, save_dir, logger = None):
        self.save_dir = save_dir
        self.rank = get_rank()
        self.logger = logger

    def save_to_file(self, obj, filename):
        if (self.rank != 0):
            return
        path = os.path.join(self.save_dir, filename)
        try:
            with open(path, 'wb') as f:
                pickle.dump(obj, f)
            if(self.logger):
                self.logger.info('save success, rank:{},path:{}'.format(self.rank, path))
            else:
                print('save success, rank:{},path:{}'.format(self.rank, path))
        except Exception as e:
            if(self.logger):
                self.logger.info('save fail, info:{}'.format(str(e)))
            else:
                print('save fail, info:{}'.format(str(e)))

    def load_from_file(self, filename):
        path = os.path.join(self.save_dir, filename)
        try:
            with open(path, 'rb') as f:
                obj = pickle.load(f)
            if(self.logger):
                self.logger.info('rank:{}, load success,path:{}'.format(self.rank, path))
            else:
                print('rank:{}, load success,path:{}'.format(self.rank, path))
            return obj
        except Exception as e:
            if(self.logger):
                self.logger.info('load fail, info:{}'.format(str(e)))
            else:
                print('load fail, info:{}'.format(str(e)))
            return None
