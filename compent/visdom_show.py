import torch
from visdom import Visdom


class My_Visdom():
    def __init__(self, port, env_name = 'main', ):
        super(My_Visdom, self).__init__()
        self.visdom = Visdom(port = port)
        self.record_Y = {}
        self.record_X = {}
        self.env_name = env_name

    def plot_record(self, Y_value, win_name, X_value = None):
        if (isinstance(Y_value, torch.Tensor)):
            Y_value = Y_value.item()
        assert isinstance(Y_value, int) or isinstance(Y_value, float)
        if (win_name not in self.record_Y):
            self.record_Y[win_name] = []
            self.record_X[win_name] = []
        self.record_Y[win_name].append(Y_value)
        if (X_value is None):
            if (len(self.record_X[win_name]) == 0):
                self.record_X[win_name].append(1)
            else:
                self.record_X[win_name].append(self.record_X[win_name][-1] + 1)
        else:
            self.record_X[win_name].append(X_value)
        self.visdom.line(Y = self.record_Y[win_name], X = self.record_X[win_name], win = win_name,
                         opts = dict(title = win_name), env = self.env_name)

    def text(self, text, win_name, append):
        if (append):
            if (self.visdom.win_exists(win = win_name, env = self.env_name)):
                self.visdom.text(text = text, win = win_name, append = True, env = self.env_name,
                                 opts = dict(title = win_name))
            else:
                self.visdom.text(text = text, win = win_name, append = False, env = self.env_name,
                                 opts = dict(title = win_name))
        else:
            self.visdom.text(text = text, win = win_name, append = False, opts = dict(title = win_name),
                             env = self.env_name)

    def table(self, tbl, win_name):
        tbl_str = "<table width=\"100%\"> "
        tbl_str += "<tr> \
                 <th>Term</th> \
                 <th>Value</th> \
                 </tr>"
        for k, v in tbl.items():
            tbl_str += "<tr> \
                       <td>%s</td> \
                       <td>%s</td> \
                       </tr>" % (k, v)

        tbl_str += "</table>"
        default_opts = {'title': win_name}
        self.visdom.text(text = tbl_str, win = win_name, append = False, env = self.env_name, opts = default_opts)

    def clear_record(self, win_name):
        if (win_name in self.record_Y):
            self.record_Y[win_name] = []
            self.record_X[win_name] = []
        else:
            raise Exception('win name not in env')

    def close_all_curves(self):
        self.visdom.close(win = None, env = self.env_name)


if __name__ == '__main__':
    visdom = My_Visdom(env_name = 'main', port = 8888)
    visdom.plot_record(Y_value = 1, win_name = 'hello', X_value = 1)
    visdom.plot_record(Y_value = 10, win_name = 'hello', X_value = 2)

    # import time
    # time.sleep(0.1)™
    # visdom.close_all_curves()
    visdom.text('hello, world,123', 'text 123', True)
    # visdom.text('hello, world,234', 'text 123', True)
