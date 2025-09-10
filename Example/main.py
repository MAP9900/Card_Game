from src.analysis import make_data, save_data, load_data
from src.viz import make_figure

# print(make_data())

data = make_data(n = 10)
# print(data)
# print('------------------')
# save_data(data, 'test.npy')
# data = load_data('test.npy')
# print(data)

make_figure(data, 'test.svg')