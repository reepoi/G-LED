import torch
from torch.utils.data import Dataset


class bfs_dataset(Dataset):
    def __init__(self, data_dir,
                 trajec_max_len=50,
                 start_n=0,
                 n_span=510):
        self.data_dir = data_dir
        assert n_span > trajec_max_len
        self.start_n = start_n
        self.n_span  = n_span
        self.trajec_max_len = trajec_max_len

        # print('load data0')
        # solution0 = np.load(data_location[0],allow_pickle = True)
        # print('load data1')
        # solution1 = np.load(data_location[1],allow_pickle = True)
        # print('cat')
        # solution  = np.concatenate([solution0,
        #                             solution1],axis = 0)
        # print('convert to torch')
        # self.solution = torch.from_numpy(solution[start_n:start_n+n_span])
        assert (self.data_dir/'data0.npy').exists()
        assert (self.data_dir/'data1.npy').exists()
        assert (self.data_dir/'data_cat_f32.pt').exists()
        solution = torch.load(self.data_dir/'data_cat_f32.pt')
        self.solution = solution[start_n:start_n+n_span]

    def __len__(self):
        return self.n_span - self.trajec_max_len

    def __getitem__(self, index):
        item = self.solution[index:index+self.trajec_max_len]
        return item
