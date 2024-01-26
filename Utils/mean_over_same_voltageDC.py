import numpy as np
from matplotlib import pyplot as plt
def mean_over_same_v(time, V, I, take_only_ind=None):
    # I = I[:30]
    # V= V[:30]

    new_time = list()
    new_i = list()
    new_v = list()

    curr_v = V[0]
    new_v.append(curr_v)
    temp_i = [I[0]]
    temp_time = [time[0]]
    for ind in range(1, len(V)):
        v = V[ind]
        if v == curr_v:
            temp_i.append(I[ind])
            temp_time.append(time[ind])
        elif v != curr_v:
            new_v.append(v)
            if take_only_ind is not None:
                new_time.append(temp_time[take_only_ind])
                new_i.append(temp_i[take_only_ind])
            else:
                new_time.append(np.mean(temp_time))
                new_i.append(np.mean(temp_i))

            temp_i = [I[ind]]
            temp_time = [time[ind]]
            curr_v = v
        if ind==(len(V)-1):
            # new_time.append(np.mean(time[ind]))
            # new_i.append(np.mean(temp_i))
            new_time.append(time[-1])
            new_i.append(temp_i[-1])

    # plt.title('{:.2f} V/s'.format(2*np.max(new_v)/(np.max(new_time)/2)))
    # plt.scatter(new_v, new_i, s=.6)
    # plt.show()
    return np.array(new_time), np.array(new_v), np.array(new_i)
