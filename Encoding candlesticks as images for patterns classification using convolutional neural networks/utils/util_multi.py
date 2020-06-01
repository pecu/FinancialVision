import multiprocessing as mp


def auto_multi(data, tasks_ls, pro_num):
    '''
    Args:
        data (dataframe): csv data after `process_data` function.

    Returns:
        res_ls (dict):
            keys: 'evening', 'morning', ...
            values: [0, 0, 1, 0, 0, ...], [], ...
    '''
    tasks_num = len(tasks_ls.copy())
    
    # Create a queue for all tasks
    q = mp.Queue()

    # Maximum of ps_ls & tmp_ls depends on `pro_num`
    # Maximum of res_ls depends on number of `tasks_num`
    ps_ls = []
    res_ls, tmp_ls = [], []
    while True:
        # termination condition
        if len(res_ls) >= tasks_num:
            break

        if (len(ps_ls) >= pro_num):
            
            for p in ps_ls:
                p.start()

            while True:
                # termination condition
                if len(tmp_ls) >= pro_num:
                    res_ls.extend(tmp_ls)
                    tmp_ls.clear()
                    break

                if q.qsize():
                    tmp_res = q.get()
                    tmp_ls.append(tmp_res)
                    
            for p in ps_ls:
                p.join()

            ps_ls.clear()

        if tasks_ls:
            quotient = len(tasks_ls) // pro_num
            remainder = len(tasks_ls) % pro_num
            if quotient:
                for _ in range(pro_num):
                    task = tasks_ls.pop()
                    ps_ls.append(mp.Process(target=task, args=(data, q, True,)))
            else:
                for _ in range(remainder):
                    task = tasks_ls.pop()
                    ps_ls.append(mp.Process(target=task, args=(data, q, True,)))
                pro_num = remainder
    
    return res_ls