from multiprocessing import Process, Queue, Pool, current_process, Event
from tools.general_tools import Obj
import time

'''
Example shows how the nested multiprocessing will work in the b2ar workflow.
The parent process will init x processes that will each run a wandb sweep. Each sweep will then
spawn a worker pool and parallel process the resampling methods.
'''
class ChildProcess:
    def __init__(self, timeout):
        self.timeout = timeout

    def run(self, data):
        print(f'Child process executing: {data}')
        time.sleep(self.timeout)
        return data


class ParentProcess:
    def __init__(self, timeout):
        self.timeout = timeout

    def run(self, job_q: Queue, kill_self: Event, timeout_children: bool):
        """
        :param job_q: Queue with a task for the process to complete
        :param kill_e: Event that triggers the process to kill its child process
        :param timeout_children: If true if a sweep process(parent) process timesout then it will
        immediatly timeout the child process pool. If false it will wait for the child process to finish then it
        will terminate both children and parent processes
        """
        j = job_q.get()
        data = j.data

        with Pool(processes=len(data)) as pool:
            print('Creating Child Process')
            child_p = ChildProcess(timeout=10)

            if not timeout_children:
                start = time.time()
                results = pool.map(child_p.run, data)
                print(f'{current_process().name} time: {time.time() - start} val output: {results}')
            else:
                results = pool.map_async(child_p.run, data)

                if kill_self.wait():
                    print('kill Children')
                    pool.terminate()

                print(f'{current_process().name}  val output: {results.get()}')

if __name__ == '__main__':

    jobs = [
        Obj(
            id=0,
            data=[1, 2, 3]
        ),
        Obj(
            id=1,
            data=[4, 5, 6]
        ),
        Obj(
            id=2,
            data=[7, 8, 9]
        )
    ]
    job_q = Queue()
    kill_event = Event()
    processes = []
    for i in range(len(jobs)):
        parent_p = ParentProcess(timeout=60)
        p = Process(target=parent_p.run, args=(job_q, kill_event, False))
        p.daemon = False
        processes.append(p)
        p.start()

    for job in jobs:
        job_q.put(job)
        start = time.time()

    start = time.time()
    time.sleep(5)
    print(f'VALIDATION TIMEOUT PARENTS: Compute time={time.time() - start}')
    kill_event.set()
    for p in processes:
        p.join()

    # output = []
    # while not out_q.empty():
    #     output.append(out_q.get())
    # print(f'Final Output: {output}')

    # children_killed = []
    # while not kill_q.empty():
    #     children_killed.append(kill_q.get())
    #
    # if all(children_killed):
    #     print(children_killed)
    #     print('All children Killed')
    #     for p in processes:
    #         p.terminate()






# def test_func(job_q: Queue):
#
#     signal = job_q.get()
#     if signal is None:
#         # terminate process
#
#
#     print(f'Process {current_process().name} recieved: {data.data}')
#     time.sleep(60)
