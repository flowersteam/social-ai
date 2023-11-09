from multiprocessing import Process, Pipe
import gym

def worker(conn, env):
    while True:
        cmd, data = conn.recv()
        if cmd == "step":
            obs, reward, done, info = env.step(data)
            if done:
                obs = env.reset()
            conn.send((obs, reward, done, info))
        elif cmd == "set_curriculum_parameters":
            env.set_curriculum_parameters(data)
            conn.send(None)
        elif cmd == "reset":
            obs = env.reset()
            conn.send(obs)
        elif cmd == "get_mission":
            ks = env.get_mission()
            conn.send(ks)
        else:
            raise NotImplementedError

class ParallelEnv(gym.Env):
    """A concurrent execution of environments in multiple processes."""

    def __init__(self, envs):
        assert len(envs) >= 1, "No environment given."

        self.envs = envs
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

        if hasattr(self.envs[0], "curriculum"):
            self.curriculum = self.envs[0].curriculum

        self.locals = []
        for env in self.envs[1:]:
            local, remote = Pipe()
            self.locals.append(local)
            p = Process(target=worker, args=(remote, env))
            p.daemon = True
            p.start()
            remote.close()

    def broadcast_curriculum_parameters(self, data):
        # broadcast curriculum_data to every worker
        for local in self.locals:
            local.send(("set_curriculum_parameters", data))
        results = [self.envs[0].set_curriculum_parameters(data)] + [local.recv() for local in self.locals]

    def get_mission(self):
        for local in self.locals:
            local.send(("get_mission", None))
        results = [self.envs[0].get_mission()] + [local.recv() for local in self.locals]
        return results

    def reset(self):
        for local in self.locals:
            local.send(("reset", None))
        results = [self.envs[0].reset()] + [local.recv() for local in self.locals]
        return results

    def step(self, actions):
        for local, action in zip(self.locals, actions[1:]):
            local.send(("step", action))
        obs, reward, done, info = self.envs[0].step(actions[0])
        if done:
            obs = self.envs[0].reset()
        results = zip(*[(obs, reward, done, info)] + [local.recv() for local in self.locals])
        return results

    def render(self):
        raise NotImplementedError