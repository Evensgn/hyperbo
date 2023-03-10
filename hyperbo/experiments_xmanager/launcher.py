from xmanager import xm
from xmanager import xm_local
from absl import app
import time
import itertools
import jax
import os
from experiment_defs import *


def print_and_say(s):
    print(s)
    os.system('say -v Samantha {}'.format(s))


@xm.run_in_asyncio_loop
async def main(_):
    async with xm_local.create_experiment(experiment_title='bayesopt') as experiment:
        time_0 = time.time()
        # construct the initial jax random key
        key = jax.random.PRNGKey(RANDOM_SEED)

        print_and_say('run config')

        spec = xm.PythonContainer(
            path='/Users/zfan/code/bo_research/hyperbo',
            base_image='python:3.7.11',
            entrypoint=xm.CommandList([
                'python3 hyperbo/experiments_xmanager/test_hyperbo_plus_split_config.py',
            ]),
            docker_instructions=[
                'WORKDIR bo',
                'COPY hyperbo/hyperbo hyperbo',
                'COPY hyperbo/hpob hpob',
                'COPY hyperbo/setup.py setup.py',
                'RUN pip3 install .',
            ],
        )

        [executable] = experiment.package([
            xm.Packageable(
                executable_spec=spec,
                executor_spec=xm_local.Vertex.Spec(),
            ),
        ])

        requirements = xm.JobRequirements(
            cpu=4,
            ram=4 * xm.GiB,
        )

        job = xm.Job(
            executable=executable,
            executor=xm_local.Vertex(requirements=requirements),
            args=None,
        )
        work_unit = await experiment.add(job)
        await work_unit.wait_until_complete()

        time_1 = time.time()
        
        print_and_say('end-to-end fitting of the hierarchical GP')

        spec = xm.PythonContainer(
            path='/Users/zfan/code/bo_research/hyperbo',
            base_image='python:3.7.11',
            entrypoint=xm.CommandList([
                'python3 hyperbo/experiments_xmanager/test_hyperbo_plus_split_worker_v3.py $@',
            ]),
            docker_instructions=[
                'WORKDIR bo',
                'COPY hyperbo/hyperbo hyperbo',
                'COPY hyperbo/hpob hpob',
                'COPY hyperbo/setup.py setup.py',
                'RUN pip3 install .',
            ],
        )

        [executable] = experiment.package([
            xm.Packageable(
                executable_spec=spec,
                executor_spec=xm_local.Vertex.Spec(),
            ),
        ])

        requirements = xm.JobRequirements(
            cpu=60,
            ram=240 * xm.GiB,
        )

        new_key, key = jax.random.split(key)
        job = xm.Job(
            executable=executable,
            executor=xm_local.Vertex(requirements=requirements),
            args={
                'group_id': GROUP_ID,
                'mode': 'fit_hierarchical_gp_params_e2e_setup_b',
                'dataset_id': '',
                'key_0': str(new_key[0]),
                'key_1': str(new_key[1])
            }
        )
        work_unit = await experiment.add(job)

        await work_unit.wait_until_complete()

        time_2 = time.time()

        print_and_say('running BO')

        spec = xm.PythonContainer(
            path='/Users/zfan/code/bo_research/hyperbo',
            base_image='python:3.7.11',
            entrypoint=xm.CommandList([
                'python3 hyperbo/experiments_xmanager/test_hyperbo_plus_split_worker_v3.py $@',
            ]),
            docker_instructions=[
                'WORKDIR bo',
                'COPY hyperbo/hyperbo hyperbo',
                'COPY hyperbo/hpob hpob',
                'COPY hyperbo/setup.py setup.py',
                'RUN pip3 install .',
            ],
        )

        [executable] = experiment.package([
            xm.Packageable(
                executable_spec=spec,
                executor_spec=xm_local.Vertex.Spec(),
            ),
        ])

        requirements = xm.JobRequirements(
            cpu=10,
            ram=40 * xm.GiB,
        )

        work_units = []

        for test_id in FULL_ID_LIST:
            new_key, key = jax.random.split(key)
            job = xm.Job(
                executable=executable,
                executor=xm_local.Vertex(requirements=requirements),
                args={
                    'group_id': GROUP_ID,
                    'mode': 'test_bo_setup_b_id_temp',
                    'dataset_id': test_id,
                    'key_0': str(new_key[0]),
                    'key_1': str(new_key[1]),
                }
            )
            work_units.append(await experiment.add(job))

        for work_unit in work_units:
            await work_unit.wait_until_complete()

        time_3 = time.time()
        print_and_say('Experiment finished in {} seconds'.format(time_3 - time_0))

        '''
        print_and_say('GP fitting')

        spec = xm.PythonContainer(
            path='/Users/zfan/code/bo_research/hyperbo',
            base_image='python:3.7.11',
            entrypoint=xm.CommandList([
                'python3 hyperbo/experiments_xmanager/test_hyperbo_plus_split_worker_old.py $@',
            ]),
            docker_instructions=[
                'WORKDIR bo',
                'COPY hyperbo/hyperbo hyperbo',
                'COPY hyperbo/hpob hpob',
                'COPY hyperbo/setup.py setup.py',
                'RUN pip3 install .',
            ],
        )

        [executable] = experiment.package([
            xm.Packageable(
                executable_spec=spec,
                executor_spec=xm_local.Vertex.Spec(),
            ),
        ])

        requirements = xm.JobRequirements(
            cpu=20,
            ram=40 * xm.GiB,
        )

        work_units = []

        for train_id in TRAIN_ID_LIST:
            new_key, key = jax.random.split(key)
            job = xm.Job(
                executable=executable,
                executor=xm_local.Vertex(requirements=requirements),
                args={
                    'group_id': GROUP_ID,
                    'mode': 'fit_gp_params_setup_a_id',
                    'dataset_id': train_id,
                    'key_0': str(new_key[0]),
                    'key_1': str(new_key[1]),
                }
            )
            work_units.append(await experiment.add(job))

        for train_id in FULL_ID_LIST:
            new_key, key = jax.random.split(key)
            job = xm.Job(
                executable=executable,
                executor=xm_local.Vertex(requirements=requirements),
                args={
                    'group_id': GROUP_ID,
                    'mode': 'fit_gp_params_setup_b_id',
                    'dataset_id': train_id,
                    'key_0': str(new_key[0]),
                    'key_1': str(new_key[1]),
                }
            )
            work_units.append(await experiment.add(job))

        for work_unit in work_units:
            await work_unit.wait_until_complete()

        time_2 = time.time()

        # fitting the prior distribution

        spec = xm.PythonContainer(
            path='/Users/zfan/code/bo_research/hyperbo',
            base_image='python:3.7.11',
            entrypoint=xm.CommandList([
                'python3 hyperbo/experiments_xmanager/test_hyperbo_plus_split_worker_old.py $@',
            ]),
            docker_instructions=[
                'WORKDIR bo',
                'COPY hyperbo/hyperbo hyperbo',
                'COPY hyperbo/hpob hpob',
                'COPY hyperbo/setup.py setup.py',
                'RUN pip3 install .',
            ],
        )

        [executable] = experiment.package([
            xm.Packageable(
                executable_spec=spec,
                executor_spec=xm_local.Vertex.Spec(),
            ),
        ])

        requirements = xm.JobRequirements(
            cpu=4,
            ram=4 * xm.GiB,
        )

        work_units = []

        new_key, key = jax.random.split(key)
        job = xm.Job(
            executable=executable,
            executor=xm_local.Vertex(requirements=requirements),
            args={
                'group_id': GROUP_ID,
                'mode': 'alpha_mle_setup_a',
                'dataset_id': '',
                'key_0': str(new_key[0]),
                'key_1': str(new_key[1])
            }
        )
        work_units.append(await experiment.add(job))

        new_key, key = jax.random.split(key)
        job = xm.Job(
            executable=executable,
            executor=xm_local.Vertex(requirements=requirements),
            args={
                'group_id': GROUP_ID,
                'mode': 'alpha_mle_setup_b',
                'dataset_id': '',
                'key_0': str(new_key[0]),
                'key_1': str(new_key[1])
            }
        )
        work_units.append(await experiment.add(job))

        for work_unit in work_units:
            await work_unit.wait_until_complete()

        time_3 = time.time()

        print_and_say('merge the results')

        spec = xm.PythonContainer(
            path='/Users/zfan/code/bo_research/hyperbo',
            base_image='python:3.7.11',
            entrypoint=xm.CommandList([
                'python3 hyperbo/experiments_xmanager/test_hyperbo_plus_split_worker_old.py $@',
            ]),
            docker_instructions=[
                'WORKDIR bo',
                'COPY hyperbo/hyperbo hyperbo',
                'COPY hyperbo/hpob hpob',
                'COPY hyperbo/setup.py setup.py',
                'RUN pip3 install .',
            ],
        )

        [executable] = experiment.package([
            xm.Packageable(
                executable_spec=spec,
                executor_spec=xm_local.Vertex.Spec(),
            ),
        ])

        requirements = xm.JobRequirements(
            cpu=4,
            ram=4 * xm.GiB,
        )

        new_key, key = jax.random.split(key)
        job = xm.Job(
            executable=executable,
            executor=xm_local.Vertex(requirements=requirements),
            args={
                'group_id': GROUP_ID,
                'mode': 'merge',
                'dataset_id': '',
                'key_0': str(new_key[0]),
                'key_1': str(new_key[1])
            }
        )
        work_unit = await experiment.add(job)

        await work_unit.wait_until_complete()

        time_4 = time.time()

        print_and_say('The experiment has finished.')
        print('config', time_1 - time_0)
        print('GP fitting', time_2 - time_1)
        print('alpha fitting', time_3 - time_2)
        print('merge', time_4 - time_3)
        '''

if __name__ == '__main__':
    try:
        app.run(main)
    except Exception:
        print_and_say('The experiment has failed.')
        raise
