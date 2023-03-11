from xmanager import xm
from xmanager import xm_local
from absl import app
import time
import jax
import os
from experiment_defs import *


def print_and_say(s):
    print(s)
    # os.system('say -v Samantha {}'.format(s))


@xm.run_in_asyncio_loop
async def main(_):
    async with xm_local.create_experiment(experiment_title='hpl_bo') as experiment:
        time_0 = time.time()
        # construct the initial jax random key
        key = jax.random.PRNGKey(RANDOM_SEED)

        print_and_say('run config')

        spec = xm.PythonContainer(
            path='/home/zfan/hyperbo',
            base_image='python:3.7.11',
            entrypoint=xm.CommandList([
                'python3 hyperbo/experiments_xmanager/hpl_bo_split_config.py',
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
            cpu=BASIC_CPU_COUNT,
            ram=4 * BASIC_CPU_COUNT * xm.GiB,
        )

        job = xm.Job(
            executable=executable,
            executor=xm_local.Vertex(requirements=requirements),
            args=None,
        )
        work_unit = await experiment.add(job)
        await work_unit.wait_until_complete()

        time_1 = time.time()

        print_and_say('fit single GP')

        spec = xm.PythonContainer(
            path='/home/zfan/hyperbo',
            base_image='python:3.7.11',
            entrypoint=xm.CommandList([
                'python3 hyperbo/experiments_xmanager/hpl_bo_split_worker.py $@',
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
            cpu=FITTING_NODE_CPU_COUNT,
            ram=4 * FITTING_NODE_CPU_COUNT * xm.GiB,
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

        print_and_say('fit two-step HGP')

        spec = xm.PythonContainer(
            path='/home/zfan/hyperbo',
            base_image='python:3.7.11',
            entrypoint=xm.CommandList([
                'python3 hyperbo/experiments_xmanager/hpl_bo_split_worker.py $@',
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
            cpu=FITTING_NODE_CPU_COUNT,
            ram=4 * FITTING_NODE_CPU_COUNT * xm.GiB,
        )

        work_units = []

        new_key, key = jax.random.split(key)
        job = xm.Job(
            executable=executable,
            executor=xm_local.Vertex(requirements=requirements),
            args={
                'group_id': GROUP_ID,
                'mode': 'fit_direct_hgp_two_step_setup_a',
                'key_0': str(new_key[0]),
                'key_1': str(new_key[1]),
            }
        )
        work_units.append(await experiment.add(job))

        new_key, key = jax.random.split(key)
        job = xm.Job(
            executable=executable,
            executor=xm_local.Vertex(requirements=requirements),
            args={
                'group_id': GROUP_ID,
                'mode': 'fit_direct_hgp_two_step_setup_b',
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
                    'mode': 'fit_direct_hgp_two_step_setup_b_leaveout',
                    'dataset_id': train_id,
                    'key_0': str(new_key[0]),
                    'key_1': str(new_key[1]),
                }
            )
            work_units.append(await experiment.add(job))

        new_key, key = jax.random.split(key)
        job = xm.Job(
            executable=executable,
            executor=xm_local.Vertex(requirements=requirements),
            args={
                'group_id': GROUP_ID,
                'mode': 'fit_hpl_hgp_two_step_setup_a',
                'key_0': str(new_key[0]),
                'key_1': str(new_key[1]),
            }
        )
        work_units.append(await experiment.add(job))

        new_key, key = jax.random.split(key)
        job = xm.Job(
            executable=executable,
            executor=xm_local.Vertex(requirements=requirements),
            args={
                'group_id': GROUP_ID,
                'mode': 'fit_hpl_hgp_two_step_setup_b',
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
                    'mode': 'fit_hpl_hgp_two_step_setup_b_leaveout',
                    'dataset_id': train_id,
                    'key_0': str(new_key[0]),
                    'key_1': str(new_key[1]),
                }
            )
            work_units.append(await experiment.add(job))

        for work_unit in work_units:
            await work_unit.wait_until_complete()

        time_3 = time.time()

        print_and_say('fit end-to-end HGP')

        spec = xm.PythonContainer(
            path='/home/zfan/hyperbo',
            base_image='python:3.7.11',
            entrypoint=xm.CommandList([
                'python3 hyperbo/experiments_xmanager/hpl_bo_split_worker.py $@',
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
            cpu=FITTING_NODE_CPU_COUNT,
            ram=4 * FITTING_NODE_CPU_COUNT * xm.GiB,
        )

        work_units = []

        new_key, key = jax.random.split(key)
        job = xm.Job(
            executable=executable,
            executor=xm_local.Vertex(requirements=requirements),
            args={
                'group_id': GROUP_ID,
                'mode': 'fit_hpl_hgp_end_to_end_setup_a',
                'key_0': str(new_key[0]),
                'key_1': str(new_key[1]),
            }
        )
        work_units.append(await experiment.add(job))

        new_key, key = jax.random.split(key)
        job = xm.Job(
            executable=executable,
            executor=xm_local.Vertex(requirements=requirements),
            args={
                'group_id': GROUP_ID,
                'mode': 'fit_hpl_hgp_end_to_end_setup_b',
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
                    'mode': 'fit_hpl_hgp_end_to_end_setup_b_leaveout',
                    'dataset_id': train_id,
                    'key_0': str(new_key[0]),
                    'key_1': str(new_key[1]),
                }
            )
            work_units.append(await experiment.add(job))

        new_key, key = jax.random.split(key)
        job = xm.Job(
            executable=executable,
            executor=xm_local.Vertex(requirements=requirements),
            args={
                'group_id': GROUP_ID,
                'mode': 'fit_hpl_hgp_end_to_end_setup_a_no_init',
                'key_0': str(new_key[0]),
                'key_1': str(new_key[1]),
            }
        )
        work_units.append(await experiment.add(job))

        new_key, key = jax.random.split(key)
        job = xm.Job(
            executable=executable,
            executor=xm_local.Vertex(requirements=requirements),
            args={
                'group_id': GROUP_ID,
                'mode': 'fit_hpl_hgp_end_to_end_setup_b_no_init',
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
                    'mode': 'fit_hpl_hgp_end_to_end_setup_b_leaveout_no_init',
                    'dataset_id': train_id,
                    'key_0': str(new_key[0]),
                    'key_1': str(new_key[1]),
                }
            )
            work_units.append(await experiment.add(job))

        for work_unit in work_units:
            await work_unit.wait_until_complete()

        time_4 = time.time()

        print_and_say('run BO')

        spec = xm.PythonContainer(
            path='/home/zfan/hyperbo',
            base_image='python:3.7.11',
            entrypoint=xm.CommandList([
                'python3 hyperbo/experiments_xmanager/hpl_bo_split_worker.py $@',
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
            cpu=BO_NODE_CPU_COUNT,
            ram=4 * BO_NODE_CPU_COUNT * xm.GiB,
        )

        work_units = []

        for test_id in TEST_ID_LIST:
            new_key, key = jax.random.split(key)
            job = xm.Job(
                executable=executable,
                executor=xm_local.Vertex(requirements=requirements),
                args={
                    'group_id': GROUP_ID,
                    'mode': 'test_bo_setup_a_id',
                    'dataset_id': test_id,
                    'key_0': str(new_key[0]),
                    'key_1': str(new_key[1]),
                }
            )
            work_units.append(await experiment.add(job))

        for test_id in FULL_ID_LIST:
            new_key, key = jax.random.split(key)
            job = xm.Job(
                executable=executable,
                executor=xm_local.Vertex(requirements=requirements),
                args={
                    'group_id': GROUP_ID,
                    'mode': 'test_bo_setup_b_id',
                    'dataset_id': test_id,
                    'key_0': str(new_key[0]),
                    'key_1': str(new_key[1]),
                }
            )
            work_units.append(await experiment.add(job))

        for work_unit in work_units:
            await work_unit.wait_until_complete()

        time_5 = time.time()

        print_and_say('evaluate NLL')

        spec = xm.PythonContainer(
            path='/home/zfan/hyperbo',
            base_image='python:3.7.11',
            entrypoint=xm.CommandList([
                'python3 hyperbo/experiments_xmanager/hpl_bo_split_worker.py $@',
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
            cpu=NLL_NODE_CPU_COUNT,
            ram=4 * NLL_NODE_CPU_COUNT * xm.GiB,
        )

        work_units = []

        for dataset_id in TRAIN_ID_LIST + TEST_ID_LIST:
            new_key, key = jax.random.split(key)
            job = xm.Job(
                executable=executable,
                executor=xm_local.Vertex(requirements=requirements),
                args={
                    'group_id': GROUP_ID,
                    'mode': 'eval_nll_setup_a_id',
                    'dataset_id': dataset_id,
                    'key_0': str(new_key[0]),
                    'key_1': str(new_key[1]),
                }
            )
            work_units.append(await experiment.add(job))

        for dataset_id in FULL_ID_LIST:
            new_key, key = jax.random.split(key)
            job = xm.Job(
                executable=executable,
                executor=xm_local.Vertex(requirements=requirements),
                args={
                    'group_id': GROUP_ID,
                    'mode': 'eval_nll_setup_b_train_id',
                    'dataset_id': dataset_id,
                    'key_0': str(new_key[0]),
                    'key_1': str(new_key[1]),
                }
            )
            work_units.append(await experiment.add(job))

        for dataset_id in FULL_ID_LIST:
            new_key, key = jax.random.split(key)
            job = xm.Job(
                executable=executable,
                executor=xm_local.Vertex(requirements=requirements),
                args={
                    'group_id': GROUP_ID,
                    'mode': 'eval_nll_setup_b_test_id',
                    'dataset_id': dataset_id,
                    'key_0': str(new_key[0]),
                    'key_1': str(new_key[1]),
                }
            )
            work_units.append(await experiment.add(job))

        for work_unit in work_units:
            await work_unit.wait_until_complete()

        time_6 = time.time()

        print_and_say('merge results')

        spec = xm.PythonContainer(
            path='/home/zfan/hyperbo',
            base_image='python:3.7.11',
            entrypoint=xm.CommandList([
                'python3 hyperbo/experiments_xmanager/hpl_bo_split_worker.py $@',
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
            cpu=BO_NODE_CPU_COUNT,
            ram=4 * BO_NODE_CPU_COUNT * xm.GiB,
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

        time_7 = time.time()

        print_and_say('total time: {}'.format(time_7 - time_0))
        print_and_say('config time: {}'.format(time_1 - time_0))
        print_and_say('fit single GP time: {}'.format(time_2 - time_1))
        print_and_say('fit two-step GP time: {}'.format(time_3 - time_2))
        print_and_say('fit end-to-end GP time: {}'.format(time_4 - time_4))
        print_and_say('run BO time: {}'.format(time_5 - time_4))
        print_and_say('evaluate NLL time: {}'.format(time_6 - time_5))
        print_and_say('merge results time: {}'.format(time_7 - time_6))


if __name__ == '__main__':
    try:
        app.run(main)
    except Exception:
        print_and_say('The experiment has failed.')
        raise
