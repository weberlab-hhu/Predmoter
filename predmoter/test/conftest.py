import pytest

from predmoter.core.constants import PREDMOTER_VERSION, GIT_COMMIT


def pytest_addoption(parser):
    parser.addoption("--exclude-device", type=str.lower, action="store", default=None,
                     help="by default the command line tests will be executed on the CPU and the GPU, "
                          "if one of these should not be used (e.g. no GPU is available) it can be excluded")
    parser.addoption("--num-workers", type=int, action="store", default=4,
                     help="number of CPU cores doing the computations (not associated with device number when "
                          "training on the CPU)")
    parser.addoption("--num-devices", type=int, action="store", default=1,
                     help="only tests multiple device training when >1 (test multi device reproducibility by "
                          "adding '--reproducibility-test'), train on multiple GPUs (if available) or "
                          "multiple CPUs (number of CPU cores not CPUs)")
    parser.addoption("--reproducibility-test", action="store_true",
                     help="additionally tests command line training reproducibility")


@pytest.fixture(scope='session', autouse=True)
def exclude_device(request):
    return request.config.getoption("--exclude-device")


@pytest.fixture(scope='session', autouse=True)
def num_workers(request):
    return request.config.getoption("--num-workers")


@pytest.fixture(scope='session', autouse=True)
def num_devices(request):
    return request.config.getoption("--num-devices")


@pytest.fixture(scope='session', autouse=True)
def reproducibility_test(request):
    return request.config.getoption("--reproducibility-test")


def pytest_sessionstart(session):
    print(f"Testing Predmoter v{PREDMOTER_VERSION}. The current commit is {GIT_COMMIT}.")


def pytest_sessionfinish(session, exitstatus):
    print(f"\nTeardown tests. Deleted the temporary directory in Predmoter/predmoter.")
